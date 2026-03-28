import cv2
import numpy as np
from ultralytics import YOLO
import os


def _enhance_low_light(frame):
    """CLAHE enhancement for dark frames to improve detection accuracy."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    if l_channel.mean() > 120:
        return frame
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# COCO class IDs we care about
PERSON_CLS = 0
APPLIANCE_COCO_MAP = {
    # COCO id → display label
    62: "tv",
    63: "laptop",
    66: "keyboard",
    # ceiling fan + tubelight detected via ApplianceDetector (brightness/motion)
}


class OccupancyDetector:
    def __init__(self, det_model_path=None, pose_model_path=None):
        # Resolve model paths relative to backend/ directory
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        if det_model_path is None:
            det_model_path = os.path.join(backend_dir, 'yolov8n.pt')
        if pose_model_path is None:
            pose_model_path = os.path.join(backend_dir, 'yolov8n-pose.pt')
        
        print(f"[DETECTOR] Loading YOLOv8n detection model from {det_model_path}...")
        self.det_model = YOLO(det_model_path)
        
        print(f"[DETECTOR] Loading YOLOv8n-pose model from {pose_model_path}...")
        self.pose_model = YOLO(pose_model_path)
        
        self.person_conf = 0.35
        self.appliance_conf = 0.30  # Higher threshold = less hallucination
        
        # For grouping detections in the frontend dashboard
        self.appliance_labels = list(APPLIANCE_COCO_MAP.values()) + ["light", "tubelight", "ceiling fan"]
        
        print("[DETECTOR] Models loaded successfully. Detecting: person + appliances + pose")
    
    def detect_frame(self, frame):
        """
        Returns:
          person_count, people_detections [(x1,y1,x2,y2,conf)],
          appliance_count, appliance_detections [(x1,y1,x2,y2,conf,label)],
          appliance_breakdown {label: count},
          keypoints_list [np.array of shape (17,3)] — one per detected person
        """
        enhanced = _enhance_low_light(frame)
        
        # --- 1. Object detection (yolov8n) ---
        det_results = self.det_model(enhanced, verbose=False, conf=0.25)
        
        person_count = 0
        appliance_count = 0
        people_detections = []
        appliance_detections = []
        appliance_breakdown = {}
        
        for r in det_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if cls_id == PERSON_CLS and conf >= self.person_conf:
                    bw = int(x2) - int(x1)
                    bh = int(y2) - int(y1)
                    if bw < 25 or bh < 40:
                        continue
                    person_count += 1
                    people_detections.append((int(x1), int(y1), int(x2), int(y2), conf))
                    
                elif cls_id in APPLIANCE_COCO_MAP and conf >= self.appliance_conf:
                    label = APPLIANCE_COCO_MAP[cls_id]
                    appliance_count += 1
                    appliance_detections.append((int(x1), int(y1), int(x2), int(y2), conf, label))
                    appliance_breakdown[label] = appliance_breakdown.get(label, 0) + 1
        
        # --- 2. Pose estimation (yolov8n-pose) ---
        keypoints_list = []
        try:
            pose_results = self.pose_model(enhanced, verbose=False, conf=0.30)
            for r in pose_results:
                if r.keypoints is not None and r.keypoints.data is not None:
                    for kp in r.keypoints.data:
                        # kp shape: (17, 3) — x, y, confidence per keypoint
                        keypoints_list.append(kp.cpu().numpy())
        except Exception:
            pass  # Pose model failure should not break detection
        
        return person_count, people_detections, appliance_count, appliance_detections, appliance_breakdown, keypoints_list


class ApplianceDetector:
    """Brightness + motion analysis for lights/fans that YOLO can't reliably detect."""
    def __init__(self, history_frames=50):
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=history_frames, varThreshold=50, detectShadows=False)

    def analyze_environment(self, frame, people_detections):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2].mean()
        # Threshold 140: ambient daylight ~80-120, tubelights ON pushes above 140
        light_is_on = brightness > 140
        
        fgMask = self.backSub.apply(frame)
        for (x1, y1, x2, y2, conf) in people_detections:
            cv2.rectangle(fgMask, (x1, y1), (x2, y2), 0, -1)
            
        motion_level = cv2.countNonZero(fgMask)
        fan_is_running = motion_level > 1000
        
        # Build appliance breakdown for dashboard display
        env_breakdown = {}
        env_count = 0
        if light_is_on:
            env_breakdown["tubelight"] = 6  # 6 tubelights in room
            env_count += 6
        if fan_is_running:
            env_breakdown["ceiling fan"] = 1
            env_count += 1
        
        return light_is_on, fan_is_running, brightness, motion_level, env_breakdown, env_count
