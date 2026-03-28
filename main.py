from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading
import cv2
import numpy as np
import time
import os
import io
import csv
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient
from app.cv_model.detector import OccupancyDetector, ApplianceDetector
from app.utils.privacy import apply_ghost_mode
from app.utils.notifications import send_energy_alert, get_notification_status
from app.utils.zone_detector import compute_zone_occupancy, get_zone_light_commands
from app.utils.esp8266_controller import (
    update_zones, get_zone_status, check_health,
    get_controller_status, manual_zone_control
)
from app.utils.qr_scanner import get_pin_config, get_zone_labels, reset_config as reset_pin_config

# --- GLOBAL STATE ---
# We store the latest processed CV results in memory so the API can serve it instantly.
# We include one LIVE room hooked up to your camera, and two MOCK rooms for dashboard aesthetics.
global_frames = {0: None, 1: None, 2: None}
camera_status = {0: {"connected": False, "message": "Initializing..."},
                 1: {"connected": False, "message": "Initializing..."},
                 2: {"connected": False, "message": "Initializing..."}}
recording_states = {
    0: {"is_recording": False, "writer": None, "empty_timer": None},
    1: {"is_recording": False, "writer": None, "empty_timer": None},
    2: {"is_recording": False, "writer": None, "empty_timer": None}
}
history_log = []  # Stores timestamped snapshots of room state
MAX_HISTORY = 500  # Keep last 500 entries

# --- MONGODB INIT ---
MONGO_URI = "mongodb+srv://jishnuroy200316_db_user:Frost123@cluster0.unhvyy8.mongodb.net/?appName=Cluster0"
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["wattwatch"]
    history_collection = db["history_logs"]
    print("[INIT] Connected to MongoDB Atlas")
except Exception as e:
    print(f"[ERROR] Failed to connect to MongoDB: {e}")
    history_collection = None

ROOMS_STATE = [
    {
        "id": "Room-CAD LAB",
        "person_count": 0,
        "appliance_state": "OFF",
        "appliance_count": 0,
        "appliance_breakdown": {},
        "alert": False,
        "energy_saved_kwh": 0.0,
        "cumulative_kwh": 0.0,
        "cumulative_cost": 0.0,
        "zone_occupancy": {"zone1": 0, "zone2": 0, "zone3": 0, "zone4": 0},
        "zone_light_states": {"zone1": 1, "zone2": 1, "zone3": 1, "zone4": 1}
    },
    {
        "id": "Room 102 (Class)",
        "person_count": 34,
        "appliance_state": "ON",
        "appliance_count": 6,
        "appliance_breakdown": {"light": 2, "ceiling fan": 2, "projector": 1, "laptop": 1},
        "alert": False,
        "energy_saved_kwh": 14.5
    },
    {
        "id": "Room 103 (Lab)",
        "person_count": 0,
        "appliance_state": "OFF",
        "appliance_count": 0,
        "appliance_breakdown": {},
        "alert": False,
        "energy_saved_kwh": 8.1
    }
]

def generate_status_frame(message="NO CAMERA", sub_message="Waiting for video source..."):
    """Generate a diagnostic frame when no camera is available."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Dark blue background
    frame[:] = (30, 15, 5)
    # Draw border
    cv2.rectangle(frame, (10, 10), (630, 470), (255, 229, 0), 1)
    # Main text
    cv2.putText(frame, message, (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 229, 0), 2)
    # Sub text
    cv2.putText(frame, sub_message, (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # Timestamp
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"WATT-WATCH // {timestamp}", (180, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    return frame


def vision_processing_loop(room_index, video_source):
    """
    This function runs continuously in the background parsing the camera or video file.
    Room 0 = live webcam with ESP8266 zone control + detailed energy tracking.
    Rooms 1-2 = mock video feeds with simplified energy tracking.
    """
    global global_frames, camera_status, recording_states
    print(f"[BACKGROUND] Starting Vision Processing Thread for Room {room_index}...")
    
    try:
        detector = OccupancyDetector()
        appliance_detector = ApplianceDetector()
    except Exception as e:
        print("[WARNING] CV Models failed to initialize:", e)
        camera_status[room_index] = {"connected": False, "message": f"Model init failed: {e}"}
        while True:
            global_frames[room_index] = generate_status_frame("MODEL ERROR", str(e)[:50])
            time.sleep(1)
        return
    
    # Use DirectShow on Windows to prevent MSMF crash spam
    if isinstance(video_source, int) and os.name == 'nt':
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"[WARNING] Video source {video_source} could not be opened.")
        camera_status[room_index] = {"connected": False, "message": "Camera/Video not available"}
        while True:
            global_frames[room_index] = generate_status_frame("NO CAMERA", f"Source {video_source} failed")
            time.sleep(1)
        return
    
    # --- Video Recording Setup ---
    recordings_dir = Path(__file__).resolve().parent.parent / 'recordings'
    recordings_dir.mkdir(exist_ok=True)
    rec_filename = recordings_dir / f"wattwatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps_out = 15.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    video_writer = cv2.VideoWriter(str(rec_filename), fourcc, fps_out, (frame_w, frame_h))
    print(f"[RECORDING] Saving video to {rec_filename}")

    source_label = "Live camera active" if isinstance(video_source, int) else "Video active"
    camera_status[room_index] = {"connected": True, "message": source_label}
    print(f"[BACKGROUND] Source {video_source} opened successfully for Room {room_index}.")
        
    empty_start_time = None
    ALERT_DELAY_SECONDS = 5
    failed_frames = 0
    
    # --- Real Energy Calculation ---
    # 6 tube lights × 20W each = 120W total room load
    ROOM_WATTAGE = 6 * 20            # 120 Watts
    COST_PER_KWH = 7.0               # ₹7 per unit (kWh)
    CO2_PER_KWH = 0.82               # kg CO2 per kWh (Indian grid avg)
    TIME_MULTIPLIER = 200             # Speed up accumulation for demo (1 real sec → 200 simulated sec)
    last_energy_time = time.time()    # Track elapsed time for energy calc
    
    # --- Performance: run YOLO inference every Nth frame, reuse cached results ---
    INFERENCE_INTERVAL = 3  # Run YOLO every 3rd frame
    frame_idx = 0
    cached_person_count = 0
    cached_people_detections = []
    cached_appliance_count = 0
    cached_appliance_detections = []
    cached_appliance_breakdown = {}
    cached_brightness = 0.0
    cached_motion = 0
    cached_light_on = False
    cached_fan_running = False
    cached_keypoints = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(video_source, str):
                # Loop the video file back to the start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                failed_frames += 1
                if failed_frames > 15:
                    print(f"[ERROR] Camera stream {room_index} lost.")
                    camera_status[room_index] = {"connected": False, "message": "Camera stream lost"}
                    cap.release()
                    while True:
                        global_frames[room_index] = generate_status_frame("SIGNAL LOST", "Camera disconnected")
                        time.sleep(1)
                    return
                time.sleep(0.1)
                continue
            
        failed_frames = 0
        frame_idx += 1
            
        # 1. Run HEAVY inference only every Nth frame
        if frame_idx % INFERENCE_INTERVAL == 0:
            result = detector.detect_frame(frame)
            cached_person_count = result[0]
            cached_people_detections = result[1]
            cached_appliance_count = result[2]
            cached_appliance_detections = result[3]
            cached_appliance_breakdown = result[4]
            cached_keypoints = result[5]
            cached_light_on, cached_fan_running, cached_brightness, cached_motion, env_breakdown, env_count = \
                appliance_detector.analyze_environment(frame, cached_people_detections)
            # Merge environment-detected appliances (tubelight, ceiling fan) into the YOLO breakdown
            cached_appliance_breakdown.update(env_breakdown)
            cached_appliance_count += env_count
        
        # Use cached results for every frame (fast)
        person_count = cached_person_count
        people_detections = cached_people_detections
        appliance_count = cached_appliance_count
        
        h, w = frame.shape[:2]
        
        # --- ZONE-BASED OCCUPANCY DETECTION (Room 0 only — live cam with ESP8266) ---
        zone_occupancy = None
        zone_commands = None
        if room_index == 0:
            zone_occupancy = compute_zone_occupancy(people_detections, w, h)
            zone_commands = get_zone_light_commands(zone_occupancy)
            
            # Send zone commands to ESP8266 (only on inference frames to reduce traffic)
            if frame_idx % INFERENCE_INTERVAL == 0:
                sent_states = update_zones(zone_occupancy)
                if sent_states:
                    ROOMS_STATE[0]["zone_light_states"] = sent_states
            
            ROOMS_STATE[0]["zone_occupancy"] = zone_occupancy
        
        # Apply privacy mode (smooth blur + stick figures + appliance labels)
        annotated = apply_ghost_mode(frame, people_detections, cached_appliance_detections, cached_keypoints)
        
        # --- ZONE GRID HUD OVERLAY (Room 0 only) ---
        if room_index == 0 and zone_occupancy is not None:
            mid_x, mid_y = w // 2, h // 2
            # Draw zone divider lines (thin cyan)
            cv2.line(annotated, (mid_x, 0), (mid_x, h), (0, 229, 255), 1, cv2.LINE_AA)
            cv2.line(annotated, (0, mid_y), (w, mid_y), (0, 229, 255), 1, cv2.LINE_AA)
            
            # Draw zone labels with occupancy count
            zone_positions = [
                ("Z1", 10, 30, zone_occupancy["zone1"], zone_commands["zone1"]),
                ("Z2", mid_x + 10, 30, zone_occupancy["zone2"], zone_commands["zone2"]),
                ("Z3", 10, mid_y + 25, zone_occupancy["zone3"], zone_commands["zone3"]),
                ("Z4", mid_x + 10, mid_y + 25, zone_occupancy["zone4"], zone_commands["zone4"]),
            ]
            for label, x, y, count, light_on in zone_positions:
                color = (0, 255, 0) if count > 0 else (0, 0, 255)  # Green=occupied, Red=empty
                icon = "ON" if light_on else "OFF"
                cv2.putText(annotated, f"{label}: {count}P | {icon}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # Add HUD overlay text
        cv2.putText(annotated, f"TOTAL OCCUPANTS: {person_count}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 1, cv2.LINE_AA)
        status_text = "APPLIANCES: ON" if (cached_light_on or cached_fan_running) else "APPLIANCES: OFF"
        cv2.putText(annotated, status_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1, cv2.LINE_AA)
        
        global_frames[room_index] = annotated
        
        # Write annotated frame to continuous recording
        if video_writer is not None:
            video_writer.write(annotated)
        
        appliances_active = cached_light_on or cached_fan_running
        appliance_status_str = "ON" if appliances_active else "OFF"
        
        # 2. Alert logic — room empty but appliances still running
        alert = False
        if person_count == 0 and appliances_active:
            if empty_start_time is None:
                empty_start_time = time.time()
                
            if (time.time() - empty_start_time) >= ALERT_DELAY_SECONDS:
                alert = True
        else:
            empty_start_time = None
            
        # 3. Update the Global State for this room
        ROOMS_STATE[room_index]["person_count"] = person_count
        ROOMS_STATE[room_index]["appliance_state"] = appliance_status_str
        ROOMS_STATE[room_index]["appliance_count"] = appliance_count
        ROOMS_STATE[room_index]["appliance_breakdown"] = cached_appliance_breakdown
        ROOMS_STATE[room_index]["alert"] = alert
        
        # --- Real Energy Accumulation (runs EVERY frame, not just alerts) ---
        now = time.time()
        elapsed_seconds = now - last_energy_time
        last_energy_time = now
        
        if room_index == 0:
            # Detailed energy tracking for the live room
            if appliances_active:
                # 120W = 0.12 kW; energy = power × simulated_time
                sim_seconds = elapsed_seconds * TIME_MULTIPLIER
                energy_this_tick = (ROOM_WATTAGE / 1000.0) * (sim_seconds / 3600.0)  # kWh
                ROOMS_STATE[0]["cumulative_kwh"] += energy_this_tick
                ROOMS_STATE[0]["cumulative_cost"] = round(ROOMS_STATE[0]["cumulative_kwh"] * COST_PER_KWH, 4)
            
            # Track energy wasted (when room empty but appliances ON)
            if alert:
                ROOMS_STATE[0]["energy_saved_kwh"] += (ROOM_WATTAGE / 1000.0) * (elapsed_seconds * TIME_MULTIPLIER / 3600.0)
                # Send Twilio SMS notification on energy drain
                send_energy_alert(
                    room_id=ROOMS_STATE[0]["id"],
                    appliance_breakdown=cached_appliance_breakdown
                )
        else:
            # Simplified energy tracking for mock rooms
            if alert:
                ROOMS_STATE[room_index]["energy_saved_kwh"] += 0.005
        
        # --- History Logging (every ~3 seconds / 100th frame at 30ms interval) ---
        if not hasattr(vision_processing_loop, '_frame_counter'):
            vision_processing_loop._frame_counter = {}
        if room_index not in vision_processing_loop._frame_counter:
            vision_processing_loop._frame_counter[room_index] = 0
            
        vision_processing_loop._frame_counter[room_index] += 1
        
        if vision_processing_loop._frame_counter[room_index] % 100 == 0:
            cumulative_kwh = ROOMS_STATE[0].get("cumulative_kwh", 0)
            cumulative_cost = ROOMS_STATE[0].get("cumulative_cost", 0)
            co2_kg = round(cumulative_kwh * CO2_PER_KWH, 4)
            
            snapshot = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "room": ROOMS_STATE[room_index]["id"],
                "person_count": person_count,
                "appliance_state": appliance_status_str,
                "alert": alert,
                "energy_saved_kwh": round(ROOMS_STATE[room_index]["energy_saved_kwh"], 4),
                "daily_kwh": round(cumulative_kwh, 4),
                "daily_cost": round(cumulative_cost, 2),
                "co2_footprint": round(co2_kg, 3),
                "brightness": round(float(cached_brightness), 1),
                "motion_level": int(cached_motion)
            }
            if history_collection is not None:
                try:
                    history_collection.insert_one(snapshot.copy())
                except Exception as e:
                    pass

            history_log.append(snapshot)
            if len(history_log) > MAX_HISTORY:
                history_log[:] = history_log[-MAX_HISTORY:]
                
        # --- On-Demand Recording Logic ---
        r_state = recording_states[room_index]
        if r_state["is_recording"]:
            if r_state["writer"] is None:
                h, w = frame.shape[:2]
                os.makedirs("data/records", exist_ok=True)
                filename = f"data/records/room{room_index}_{int(time.time())}.mp4"
                fourcc_rec = cv2.VideoWriter_fourcc(*'mp4v')
                r_state["writer"] = cv2.VideoWriter(filename, fourcc_rec, 30.0, (w, h))
                print(f"[RECORDING] Started for room {room_index}: {filename}")
                
            r_state["writer"].write(annotated)
            # Add recording indicator UI to frame
            cv2.circle(annotated, (w-30, 30), 10, (0, 0, 255), -1)
            cv2.putText(annotated, "REC", (w-80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if person_count == 0:
                if r_state["empty_timer"] is None:
                    r_state["empty_timer"] = time.time()
                elif time.time() - r_state["empty_timer"] >= 5.0:
                    r_state["writer"].release()
                    r_state["writer"] = None
                    r_state["is_recording"] = False
                    r_state["empty_timer"] = None
                    print(f"[RECORDING] Stopped automatically for room {room_index} (empty for 5s)")
            else:
                r_state["empty_timer"] = None
            
        # ~30 FPS display rate
        time.sleep(0.03)


# --- API LIFESPAN THREAD SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs when application starts
    # Live Webcam
    threading.Thread(target=vision_processing_loop, args=(0, 0), daemon=True).start()
    # Local video footage feeds
    threading.Thread(target=vision_processing_loop, args=(1, "data_/Room102/room102.mp4"), daemon=True).start()
    threading.Thread(target=vision_processing_loop, args=(2, "data_/Room103/room103.mp4"), daemon=True).start()
    yield
    # Runs when application shutdowns
    print("Shutting down API...")

app = FastAPI(title="Watt-Watch Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINTS ---
@app.get("/")
def read_root():
    return {"status": "Watt-Watch API is live. Background CV processing is active."}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Browsers automatically request a favicon. This blank response stops the 404 terminal spam.
    return Response(content=b"", media_type="image/x-icon")

@app.get("/api/status")
def get_room_status():
    """
    Returns the real-time calculated global state of the rooms securely.
    """
    return {
        "rooms": ROOMS_STATE
    }

@app.get("/api/status/{room_index}")
def get_single_room_status(room_index: int):
    """
    Returns the real-time state of a specific room by index.
    """
    if 0 <= room_index < len(ROOMS_STATE):
        return {"room": ROOMS_STATE[room_index]}
    return {"error": "Invalid room index"}, 400

@app.get("/api/notifications/status")
def notification_status():
    """Returns current Twilio notification configuration status."""
    return get_notification_status()

@app.get("/api/camera_status")
def get_camera_status():
    return camera_status

@app.get("/api/history")
def get_history():
    """
    Returns historical KPI data for the analytics dashboard.
    If no live data yet, returns mock seed data for demo purposes.
    """
    if history_collection is not None:
        try:
            # Get the last 100 entries from DB
            cursor = history_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(100)
            db_history = list(cursor)[::-1]  # reverse to chronological order
            if len(db_history) > 0:
                return {"history": db_history}
        except Exception as e:
            print(f"[ERROR] Failed to fetch history from DB: {e}")

    if len(history_log) < 5:
        # Provide seed data for demo
        seed_data = [
            {"timestamp": "2026-03-26 08:00:00", "room": "Room 101 (Live Cam)", "person_count": 12, "appliance_state": "ON", "alert": False, "energy_saved_kwh": 0.0, "brightness": 145.2, "motion_level": 3200},
            {"timestamp": "2026-03-26 09:15:00", "room": "Room 101 (Live Cam)", "person_count": 0, "appliance_state": "ON", "alert": True, "energy_saved_kwh": 2.1, "brightness": 142.8, "motion_level": 800},
            {"timestamp": "2026-03-26 10:30:00", "room": "Room 102 (Class)", "person_count": 34, "appliance_state": "ON", "alert": False, "energy_saved_kwh": 5.5, "brightness": 160.0, "motion_level": 5500},
            {"timestamp": "2026-03-26 11:45:00", "room": "Room 103 (Lab)", "person_count": 0, "appliance_state": "OFF", "alert": False, "energy_saved_kwh": 8.1, "brightness": 45.3, "motion_level": 120},
            {"timestamp": "2026-03-26 12:00:00", "room": "Room 101 (Live Cam)", "person_count": 5, "appliance_state": "ON", "alert": False, "energy_saved_kwh": 8.1, "brightness": 155.0, "motion_level": 4100},
            {"timestamp": "2026-03-26 13:15:00", "room": "Room 102 (Class)", "person_count": 0, "appliance_state": "ON", "alert": True, "energy_saved_kwh": 10.3, "brightness": 138.5, "motion_level": 950},
            {"timestamp": "2026-03-26 14:30:00", "room": "Room 103 (Lab)", "person_count": 8, "appliance_state": "ON", "alert": False, "energy_saved_kwh": 12.0, "brightness": 170.2, "motion_level": 6200},
            {"timestamp": "2026-03-26 15:00:00", "room": "Room 101 (Live Cam)", "person_count": 0, "appliance_state": "OFF", "alert": False, "energy_saved_kwh": 14.5, "brightness": 30.1, "motion_level": 50},
        ]
        return {"history": seed_data + history_log}
    return {"history": history_log}

@app.get("/api/history/csv")
def get_history_csv():
    """
    Returns historical data as a downloadable CSV file.
    """
    data = get_history()["history"]
    
    output = io.StringIO()
    if data:
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    csv_content = output.getvalue()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=wattwatch_history.csv"}
    )

@app.get("/api/video_feed/{room_index}")
def video_feed(room_index: int):
    def generate():
        while True:
            frame = global_frames.get(room_index)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# --- ESP8266 ZONE CONTROL ENDPOINTS ---

@app.get("/api/esp8266/status")
def esp8266_status():
    """Returns ESP8266 connection status and current zone light states."""
    return get_controller_status()

@app.get("/api/esp8266/health")
def esp8266_health():
    """Heartbeat check for the ESP8266."""
    health = check_health()
    if health:
        return {"reachable": True, **health}
    return {"reachable": False, "error": "ESP8266 not responding"}

@app.get("/api/esp8266/zones")
def esp8266_zones():
    """Get live zone status directly from ESP8266."""
    status = get_zone_status()
    if status:
        return {"source": "esp8266", **status}
    # Fallback to cached state
    return {"source": "cache", **ROOMS_STATE[0].get("zone_light_states", {})}

@app.post("/api/esp8266/zone")
async def esp8266_manual_zone(request_body: dict):
    """
    Manual override for zone lights.
    Body: {"zone1": 1, "zone2": 0, "zone3": 1, "zone4": 0}
    Values: 1 = ON, 0 = OFF
    """
    # Validate input
    valid_keys = {"zone1", "zone2", "zone3", "zone4"}
    zone_states = {k: v for k, v in request_body.items() if k in valid_keys}
    
    if not zone_states:
        return {"error": "Provide at least one zone (zone1-zone4) with value 0 or 1"}
    
    success, response = manual_zone_control(zone_states)
    if success:
        ROOMS_STATE[0]["zone_light_states"].update(zone_states)
        return {"success": True, "zones": response}
    return {"success": False, "error": "Failed to reach ESP8266"}

@app.get("/api/pinconfig")
def api_get_pin_config():
    """Returns the current pin mapping (from QR scan or defaults)."""
    config = get_pin_config()
    labels = get_zone_labels(config)
    return {
        "config": config,
        "zone_labels": labels,
        "scanned_at": config.get("scanned_at", None)
    }

@app.post("/api/pinconfig/reset")
def api_reset_pin_config():
    """Reset pin config so QR scan runs on next startup."""
    success = reset_pin_config()
    return {"success": success, "message": "Pin config reset. Restart server to re-scan QR."}

@app.post("/api/record_video/{room_index}")
def api_record_video(room_index: int):
    if room_index in recording_states:
        recording_states[room_index]["is_recording"] = True
        # Reset the empty timer if it was set
        recording_states[room_index]["empty_timer"] = None
        return {"status": "success", "message": f"Recording started for room {room_index}"}
    return {"status": "error", "message": "Invalid room index"}, 400

@app.get("/api/recording_status/{room_index}")
def get_recording_status(room_index: int):
    if room_index in recording_states:
        return {"is_recording": recording_states[room_index]["is_recording"]}
    return {"error": "Invalid room index"}, 400
