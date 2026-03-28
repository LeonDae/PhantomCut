# Campus "Watt-Watch" Intelligent Energy Auditor

An intelligent energy auditor system that utilizes existing CCTV feeds to detect "Empty but Active" rooms and reduces phantom energy loads on campus.

## Overview
This system utilizes computer vision (YOLOv8 & OpenCV) to detect human occupancy and high-energy appliance status. It features a privacy-first "Ghost" mode that blurs faces or extracts only skeletal poses to ensure student privacy. The project includes a FastAPI backend for the logic engine and CV module, and a Streamlit frontend for the Facility Manager Dashboard.

## Project Structure
- `backend/`: FastAPI application containing the YOLOv8 CV module, privacy utilities, and endpoints.
- `frontend/`: Dashboard displaying room status and energy savings.
- `data/`: Placeholders for mock data or video feeds.
- `notebooks/`: Jupyter notebooks for CV experimentation.

## Setup Instructions

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
pip install -r requirements.txt
run app.py
```
