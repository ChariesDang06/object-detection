from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import asyncio
import time
import pymongo
import gridfs
from detection.person_detector import PersonDetector
from bson import ObjectId
from datetime import datetime
import urllib.parse

app = FastAPI()

# MongoDB setup
username = urllib.parse.quote_plus("2112727")
password = urllib.parse.quote_plus("admin@123")
MONGO_URI = f"mongodb+srv://{username}:{password}@object-detection.3sedu.mongodb.net/?retryWrites=true&w=majority&appName=object-detection"
sync_client = pymongo.MongoClient(MONGO_URI)
db = sync_client["video_storage"]
fs = gridfs.GridFS(db)
events_collection = db["events"]

# Load the person detection model
detector = PersonDetector("models/person_model.pt")

# Camera streams (replace with actual URLs or indexes)
camera_sources = {
    "CAM_001": 0,  # Example: Local webcam
    "CAM_002": "rtsp://your_camera_ip"
}

camera_streams = {}
recording_buffers = {camera: [] for camera in camera_sources}
latest_events = {}
BUFFER_TIME = 2  # Seconds before stopping recording
frame_intervals = {camera: 0 for camera in camera_sources}  # Track frames per camera
person_counts = {camera: 0 for camera in camera_sources}  # Track person count per camera

# Open cameras
for cam_id, source in camera_sources.items():
    camera_streams[cam_id] = cv2.VideoCapture(source)

async def capture_frames():
    """Continuously capture frames from multiple cameras."""
    global frame_intervals
    while True:
        for cam_id, cap in camera_streams.items():
            ret, frame = cap.read()
            if ret:
                frame_intervals[cam_id] += 1
                if frame_intervals[cam_id] % 10 in [5, 15]:  # 5th and 15th frame
                    await process_frame(cam_id, frame)
        await asyncio.sleep(0.05)  # 20 FPS (adjust based on camera speed)

async def process_frame(cam_id, frame):
    """Process selected frames for person detection."""
    global latest_events, person_counts
    person_count = detector.count_people(frame)
    prev_count = person_counts[cam_id]
    person_counts[cam_id] = person_count

    if person_count > 0:
        recording_buffers[cam_id].append(frame)

    if person_count != prev_count:
        event = {
            "event_type": "Count changes",
            "message": "Count changes",
            "previousCount": prev_count,
            "currentCount": person_count,
            "cameraID": cam_id,
            "event_time": datetime.utcnow().isoformat()
        }
        latest_events[cam_id] = event
        events_collection.insert_one(event)

    if prev_count > 0 and person_count == 0:
        event = {
            "event_type": "Object leaving detected",
            "message": "Object leaving detected",
            "cameraID": cam_id,
            "event_time": datetime.utcnow().isoformat()
        }
        latest_events[cam_id] = event
        events_collection.insert_one(event)

@app.get("/latest_event/{cam_id}")
async def get_latest_event(cam_id: str):
    return latest_events.get(cam_id, {"message": "No detection events recorded yet."})

@app.websocket("/stream/{cam_id}")
async def stream_camera(websocket: WebSocket, cam_id: str):
    """Stream camera feed to WebSocket clients."""
    await websocket.accept()
    try:
        while True:
            ret, frame = camera_streams[cam_id].read()
            if not ret:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.05)  # Limit FPS
    except WebSocketDisconnect:
        print(f"Client disconnected from {cam_id}")

@app.on_event("startup")
async def start_camera_task():
    asyncio.create_task(capture_frames())
