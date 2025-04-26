from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Response, HTTPException
import cv2
import asyncio
import time
import pymongo
import gridfs
from detection.person_detector import PersonDetector
from detection.pig_detector import PigDetector
from bson import ObjectId
from datetime import datetime
import urllib.parse
import os
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import glob

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
username = urllib.parse.quote_plus("2112727")
password = urllib.parse.quote_plus("admin@123")
MONGO_URI = f"mongodb+srv://{username}:{password}@object-detection.3sedu.mongodb.net/?retryWrites=true&w=majority&appName=object-detection"
sync_client = pymongo.MongoClient(MONGO_URI)
db = sync_client["detection"]
fs = gridfs.GridFS(db)
events_collection = db["events"]
PIG_ZONE_POINTS = [(150, 0), (1130, 0), (1130, 720), (150, 720)]


# Load detectors
detector = PersonDetector("models/person_model.pt")
pig_detector = PigDetector("models/pig_model.pt", zone_points=PIG_ZONE_POINTS)

# Globals
camera_sources = {"LAPTOP_CAM": 0}
camera_streams = {cam: cv2.VideoCapture(src) for cam, src in camera_sources.items()}
recording_buffers = {}
latest_events = {}
BUFFER_TIME = 2
frame_intervals = {}
person_counts = {}
pig_counts = {}


# Upload-based detection vars
recording = False
last_person_detected_time = None
video_buffer = []
cameraID = "UPLOAD_CAM"

# Startup: initialize buffers and tasks
@app.on_event("startup")
async def initialize_all():
    global recording_buffers, frame_intervals, person_counts, pig_counts, latest_events

    # Init live cameras
    for cam in camera_sources:
        recording_buffers[cam] = []
        frame_intervals[cam] = 0
        person_counts[cam] = 0
        pig_counts[cam] = 0
        

    # Init upload cam
    latest_events[cameraID] = {}

    # Launch tasks
    asyncio.create_task(capture_frames())
    asyncio.create_task(feed_images("SIM_CAM"))
    asyncio.create_task(track_pig_cross_line())

async def feed_images(cam_id: str):
    """
    Only pig count events from folder images every 15 seconds.
    """
    folder = os.path.join(os.path.dirname(__file__), "pig-images-feed")
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files.sort()

    while True:
        for img_path in files:
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            prev = pig_counts.get(cam_id, 0)
            count = pig_detector.count(frame)
            pig_counts[cam_id] = count

            if count != prev:
                evt = {
                    "event_type": "Pig count changes",
                    "previousCount": prev,
                    "currentCount": count,
                    "cameraID": cam_id,
                    "event_time": datetime.now().isoformat()
                }
                latest_events[cam_id] = evt
                events_collection.insert_one(evt)
                print(f"[{cam_id}] [FEED] Pig count changed: {prev}→{count}")

            await asyncio.sleep(5.0)
        # Loop files again

async def capture_frames():
    """
    Live camera processing for human detection only.
    """
    while True:
        for cam_id, cap in camera_streams.items():
            ret, frame = cap.read()
            if not ret:
                continue

            frame_intervals[cam_id] += 1
            if frame_intervals[cam_id] % 10 in (5, 15):
                await process_frame(cam_id, frame)
        await asyncio.sleep(0.05)

async def track_pig_cross_line():
    video_path = os.path.join(os.path.dirname(__file__), "pig-cross-line.mp4")
    cap = cv2.VideoCapture(video_path)
    cam_id = "PIG_CROSS_LINE_CAM"

    if not cap.isOpened():
        print("Error: Cannot open pig-cross-line video.")
        return

    # initialize both trackers
    pig_detector.prev_out_count[cam_id] = 0
    pig_counts[cam_id] = 0  # raw-detection buffer

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing pig-cross-line video.")
            break

        # 1) Proper zone-exit detection
        exits = pig_detector.track_zone_exit(frame, cam_id)

        # 2) Fallback: raw-count drop
        prev_cnt = pig_counts.get(cam_id, 0)
        curr_cnt = pig_detector.count(frame)
        pig_counts[cam_id] = curr_cnt
        count_drop = max(0, prev_cnt - curr_cnt)

        total_exits = exits + count_drop

        for _ in range(total_exits):
            # encode this exact frame
            success, jpeg = cv2.imencode('.jpg', frame)
            if success:
                img_bytes = jpeg.tobytes()
                # store snapshot in GridFS
                img_id = fs.put(img_bytes,
                                filename=f"{cam_id}_exit_{int(time.time())}.jpg")
            else:
                img_id = None

            event = {
                "event_type": "Object leaving detected",
                "cameraID":    cam_id,
                "event_time":  datetime.now().isoformat(),
                "snapshot_id": str(img_id)  # reference to the frame in GridFS
            }
            latest_events[cam_id] = event
            events_collection.insert_one(event)
            print(f"[{cam_id}] Object leaving detected! snapshot_id={img_id}")

        await asyncio.sleep(0.05)

    cap.release()
    print("Released pig-cross-line capture.")



async def process_frame(cam_id: str, frame):
    """
    Handle human detection on live streams. Pig events are disabled here.
    """
    global recording, last_person_detected_time, video_buffer, record_start_time

    # Human detection logic
    person_count = detector.count_people(frame)
    prev_p = person_counts.get(cam_id, 0)
    person_counts[cam_id] = person_count
    detected = detector.detect(frame)

    if cam_id == "SIM_CAM":
        return

    if detected:
        last_person_detected_time = time.time()
        if not recording:
            recording = True
            video_buffer = []
            record_start_time = time.time()
            print("Recording started")

    if recording:
        video_buffer.append(frame)

    # Stop recording when person leaves
    if recording and not detected and time.time() - last_person_detected_time > (BUFFER_TIME + 2):
        recording = False
        video_id = save_video_to_mongodb(video_buffer, record_start_time, time.time())
        evt = {
            "event_type": "Human detect",
            "video_recorded": str(video_id),
            "event_time": datetime.now().isoformat(),
            "cameraID": cameraID
        }
        latest_events[cameraID] = evt
        events_collection.insert_one(evt)
        print("Recording stopped and event saved")

# Endpoints
@app.get("/latest_event/{cam_id}")
async def get_latest_event(cam_id: str):
    return latest_events.get(cam_id, {"message": "No detection events recorded yet."})

@app.websocket("/stream/{cam_id}")
async def stream_camera(websocket: WebSocket, cam_id: str):
    await websocket.accept()
    try:
        while True:
            ret, frame = camera_streams[cam_id].read()
            if not ret:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print(f"Client disconnected from {cam_id}")

# Upload detection save function

def save_video_to_mongodb(frames, start_time, end_time):
    if not frames:
        return None

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_filename = f"temp_video_{int(time.time())}.mp4"

    duration = end_time - start_time
    fps = len(frames) / duration if duration > 0 else 20.0
    adjusted_fps = fps * 1.2

    out = cv2.VideoWriter(temp_filename, fourcc, adjusted_fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    with open(temp_filename, "rb") as f:
        video_id = fs.put(f, filename=os.path.basename(temp_filename))
    os.remove(temp_filename)
    return video_id

@app.get("/download/{video_id}")
async def download_video(video_id: str):
    try:
        video_file = fs.get(ObjectId(video_id))
        return Response(
            content=video_file.read(),
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename={video_file.filename}"}
        )
    except gridfs.errors.NoFile:
        raise HTTPException(status_code=404, detail="Video not found")

@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: str):
    def generate():
        while True:
            ret, frame = camera_streams[cam_id].read()
            if not ret:
                continue
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            )
            time.sleep(0.03)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/events")
async def get_events():
    events = list(events_collection.find({}, {"_id": 0}))
    return {"events": events}


