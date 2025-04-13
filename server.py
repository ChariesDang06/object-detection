from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Response, HTTPException
import cv2
import numpy as np
import asyncio
import time
import pymongo
import gridfs
from detection.person_detector import PersonDetector
from bson import ObjectId
from datetime import datetime, timedelta
import urllib.parse
import openai
import os
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Configure CORSMiddleware
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

# Load the person detection model
detector = PersonDetector("models/person_model.pt")

# Global variables for live camera streams
camera_sources = {}
camera_sources = {"LAPTOP_CAM": 0}
camera_streams = {}
recording_buffers = {}
latest_events = {}
BUFFER_TIME = 2  # Seconds before stopping recording for live cameras
frame_intervals = {}
person_counts = {}

# Global variables for the upload-based detection (/detect endpoint)
# "UPLOAD_CAM" is used as an identifier for videos coming from uploaded files.
recording = False
last_person_detected_time = None
video_buffer = []
cameraID = "UPLOAD_CAM"

# ----------------- New Endpoints for Camera -----------------

@app.post("/add_cameras")
async def add_camera(request: Request):
    camera = await request.json()
    # Check if the camera already exists using the provided _id
    existing_camera = db["cameras"].find_one({"_id": camera["_id"]})
    if existing_camera:
        return {"error": "Camera already exists."}
    # Directly insert the camera information into the "cameras" collection
    db["cameras"].insert_one(camera)
    return {"message": "Camera added successfully."}

@app.get("/cameras")
async def get_cameras():
    # Lấy toàn bộ camera 
    cameras = list(db["cameras"].find({}))
    for cam in cameras:
        cam["_id"] = str(cam["_id"])
    return {"cameras": cameras}


# ----------------- Initialization on Startup -------------------

@app.on_event("startup")
async def load_camera_sources():
    """Load camera sources from the database on application startup and include default sources."""
    global camera_sources, camera_streams
    try:
        # Optionally, load additional cameras from the database
        cameras = db["cameras"].find({"is_active": True})
        for camera in cameras:
            cam_id = camera["_id"]
            source = camera["rtsp_url"]
            # Add or update the camera_sources dictionary
            camera_sources[cam_id] = source
        
        # Print the final camera sources dictionary to verify that it's set correctly
        print(f"Loaded camera sources: {camera_sources}")

        # Open each camera stream using OpenCV
        for cam_id, source in camera_sources.items():
            camera_streams[cam_id] = cv2.VideoCapture(source)

    except Exception as e:
        print(f"Error loading camera sources: {e}")

@app.on_event("startup")
async def initialize_buffers_and_detection_vars():
    """Initialize the buffers for live streaming and the globals for upload detection."""
    global recording_buffers, frame_intervals, person_counts
    # Initialize structures for live cameras (if any)
    recording_buffers = {camera: [] for camera in camera_sources}
    frame_intervals = {camera: 0 for camera in camera_sources}
    person_counts = {camera: 0 for camera in camera_sources}
    
    # Initialize detection variables for the upload-based detection endpoint
    global recording, last_person_detected_time, video_buffer, latest_events
    recording = False
    last_person_detected_time = 0  # Ensure a numeric starting point
    video_buffer = []
    # Pre-initialize the UPLOAD_CAM key for the latest events dictionary
    latest_events[cameraID] = {}
    
@app.on_event("startup")
async def start_camera_task():
    asyncio.create_task(capture_frames())

# ----------------- Live Camera Processing -------------------

async def capture_frames():
    global frame_intervals
    while True:
        for cam_id, cap in camera_streams.items():
            ret, frame = cap.read()
            if ret:
                frame_intervals[cam_id] += 1
                # Process frames at specific intervals for performance
                if frame_intervals[cam_id] % 10 in [5, 15]:
                    await process_frame(cam_id, frame)
        await asyncio.sleep(0.05)

async def process_frame(cam_id, frame):
    global latest_events, person_counts, recording, video_buffer, last_person_detected_time, recording_buffers
    global cameraID, BUFFER_TIME
    person_count = detector.count_people(frame)
    prev_count = person_counts.get(cam_id, 0)
    person_counts[cam_id] = person_count
    person_detected = detector.detect(frame)

    if person_detected:
        last_person_detected_time = time.time()
        if not recording:
            recording = True
            video_buffer = []
            print("Recording started")

    if recording:
        video_buffer.append(frame)

    # If no person detected and the buffer time has expired, stop recording and save
    if recording and not person_detected:
        if time.time() - last_person_detected_time > BUFFER_TIME:
            recording = False
            video_id = save_video_to_mongodb(video_buffer)

            latest_event = {
                "event_type": "Human detect",
                "video_recorded": str(video_id),
                "event_time": datetime.now().isoformat(),
                "cameraID": cameraID
            }
            latest_events[cameraID] = latest_event
            events_collection.insert_one(latest_event)
            print("Recording stopped and event saved to MongoDB")


    if person_count > 0:
        recording_buffers[cam_id].append(frame)

    if person_count != prev_count:
        event = {
            "event_type": "Object count changes",
            "previousCount": prev_count,
            "currentCount": person_count,
            "cameraID": cam_id,
            "event_time": datetime.now().isoformat()
        }
        latest_events[cam_id] = event
        events_collection.insert_one(event)

    if prev_count > 0 and person_count == 0:
        event = {
            "event_type": "Object leaving detected",
            "cameraID": cam_id,
            "event_time": datetime.now().isoformat()
        }
        latest_events[cam_id] = event
        events_collection.insert_one(event)

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

# ----------------- Upload Detection Endpoint -------------------

def save_video_to_mongodb(frames):
    if not frames:
        return None

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_filename = f"temp_video_{int(time.time())}.mp4"

    out = cv2.VideoWriter(temp_filename, fourcc, 20.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    with open(temp_filename, "rb") as video_file:
        video_id = fs.put(video_file, filename=f"video_{int(time.time())}.mp4")

    os.remove(temp_filename)
    return video_id

@app.get("/api/videos/{video_id}")
async def stream_video(video_id: str):
    try:
        # Convert string ID to ObjectId
        obj_id = ObjectId(video_id)
    
        # Find the video in MongoDB
        video = events_collection.find_one({"video_recorded": obj_id})
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get the binary data and content type
        video_data = video.get("binaryData")
        content_type = video.get("contentType", "video/mp4")
        
        # Create a file-like object from binary data
        video_stream = io.BytesIO(video_data)
        
        # Return as streaming response
        return StreamingResponse(
            video_stream,
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename={video_id}.mp4",
                "Accept-Ranges": "bytes"
            }
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="Server error")

#======TO DO ===================
#Change dơwnload video to use streaming response select by event_type
# @app.get("/latest_event/{cam_id}")
# async def latest_event(cam_id: str):
#     return {"event": f"Latest event for {cam_id}"}

@app.get("/events")
async def get_events():
    events = list(events_collection.find({}, {"_id": 0}))
    return {"events": events}


#==============REQUEST OPENAI===================

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_weekly_analysis_request, 'cron', day_of_week='sun', hour=23, minute=59)



# Hàm tạo prompt và gửi tới ChatGPT
# def send_weekly_analysis_request():
#     today = datetime.utcnow()
#     last_sunday = today - timedelta(days=today.weekday() + 1)
#     this_sunday = last_sunday + timedelta(days=7)

#     # Lấy các event Count changes trong tuần
#     events = list(events_collection.find({
#         "timestamp": {
#             "$gte": last_sunday,
#             "$lt": this_sunday
#         }
#     }))

#     if not events:
#         print("No count change events this week.")
#         return

#     # Tạo prompt gửi cho ChatGPT
#     prompt = "Dưới đây là danh sách các sự kiện Count changes trong tuần qua. Phân tích và cho biết:\n"\
#              "- Số liệu thay đổi tăng hay giảm? Đâu là dấu hiệu tốt? Đâu là dấu hiệu bất thường?\n"\
#              "- Nếu phát huy tốt thì khuyến khích nông trại làm gì để giữ hoặc nâng cao chất lượng?\n"\
#              "- Nếu bất thường thì nên làm gì để cải thiện trong tuần tới?\n\n"

#     for event in events:
#         prompt += (
#             f"📷 Camera: {event['camera_id']}\n"
#             f"⏱️ Thời gian: {event['timestamp']}\n"
#             f"🔢 Trước: {event['count_before']}, Sau: {event['count_after']}, Thay đổi: {event['change']}\n"
#             f"📝 Ghi chú: {event.get('note', 'Không có ghi chú')}\n\n"
#         )

#     # Gửi tới ChatGPT API
#     openai.api_key = "key"
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "Bạn là chuyên gia phân tích dữ liệu nông nghiệp."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     result = response['choices'][0]['message']['content']

#     # Lưu kết quả vào MongoDB (hoặc trả về cho frontend thông qua API)
#     db["weekly_analysis"].insert_one({
#         "week_start": last_sunday,
#         "week_end": this_sunday,
#         "created_at": datetime.utcnow(),
#         "analysis": result
#     })

#     print("✅ Weekly analysis created and stored.")
def generate_prompt(events: list) -> str:
    if not events:
        return "Không có sự kiện Count changes nào để phân tích."

    prompt = "Dưới đây là danh sách các sự kiện Count changes. Hãy phân tích:\n"\
             "- Số liệu thay đổi tăng hay giảm? Đâu là dấu hiệu tốt? Đâu là dấu hiệu bất thường?\n"\
             "- Nếu phát huy tốt thì khuyến khích nông trại làm gì để giữ hoặc nâng cao chất lượng?\n"\
             "- Nếu bất thường thì nên làm gì để cải thiện trong tuần tới?\n\n"

    for event in events:
        prompt += (
            f"📷 Camera: {event['camera_id']}\n"
            f"⏱️ Thời gian: {event['timestamp']}\n"
            f"🔢 Trước: {event['count_before']}, Sau: {event['count_after']}, Thay đổi: {event['change']}\n"
            f"📝 Ghi chú: {event.get('note', 'Không có ghi chú')}\n\n"
        )

    return prompt


@app.get("/api/weekly-analysis/latest")
def get_latest_analysis():
    latest = db["weekly_analysis"].find_one(sort=[("created_at", -1)])
    if not latest:
        return jsonify({"message": "No analysis yet."}), 404

    return jsonify({
        "week_start": latest["week_start"],
        "week_end": latest["week_end"],
        "analysis": latest["analysis"]
    })



@app.post("/api/analyze-now")
def analyze_now_route(request: Request):
    events = list(events_collection.find({
        "event_type": "Count changes"
    }))

    if not events:
        return jsonify({"message": "Không có sự kiện Count changes nào."}), 404

    openai.api_key = "key"
    response = openai.completions.create(
    model="gpt-4o",
    prompt=generate_prompt(events),
    max_tokens=100
    )
    result = response['choices'][0]['text']

    # result = response['choices'][0]['message']['content']

    return jsonify({
        "analysis": result
    })