from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
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
import openai
import os

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
# camera_sources = {
#     "CAM_001": 0,  # Example: Local webcam
#     "CAM_002": "rtsp://your_camera_ip"
# }

# camera_streams = {}
# recording_buffers = {camera: [] for camera in camera_sources}
# latest_events = {}
# BUFFER_TIME = 2  # Seconds before stopping recording
# frame_intervals = {camera: 0 for camera in camera_sources}  # Track frames per camera
# person_counts = {camera: 0 for camera in camera_sources}  # Track person count per camera

# # Open cameras
# for cam_id, source in camera_sources.items():
#     camera_streams[cam_id] = cv2.VideoCapture(source)

# Fetch camera sources from the database
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Response, HTTPException
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
import os

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

# Camera sources and stream setup
camera_sources = {}
camera_streams = {}
recording_buffers = {}
latest_events = {}
BUFFER_TIME = 2
frame_intervals = {}
person_counts = {}

# Detection recording control variables (for /detect endpoint)
recording = False
last_person_detected_time = None
video_buffer = []
cameraID = "UPLOAD_CAM"  # Identifier for uploaded video (not live stream)

@app.on_event("startup")
async def load_camera_sources():
    """Load camera sources from the database on application startup."""
    global camera_sources
    try:
        cameras = db["cameras"].find({"is_active": True})
        for camera in cameras:
            cam_id = camera["_id"]
            source = camera["rtsp_url"]
            camera_sources[cam_id] = source

        for cam_id, source in camera_sources.items():
            camera_streams[cam_id] = cv2.VideoCapture(source)

        print(f"Loaded camera sources: {camera_sources}")
    except Exception as e:
        print(f"Error loading camera sources: {e}")

@app.on_event("startup")
async def initialize_buffers():
    global recording_buffers, frame_intervals, person_counts
    recording_buffers = {camera: [] for camera in camera_sources}
    frame_intervals = {camera: 0 for camera in camera_sources}
    person_counts = {camera: 0 for camera in camera_sources}

@app.on_event("startup")
async def start_camera_task():
    asyncio.create_task(capture_frames())

async def capture_frames():
    global frame_intervals
    while True:
        for cam_id, cap in camera_streams.items():
            ret, frame = cap.read()
            if ret:
                frame_intervals[cam_id] += 1
                if frame_intervals[cam_id] % 10 in [5, 15]:
                    await process_frame(cam_id, frame)
        await asyncio.sleep(0.05)

async def process_frame(cam_id, frame):
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




@app.post("/detect")
async def detect_human(file: UploadFile = File(...)):
    global recording, last_person_detected_time, video_buffer, latest_events

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    person_detected = detector.detect(frame)

    if person_detected:
        last_person_detected_time = time.time()
        if not recording:
            recording = True
            video_buffer = []
            print("Recording started")

    if recording:
        video_buffer.append(frame)

    if recording and not person_detected:
        if time.time() - last_person_detected_time > BUFFER_TIME:
            recording = False
            video_id = save_video_to_mongodb(video_buffer)

            latest_event = {
                "event_type": "Person Detected",
                "message": "Motion detected, video recorded",
                "video_recorded": str(video_id),
                "event_time": datetime.utcnow().isoformat(),
                "cameraID": cameraID
            }
            latest_events[cameraID] = latest_event
            events_collection.insert_one(latest_event)
            print("Recording stopped and event saved to MongoDB")

            return latest_event

    return {"person_detected": person_detected}

@app.get("/latest_event")
async def get_latest_event_upload():
    if latest_events.get(cameraID):
        return latest_events[cameraID]
    return {"message": "No detection events recorded yet."}

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

@app.get("/download/{video_id}")
async def download_video(video_id: str):
    try:
        video_file = fs.get(ObjectId(video_id))
        return Response(content=video_file.read(), media_type="video/mp4", headers={
            "Content-Disposition": f"attachment; filename={video_file.filename}"
        })
    except gridfs.errors.NoFile:
        raise HTTPException(status_code=404, detail="Video not found")

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
#     openai.api_key = "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"
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

    
    # prompt = generate_prompt(events)

    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": "Bạn là chuyên gia phân tích dữ liệu nông nghiệp."},
    #         {"role": "user", "content": prompt}
    #     ]
    # )

    openai.api_key = "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"
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