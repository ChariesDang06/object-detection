# from fastapi import FastAPI, UploadFile, File, Response, HTTPException, WebSocket
# import cv2
# import numpy as np
# import time
# import os
# import urllib.parse
# import pymongo
# import gridfs
# from detection.person_detector import PersonDetector
# from bson import ObjectId
# from dotenv import load_dotenv
# from datetime import datetime

# app = FastAPI()

# # Load environment variables
# load_dotenv()

# # Encode MongoDB credentials
# username = urllib.parse.quote_plus("2112727")
# password = urllib.parse.quote_plus("admin@123")

# # Construct MongoDB URI
# MONGO_URI = f"mongodb+srv://{username}:{password}@object-detection.3sedu.mongodb.net/?retryWrites=true&w=majority&appName=object-detection"

# # Connect to MongoDB
# sync_client = pymongo.MongoClient(MONGO_URI)
# db = sync_client["video_storage"]
# fs = gridfs.GridFS(db)
# events_collection = db["events"]

# # Load the detection model
# detector = PersonDetector("models/person_model.pt")

# # Recording state
# recording = False
# last_person_detected_time = None
# BUFFER_TIME = 2  # Seconds before stopping recording after last detection
# video_buffer = []
# cameraID = "CAM_001"  # Replace with a dynamic camera ID if needed
# latest_event = {}  # Store latest event for retrieval

# #====================================
# cameras = {}
# # Function to capture frames from a camera
# def generate_frames(camera_id):
#     cap = cv2.VideoCapture(camera_id)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # Encode frame to JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         if ret:
#             # Convert to byte array
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    
#     cap.release()

# @app.route('/video_feed/<camera_id>')
# def video_feed(camera_id):
#     # Check if the camera exists and return its stream
#     if camera_id not in cameras:
#         # Initialize the camera capture in a separate thread
#         cameras[camera_id] = threading.Thread(target=generate_frames, args=(camera_id,))
#         cameras[camera_id].start()
    
#     # Use the camera stream function to provide frames
#     return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')



# #===================================

# @app.post("/detect")
# async def detect_human(file: UploadFile = File(...)):
#     global recording, last_person_detected_time, video_buffer, latest_event

#     # Read uploaded frame
#     contents = await file.read()
#     np_array = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

#     # Detect if a person is in the frame
#     person_detected = detector.detect(frame)

#     if person_detected:
#         last_person_detected_time = time.time()

#         if not recording:
#             recording = True
#             video_buffer = []
#             print("Recording started")

#     if recording:
#         video_buffer.append(frame)

#     if recording and not person_detected:
#         if time.time() - last_person_detected_time > BUFFER_TIME:
#             recording = False
#             video_id = save_video_to_mongodb(video_buffer)

#             latest_event = {
#                 "event_type": "Person Detected",
#                 "message": "Motion detected, video recorded",
#                 "video_recorded": str(video_id),
#                 "event_time": datetime.utcnow().isoformat(),
#                 "cameraID": cameraID
#             }
#             events_collection.insert_one(latest_event)
#             print("Recording stopped and event saved to MongoDB")

#             return latest_event

#     return {"person_detected": person_detected}


# @app.get("/latest_event")
# async def get_latest_event():
#     """Retrieve the latest detection event."""
#     if latest_event:
#         return latest_event
#     return {"message": "No detection events recorded yet."}


# def save_video_to_mongodb(frames):
#     """Saves the recorded frames as a video in MongoDB GridFS."""
#     if not frames:
#         return None

#     height, width, _ = frames[0].shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     temp_filename = f"temp_video_{int(time.time())}.mp4"

#     # Write video to a temporary file
#     out = cv2.VideoWriter(temp_filename, fourcc, 20.0, (width, height))
#     for frame in frames:
#         out.write(frame)
#     out.release()

#     # Store the video in GridFS
#     with open(temp_filename, "rb") as video_file:
#         video_id = fs.put(video_file, filename=f"video_{int(time.time())}.mp4")

#     os.remove(temp_filename)  # Clean up temporary file
#     return video_id


# @app.get("/download/{video_id}")
# async def download_video(video_id: str):
#     """Fetch and return a video file from MongoDB GridFS."""
#     try:
#         video_file = fs.get(ObjectId(video_id))  # Retrieve file from GridFS
#         return Response(content=video_file.read(), media_type="video/mp4", headers={
#             "Content-Disposition": f"attachment; filename={video_file.filename}"
#         })
#     except gridfs.errors.NoFile:
#         raise HTTPException(status_code=404, detail="Video not found")


# @app.get("/events")
# async def get_events():
#     """Retrieve all recorded events."""
#     events = list(events_collection.find({}, {"_id": 0}))
#     return {"events": events}


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)


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
