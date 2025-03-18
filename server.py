from fastapi import FastAPI, UploadFile, File, Response, HTTPException
import cv2
import numpy as np
import time
import os
import urllib.parse
import pymongo
import gridfs
from detection.person_detector import PersonDetector
from bson import ObjectId
from dotenv import load_dotenv
from datetime import datetime

app = FastAPI()

# Load environment variables
load_dotenv()

# Encode MongoDB credentials
username = urllib.parse.quote_plus("2112727")
password = urllib.parse.quote_plus("admin@123")

# Construct MongoDB URI
MONGO_URI = f"mongodb+srv://{username}:{password}@object-detection.3sedu.mongodb.net/?retryWrites=true&w=majority&appName=object-detection"

# Connect to MongoDB
sync_client = pymongo.MongoClient(MONGO_URI)
db = sync_client["video_storage"]
fs = gridfs.GridFS(db)
events_collection = db["events"]

# Load the detection model
detector = PersonDetector("models/person_model.pt")

# Recording state
recording = False
last_person_detected_time = None
BUFFER_TIME = 2  # Seconds before stopping recording after last detection
video_buffer = []
cameraID = "CAM_001"  # Replace with a dynamic camera ID if needed
latest_event = {}  # Store latest event for retrieval


@app.post("/detect")
async def detect_human(file: UploadFile = File(...)):
    global recording, last_person_detected_time, video_buffer, latest_event

    # Read uploaded frame
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Detect if a person is in the frame
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
            events_collection.insert_one(latest_event)
            print("Recording stopped and event saved to MongoDB")

            return latest_event

    return {"person_detected": person_detected}


@app.get("/latest_event")
async def get_latest_event():
    """Retrieve the latest detection event."""
    if latest_event:
        return latest_event
    return {"message": "No detection events recorded yet."}


def save_video_to_mongodb(frames):
    """Saves the recorded frames as a video in MongoDB GridFS."""
    if not frames:
        return None

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_filename = f"temp_video_{int(time.time())}.mp4"

    # Write video to a temporary file
    out = cv2.VideoWriter(temp_filename, fourcc, 20.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    # Store the video in GridFS
    with open(temp_filename, "rb") as video_file:
        video_id = fs.put(video_file, filename=f"video_{int(time.time())}.mp4")

    os.remove(temp_filename)  # Clean up temporary file
    return video_id


@app.get("/download/{video_id}")
async def download_video(video_id: str):
    """Fetch and return a video file from MongoDB GridFS."""
    try:
        video_file = fs.get(ObjectId(video_id))  # Retrieve file from GridFS
        return Response(content=video_file.read(), media_type="video/mp4", headers={
            "Content-Disposition": f"attachment; filename={video_file.filename}"
        })
    except gridfs.errors.NoFile:
        raise HTTPException(status_code=404, detail="Video not found")


@app.get("/events")
async def get_events():
    """Retrieve all recorded events."""
    events = list(events_collection.find({}, {"_id": 0}))
    return {"events": events}
