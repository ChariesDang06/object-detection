from fastapi import FastAPI, UploadFile, File, Response, HTTPException
import cv2
import numpy as np
import time
import os
import pymongo  # Use pymongo for GridFS
import gridfs
from detection.person_detector import PersonDetector
from bson import ObjectId

app = FastAPI()

# Use pymongo instead of motor
MONGO_URI = "mongodb+srv://khoi:jWvEEi7jEdgTjK4p@farmdetectorcluster.7mzcc.mongodb.net/retryWrites=true&w=majority&appName=FarmDetectorCluster"
sync_client = pymongo.MongoClient(MONGO_URI)  # Sync client
db = sync_client["video_storage"]  # Use sync database for GridFS
fs = gridfs.GridFS(db)  # Now this will work correctly!

detector = PersonDetector("models/person_model.pt")
recording = False
last_person_detected_time = None
BUFFER_TIME = 0.5
video_buffer = []

@app.post("/detect")
async def detect_human(file: UploadFile = File(...)):
    global recording, last_person_detected_time, video_buffer

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    person_detected = detector.detect(frame)

    if person_detected and not recording:
        recording = True
        last_person_detected_time = time.time()
        video_buffer = []
        print("Recording started")

    if person_detected:
        last_person_detected_time = time.time()

    if recording and not person_detected:
        if time.time() - last_person_detected_time > BUFFER_TIME:
            recording = False
            video_id = await save_video_to_mongodb(video_buffer)
            print("Recording stopped and saved to MongoDB")
            return {"person_detected": person_detected, "video_id": str(video_id)}

    if recording:
        video_buffer.append(frame)

    return {"person_detected": person_detected}

async def save_video_to_mongodb(frames):
    """Saves the recorded frames as a video in MongoDB GridFS."""
    if not frames:
        return None  

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_filename = "temp_video.mp4"
    
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
    """Fetch and return a video file from MongoDB GridFS."""
    try:
        video_file = fs.get(ObjectId(video_id))  # Retrieve file from GridFS
        return Response(content=video_file.read(), media_type="video/mp4", headers={
            "Content-Disposition": f"attachment; filename={video_file.filename}"
        })
    except gridfs.errors.NoFile:
        raise HTTPException(status_code=404, detail="Video not found")