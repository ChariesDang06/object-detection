from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import time
import os
from detection.person_detector import PersonDetector

from firebase_admin import credentials, initialize_app, storage
# Initialize Firebase
cred = credentials.Certificate("firebase_config.json")

initialize_app(cred, {'storageBucket': 'your-bucket-name.appspot.com'})

app = FastAPI()

detector = PersonDetector("models/person_model.pt")
recording = False
last_person_detected_time = None
BUFFER_TIME = 1
video_writer = None
video_filename = None
SAVE_PATH = "recorded_videos/"
os.makedirs(SAVE_PATH, exist_ok=True)

@app.post("/detect")
async def detect_human(file: UploadFile = File(...)):
    global recording, last_person_detected_time, video_writer, video_filename
    
    # Read image
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    # Detect human
    person_detected = detector.detect(frame)
    
    if person_detected and not recording:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        video_filename = os.path.join(SAVE_PATH, f"video_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
        recording = True
        last_person_detected_time = time.time()
        print("Recording started")
    
    if person_detected:
        last_person_detected_time = time.time()
    
    if recording and not person_detected:
        if time.time() - last_person_detected_time > BUFFER_TIME:
            video_writer.release()
            upload_to_firebase(video_filename)
            recording = False
            print("Recording stopped and uploaded")
    
    if recording:
        video_writer.write(frame)
    
    return {"person_detected": person_detected}


def upload_to_firebase(file_path):
    bucket = storage.bucket()
    blob = bucket.blob(os.path.basename(file_path))
    blob.upload_from_filename(file_path)
    video_url = blob.public_url
    event_data = {
        "event_type": "human_detected",
        "message": "🚨 Alert: Human detected in camera zone!",
        "video_url": video_url
    }
    print("Uploaded to Firebase:", event_data)
    return event_data