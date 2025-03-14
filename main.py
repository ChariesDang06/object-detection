import cv2
import os
from camera import Camera
from detection.person_detector import PersonDetector
from detection.pig_detector import PigDetector

# Ensure recorded_videos directory exists
RECORD_DIR = "recorded_videos/"
os.makedirs(RECORD_DIR, exist_ok=True)

# Load models
person_detector = PersonDetector("models/person_model.pt")
pig_detector = PigDetector("models/pig_model.pt")

# Initialize Camera
cam = Camera(RECORD_DIR)
recording = False  # Flag to check if we are currently recording

while True:
    frame = cam.get_frame()
    if frame is None:
        break

    # Detect persons and pigs
    person_detected = person_detector.detect(frame)
    pig_detected = pig_detector.detect(frame)

    if person_detected and not recording:
        cam.start_recording()
        recording = True

    if recording:
        cam.write_frame(frame)

    # If person leaves, stop recording
    if not person_detected and recording:
        cam.stop_recording()
        recording = False

    # Display detection results
    cv2.putText(frame, f"Person: {person_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Pig: {pig_detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
