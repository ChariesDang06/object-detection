import cv2
import os
import time  # Import time module for delay
from camera import Camera
from detection.person_detector import PersonDetector
# from detection.pig_detector import PigDetector  # Commented out pig detector import

# Ensure recorded_videos directory exists
RECORD_DIR = "recorded_videos/"
os.makedirs(RECORD_DIR, exist_ok=True)

# Load models
person_detector = PersonDetector("models/person_model.pt")
# pig_detector = PigDetector("models/pig_model.pt")  # Commented out pig detector initialization

# Initialize Camera
cam = Camera(RECORD_DIR)
recording = False  # Flag to check if recording is active
last_person_detected_time = None  # Track the last time a person was detected
BUFFER_TIME = 1  # Buffer time in seconds to wait before stopping recording

while True:
    frame = cam.get_frame()
    if frame is None:
        break

    # Detect persons and pigs
    person_detected = person_detector.detect(frame)
    # pig_detected = pig_detector.detect(frame)  # Commented out pig detection

    # Start recording when a person is detected (only once)
    if person_detected and not recording:
        cam.start_recording()
        recording = True
        last_person_detected_time = time.time()  # Update the last detection time

    # Update the last detection time if a person is still detected
    if person_detected:
        last_person_detected_time = time.time()

    # Stop recording when the person leaves (only once) after the buffer time
    if recording and not person_detected:
        if time.time() - last_person_detected_time > BUFFER_TIME:
            cam.stop_recording()
            recording = False

    # Write frame only if recording is active
    if recording:
        cam.write_frame(frame)

    # Display detection results
    cv2.putText(frame, f"Person: {person_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.putText(frame, f"Pig: {pig_detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Commented out pig detection display
    
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()