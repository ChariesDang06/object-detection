import cv2
import requests
import time
import os

def send_frame_to_api(frame, api_url="http://127.0.0.1:8000/detect"):
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    response = requests.post(api_url, files=files)
    return response.json()

def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Camera Feed", frame)
        
        response = send_frame_to_api(frame)
        print("API Response:", response)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
