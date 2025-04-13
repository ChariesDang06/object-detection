import cv2
import requests
import threading
import tkinter as tk
import webbrowser

API_URL = "http://127.0.0.1:8000/detect"  # Adjust if needed
STREAM_URL = "http://127.0.0.1:5000/video_feed/0"  # Adjust for your camera ID

def send_frame_to_api(frame):
    """Encodes a frame and sends it to the FastAPI server in a separate thread."""
    def request_thread():
        _, encoded_image = cv2.imencode(".jpg", frame)  # Encode frame as JPEG
        files = {"file": ("frame.jpg", encoded_image.tobytes(), "image/jpeg")}
        
        try:
            response = requests.post(API_URL, files=files, timeout=5)  # 5s timeout
            if response.status_code == 200:
                print("Response:", response.json())
            else:
                print("Error:", response.status_code, response.text)
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
    
    threading.Thread(target=request_thread, daemon=True).start()  # Run API call in background

def open_camera():
    """Opens webcam and sends frames to the API."""
    cap = cv2.VideoCapture(0)  # Open default webcam (change index for external cams)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow("Camera Feed", frame)
        send_frame_to_api(frame)  # Send frame to FastAPI (in a separate thread)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

def watch_stream():
    """Opens the camera stream in a web browser."""
    webbrowser.open(STREAM_URL)

if __name__ == "__main__":
    # Create a simple GUI with a button to watch the stream
    root = tk.Tk()
    root.title("Camera Stream")
    
    watch_button = Button(root, text="Watch Stream", command=watch_stream, font=("Arial", 14))
    watch_button.pack(pady=20)
    
    root.mainloop()
    
    open_camera()
