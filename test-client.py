import cv2
import asyncio
import websockets
import numpy as np
import aiohttp
import time

server_url = "http://127.0.0.1:8000"
stream_url = "ws://127.0.0.1:8000/stream/CAM_001"
detect_url = f"{server_url}/detect"
event_url = f"{server_url}/latest_event/CAM_001"

async def get_latest_event():
    async with aiohttp.ClientSession() as session:
        async with session.get(event_url) as resp:
            return await resp.json()

async def detect_person(frame):
    _, img_encoded = cv2.imencode(".jpg", frame)
    async with aiohttp.ClientSession() as session:
        async with session.post(detect_url, data=img_encoded.tobytes(), headers={"Content-Type": "image/jpeg"}) as resp:
            return await resp.json()

async def stream_camera():
    async with websockets.connect(stream_url) as websocket:
        cv2.namedWindow("Laptop Camera - Click to Watch Stream", cv2.WINDOW_NORMAL)
        recording = False
        video_writer = None
        last_detected_time = None
        BUFFER_TIME = 2  # Stop recording 2 seconds after last detection

        while True:
            frame_bytes = await websocket.recv()
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("Laptop Camera - Click to Watch Stream", frame)

            # Detect person every 10 frames
            if int(time.time() * 10) % 10 == 0:
                result = await detect_person(frame)
                person_detected = result.get("person_detected", False)
                if person_detected:
                    last_detected_time = time.time()
                    if not recording:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter("recorded_video.mp4", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                        recording = True
                        print("Recording started...")
                
            # Handle recording
            if recording:
                video_writer.write(frame)
                if time.time() - last_detected_time > BUFFER_TIME:
                    recording = False
                    video_writer.release()
                    print("Recording stopped.")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(stream_camera())
