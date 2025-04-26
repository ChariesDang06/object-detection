import cv2
import torch
from ultralytics import solutions
from utils.model_loader import load_model

class PigDetector:
    def __init__(self, model_path: str, zone_points: list[tuple[int,int]] = None, zone_model: str = "models/pig_model.pt"):
        # load your base detection model (Ultralytics YOLO, etc)
        self.model = load_model(model_path)

        # optional zone tracker for exit events
        if zone_points:
            self.zone_tracker = solutions.TrackZone(
                region=zone_points,
                model=zone_model,
                show=False
            )
            self.prev_out_count: dict[str,int] = {}
        else:
            self.zone_tracker = None
            self.prev_out_count = {}

    def preprocess(self, frame):
        """Convert BGR image to normalized torch tensor [1,3,H,W]."""
        resized = cv2.resize(frame, (224, 224))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return tensor

    def detect(self, frame) -> bool:
        """
        Return True if at least one pig is detected in this frame.
        """
        tensor = self.preprocess(frame)
        with torch.no_grad():
            outputs = self.model(tensor)
        # handle Ultralytics style Results list:
        if not outputs or not hasattr(outputs[0], "boxes"):
            return False
        boxes = outputs[0].boxes.data
        return boxes.numel() > 0

    def count(self, frame) -> int:
        """
        Return the number of pigs detected in this frame.
        """
        tensor = self.preprocess(frame)
        with torch.no_grad():
            outputs = self.model(tensor)
        if not outputs or not hasattr(outputs[0], "boxes"):
            return 0
        boxes = outputs[0].boxes.data
        # each row is [x1, y1, x2, y2, conf, cls]
        return int(boxes.shape[0])

    def track_zone_exit(self, frame, cam_id: str) -> int:
        """
        If initialized with zone_points, runs TrackZone and returns how many new pigs
        have exited the zone since last call for cam_id.  
        If not initialized with a zone, raises a ValueError.
        """
        if self.zone_tracker is None:
            raise ValueError("PigDetector.zone_points not provided; cannot track zone exits.")

        # run the zone tracker
        results = self.zone_tracker(frame)
        out_count = results.out_count  # cumulative number of exits so far

        # get previous for this camera
        prev = self.prev_out_count.get(cam_id, 0)
        new_exits = out_count - prev

        # update for next time
        self.prev_out_count[cam_id] = out_count

        # only positive differences matter
        return max(0, new_exits)
