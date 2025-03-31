import cv2
import torch
import numpy as np
from utils.model_loader import load_model

class PersonDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess(self, frame):
        """Convert frame to tensor for model input"""
        resized = cv2.resize(frame, (224, 224))
        tensor = torch.tensor(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return tensor

    def detect(self, frame):
        """Detect persons in a frame, return True if detected"""
        tensor_image = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(tensor_image)
        return self.postprocess(output)

    def count_people(self, frame):
        """Count the number of detected persons in a frame"""
        tensor_image = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(tensor_image)
        
        return self.postprocess_count(output)

    def postprocess(self, output):
        threshold = 0.5  # Adjust as needed
        if hasattr(output[0], "boxes") and output[0].boxes is not None and len(output[0].boxes) > 0:
            confs = output[0].boxes.conf
            probability = confs.max().item() if isinstance(confs, torch.Tensor) else max(confs, default=0.0)
        else:
            probability = 0.0  
        return probability > threshold

    def postprocess_count(self, output):
        """Extract number of detected persons"""
        if hasattr(output[0], "boxes") and output[0].boxes is not None:
            return len(output[0].boxes)  # Count the number of detected objects
        return 0  # No detection
