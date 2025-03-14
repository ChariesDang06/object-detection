import cv2
import torch
import numpy as np
from utils.model_loader import load_model

class PigDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess(self, frame):
        """Convert frame to tensor for model input"""
        resized = cv2.resize(frame, (224, 224))
        tensor = torch.tensor(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return tensor

    def detect(self, frame):
        """Detect pigs in a frame, return True if detected"""
        tensor_image = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(tensor_image)
        return self.postprocess(output)

    def postprocess(self, output):
        """Process model output (handling Ultralytics Results object)"""
        threshold = 0.5  # Adjust this based on the model's accuracy

        if isinstance(output, list) and len(output) > 0:
            results = output[0]  # Get the first Results object
            
            if results.boxes is not None and hasattr(results.boxes, 'data'):
                boxes = results.boxes.data  # Extract the tensor containing detections
                if boxes.numel() == 0:  # Check if tensor is empty (no detections)
                    return False
                
                confidence_scores = boxes[:, 4]  # Assuming 5th column is confidence
                max_confidence = confidence_scores.max().item()  # Get highest confidence
                
                return max_confidence > threshold  # Return True if a pig is detected
            else:
                return False
        else:
            raise TypeError("Expected a list of Results objects, but got something else.")

