from ultralytics import YOLO  # For YOLOv8
import torch

def load_model(model_path):
    model = torch.load(model_path, map_location="cpu")  # Ensure correct device
    if isinstance(model, dict):  # Ensure it's a model, not a state_dict
        from ultralytics import YOLO
        model = YOLO(model_path)  # Load the YOLO model
    return model
