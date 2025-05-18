import torch
from ultralytics import YOLO
from typing import Optional
import os

def load_model(model_path: str) -> Optional[YOLO]:
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file not found at: {os.path.abspath(model_path)}")
            return None

        print(f"Attempting to load model from: {os.path.abspath(model_path)}")
        
        # Try loading with default settings first
        try:
            print("Attempting to load model with default settings...")
            model = YOLO(model_path)
            print("Model loaded successfully with default settings")
            return model
        except Exception as e1:
            print(f"Failed to load with default settings: {str(e1)}")
            
            # Try alternative loading method
            print("Attempting alternative loading method...")
            try:
                ckpt = torch.load(model_path, weights_only=False, map_location='cpu')
                model = YOLO(task='detect')
                model.model.load_state_dict(ckpt['model'].state_dict())
                print("Model loaded successfully with alternative method")
                return model
            except Exception as e2:
                print(f"Failed alternative loading method: {str(e2)}")
                return None
                
    except Exception as e:
        print(f"Unexpected error loading model: {str(e)}")
        return None 