from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import numpy as np
import base64
from model_loader import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
model = load_model("model/best.pt")
if model is None:
    raise RuntimeError("Failed to load the model")

@app.post("/detect/")
async def detect_golfballs(file: UploadFile = File(...)):
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run inference
        results = model(image)[0]
        logger.info(f"Detection completed. Found {len(results.boxes)} objects")
        
        # Process results
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence
            })
        
        # Get the plotted image with bounding boxes
        plot = results.plot()
        
        # Convert numpy array to PIL Image
        img_with_boxes = Image.fromarray(plot)
        
        # Convert image to base64
        buffered = io.BytesIO()
        img_with_boxes.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        logger.info("Successfully processed image and created response")
        return JSONResponse({
            "success": True,
            "detections": detections,
            "image": img_str
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 