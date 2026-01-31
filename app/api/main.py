from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import create_densenet121
from src.inference import PneumoniaPredictor
from src.utils import Config

app = FastAPI(title="Pneumo AI API", version="1.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
config = Config()
device = 'cpu' # Vercel uses CPU
model = create_densenet121(num_classes=3, pretrained=False)
predictor = PneumoniaPredictor(model=model, device=device)

@app.get("/")
async def root():
    return {"message": "Pneumo AI API is running", "version": "1.1.0"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)
    
    result = predictor.predict(image_np, return_heatmap=False)
    
    return {
        "prediction": result['predicted_label'],
        "confidence": result['confidence'],
        "interpretation": result['clinical_metrics']['interpretation']
    }
