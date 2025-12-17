from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pickle
import numpy as np
import os
from typing import List

# Load model
MODEL_PATH = 'models/iris_model.pkl'
model = None

class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float 
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: int
    species: str
    confidence: float

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    print("Model loaded successfully.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    load_model()
    yield
    # Shutdown: cleanup if needed
    print("Application shutting down...")

app = FastAPI(
    lifespan=lifespan,
    title="Iris Classifier API", 
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        'message': 'Iris Classifier API', 
        'status': 'running',
        'version': f'{app.version}',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'docs': '/docs'
        }
    }

@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'model_loaded': model is not None
    }

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None: 
        raise HTTPException(status_code=505, detail='Model not loaded')
    
    # Get input features
    features = np.array([[ 
        request.sepal_length, 
        request.sepal_width, 
        request.petal_length, 
        request.petal_width 
    ]])

    # Make prediction
    prediction_probs = model.predict_proba(features)[0]
    prediction = model.predict(features)[0]
    confidence = float(np.max(prediction_probs))

    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

    return PredictionResponse(
        prediction=int(prediction),
        species=species_map[prediction],
        confidence=confidence
    )

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)