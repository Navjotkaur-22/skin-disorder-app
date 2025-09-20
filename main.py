# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import load_model, predict

app = FastAPI(title="Skin Disorder Detector (tabular)")

# load once
MODEL_BUNDLE = load_model()

class FeaturesRequest(BaseModel):
    features: list  # list of numeric feature values (length must match training)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(req: FeaturesRequest):
    try:
        res = predict(MODEL_BUNDLE, req.features)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
