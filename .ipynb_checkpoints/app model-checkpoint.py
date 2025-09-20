# app/model.py
from pathlib import Path
import joblib
import numpy as np

BASE = Path(__file__).parent.parent
MODEL_PATH = BASE / "artifact" / "model.pkl"
SCALER_PATH = BASE / "artifact" / "scaler.pkl"

def load_model():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model or scaler not found in artifact/. Run training script first.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    classes = getattr(model, "classes_", None)
    return {"model": model, "scaler": scaler, "classes": classes}

def predict(bundle, features):
    """
    features: list or 1D array of numeric feature values (same order as training CSV columns except 'class')
    returns: dict with prediction and probabilities (if available)
    """
    model = bundle["model"]
    scaler = bundle["scaler"]
    X = np.array(features).reshape(1, -1)
    Xs = scaler.transform(X)
    pred = model.predict(Xs).tolist()[0]
    probs = model.predict_proba(Xs).tolist()[0] if hasattr(model, "predict_proba") else None
    return {"prediction": pred, "probs": probs}
