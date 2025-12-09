from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib, pandas as pd, glob, json
from src.features import build_features

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model, scaler, features

    model_files = sorted(glob.glob("models/lgbm_*.pkl"))
    scaler_files = sorted(glob.glob("models/scaler_*.pkl"))
    meta_files = sorted(glob.glob("models/metadata_*.json"))

    if not model_files:
        raise RuntimeError("❌ No trained model found.")

    model = joblib.load(model_files[-1])
    scaler = joblib.load(scaler_files[-1])

    with open(meta_files[-1]) as f:
        meta = json.load(f)

    features = meta["features"]

class Input(BaseModel):
    utilisation_pct: float = Field(..., ge=0, le=100)
    avg_payment_ratio: float = Field(..., ge=0, le=100)
    min_due_paid_freq: float = Field(..., ge=0, le=100)
    merchant_mix_index: float = Field(..., ge=0)
    cash_withdrawal_pct: float = Field(..., ge=0, le=100)
    recent_spend_change_pct: float
    credit_limit: float = Field(..., ge=0)

@app.post("/predict")
def predict(data: Input):
    df = pd.DataFrame([data.dict()])
    X, _, _ = build_features(df)

    # order features correctly
    X = X[features]

    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)[0][1]

    return {"prob_default_next_month": float(prob)}
