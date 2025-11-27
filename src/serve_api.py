from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd
from src.features import build_features

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model, scaler
    model = joblib.load("models/lgbm.pkl")
    scaler = joblib.load("models/scaler.pkl")

class Input(BaseModel):
    utilisation_pct: float
    avg_payment_ratio: float
    min_due_paid_freq: float
    merchant_mix_index: float
    cash_withdrawal_pct: float
    recent_spend_change_pct: float
    credit_limit: float

@app.post("/predict")
def predict(data: Input):
    df = pd.DataFrame([data.dict()])
    X, _, _ = build_features(df)
    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)[0][1]
    return {"probability": float(prob)}
