from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os, logging
import pandas as pd
from src.features import base_tabular_features

app = FastAPI(title="Early Risk Signals - Delinquency API")
logger = logging.getLogger("uvicorn")

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

class Account(BaseModel):
    utilisation_pct: float
    avg_payment_ratio: float
    min_due_paid_freq: float
    merchant_mix_index: float
    cash_withdrawal_pct: float
    recent_spend_change_pct: float
    credit_limit: float

@app.on_event("startup")
def load_models():
    global model, scaler, feature_names
    model_path = os.path.join(MODEL_DIR, 'lgbm.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    features_path = os.path.join(MODEL_DIR, 'feature_names.json')
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise RuntimeError("Models not found. Run training first.")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    import json
    with open(features_path) as f:
        feature_names = json.load(f)

@app.post("/predict")
def predict(account: Account):
    try:
        df = pd.DataFrame([account.dict()])
        X, feats = base_tabular_features(df)
        # ensure feature ordering matches training
        X = X[[c for c in feature_names if c in X.columns]]
        Xs = scaler.transform(X)
        prob = float(model.predict_proba(Xs)[:,1][0])
        return {"prob_default_next_month": prob}
    except Exception as e:
        logger.exception("Prediction failure")
        raise HTTPException(status_code=500, detail=str(e))
