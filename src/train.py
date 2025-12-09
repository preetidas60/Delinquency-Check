import os, joblib, json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import pandas as pd

from src.data_prep import load_best_dataset
from src.features import build_features

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    dtype, data = load_best_dataset()

    if dtype == "amex":
        df = data.get("train_data.csv") or next(iter(data.values()))
    else:
        df = data

    X, y, feature_names = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:,1])

    lgbm = lgb.LGBMClassifier()
    lgbm.fit(X_train, y_train)
    lgbm_auc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1])

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump(lr, f"{MODEL_DIR}/lr_{timestamp}.pkl")
    joblib.dump(lgbm, f"{MODEL_DIR}/lgbm_{timestamp}.pkl")
    joblib.dump(rf, f"{MODEL_DIR}/rf_{timestamp}.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_{timestamp}.pkl")

    meta = {
        "dataset_used": dtype,
        "timestamp": timestamp,
        "features": feature_names,
        "metrics": {
            "lr_auc": lr_auc,
            "lgbm_auc": lgbm_auc,
            "rf_auc": rf_auc
        }
    }

    with open(f"{MODEL_DIR}/metadata_{timestamp}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    train()
