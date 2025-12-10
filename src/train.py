import os, joblib, json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from src.data_prep import choose_dataset
from src.features import build_features

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train():

    # -----------------------------
    # 1. User chooses dataset
    # -----------------------------
    dtype, df = choose_dataset()

    # -----------------------------
    # 2. Build ML-ready features
    # -----------------------------
    X, y, feature_names = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 3. StandardScaler ONLY for HDFC/Synthetic datasets
    # -----------------------------
    use_scaler = (dtype != "amex")

    if use_scaler:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
    else:
        scaler = None
        X_train_s = X_train
        X_test_s  = X_test

    # -----------------------------
    # 4. Train Logistic Regression (skip for AmEx)
    # -----------------------------
    lr_auc = None
    if dtype != "amex":
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_s, y_train)
        lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])
    else:
        print("⚠️ Skipping Logistic Regression for AmEx dataset (cannot handle NaNs)")

    # -----------------------------
    # 5. Train LightGBM (best for tabular + NaNs)
    # -----------------------------
    lgbm = lgb.LGBMClassifier()
    lgbm.fit(X_train, y_train)
    lgbm_auc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])

    # -----------------------------
    # 6. Train Random Forest
    # -----------------------------
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    # -----------------------------
    # 7. Save models with versioning
    # -----------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if lr_auc is not None:
        joblib.dump(lr, f"{MODEL_DIR}/lr_{timestamp}.pkl")

    joblib.dump(lgbm, f"{MODEL_DIR}/lgbm_{timestamp}.pkl")
    joblib.dump(rf, f"{MODEL_DIR}/rf_{timestamp}.pkl")

    if scaler is not None:
        joblib.dump(scaler, f"{MODEL_DIR}/scaler_{timestamp}.pkl")

    # -----------------------------
    # 8. Save metadata
    # -----------------------------
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
