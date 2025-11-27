import os, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from src.data_prep import load_synthetic
from src.features import build_features

MODEL_DIR = "models"

def train():
    df = load_synthetic()
    X, y, features = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:,1])
    print("LR AUC:", lr_auc)

    lgbm = lgb.LGBMClassifier()
    lgbm.fit(X_train, y_train)
    lgbm_auc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1])
    print("LGBM AUC:", lgbm_auc)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
    print("RF AUC:", rf_auc)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lr, f"{MODEL_DIR}/lr.pkl")
    joblib.dump(lgbm, f"{MODEL_DIR}/lgbm.pkl")
    joblib.dump(rf, f"{MODEL_DIR}/rf.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    print("Models saved.")

if __name__ == "__main__":
    train()
