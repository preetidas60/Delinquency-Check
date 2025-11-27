import joblib
from sklearn.metrics import roc_auc_score
from src.data_prep import load_synthetic
from src.features import build_features

def evaluate():
    df = load_synthetic()
    X, y, _ = build_features(df)

    lr = joblib.load("models/lr.pkl")
    scaler = joblib.load("models/scaler.pkl")

    Xs = scaler.transform(X)
    auc = roc_auc_score(y, lr.predict_proba(Xs)[:,1])
    print("LR Full AUC:", auc)

if __name__ == "__main__":
    evaluate()
