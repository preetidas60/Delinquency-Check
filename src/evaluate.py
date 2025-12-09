import joblib
from src.data_prep import load_best_dataset
from src.features import build_features
from sklearn.metrics import roc_auc_score

def evaluate():
    dtype, data = load_best_dataset()
    df = data if dtype != "amex" else data["train_data.csv"]

    X, y, _ = build_features(df)

    model_path = "models/lr.pkl"
    scaler_path = "models/scaler.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    Xs = scaler.transform(X)
    auc = roc_auc_score(y, model.predict_proba(Xs)[:,1])

    print("📊 AUC:", auc)

if __name__ == "__main__":
    evaluate()
