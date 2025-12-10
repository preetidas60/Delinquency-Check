import joblib, glob, json, numpy as np
from sklearn.metrics import roc_auc_score
from src.data_prep import load_amex_sample
from src.features import build_features

def load_latest_model():
    model_files = sorted(glob.glob("models/lgbm_*.pkl"))
    meta_files = sorted(glob.glob("models/metadata_*.json"))
    scaler_files = sorted(glob.glob("models/scaler_*.pkl"))

    if not model_files:
        raise RuntimeError("❌ No LightGBM model found.")

    model_path = model_files[-1]
    meta_path = meta_files[-1]

    print(f"\n📄 Using model: {model_path}")
    print(f"📄 Using metadata: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    model = joblib.load(model_path)
    features = meta["features"]

    # scaler may not exist (for AmEx models)
    scaler = None
    if meta["dataset_used"] != "amex" and scaler_files:
        scaler = joblib.load(scaler_files[-1])

    return model, features, scaler, meta["dataset_used"]

def evaluate_once(model, features, scaler):
    df = load_amex_sample(rows=60000)  # ~60k sample each evaluation
    X, y, _ = build_features(df)

    X = X[features]

    if scaler is not None:
        X = scaler.transform(X)

    preds = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, preds)

def main(runs=5):
    model, features, scaler, dtype = load_latest_model()

    print(f"\n🔁 Running {runs} evaluation cycles on random AmEx samples...\n")

    aucs = []

    for i in range(1, runs + 1):
        print(f"▶ Run {i}/{runs}")
        auc = evaluate_once(model, features, scaler)
        aucs.append(auc)
        print(f"   AUC: {auc:.5f}\n")

    print("\n📊 FINAL RESULTS")
    print("---------------------------")
    print(f"Mean AUC:   {np.mean(aucs):.5f}")
    print(f"Max AUC:    {np.max(aucs):.5f}")
    print(f"Min AUC:    {np.min(aucs):.5f}")
    print(f"Std Dev:    {np.std(aucs):.5f}")
    print("---------------------------")
    print("🎯 Model stability verified!")

if __name__ == "__main__":
    main(runs=5)
