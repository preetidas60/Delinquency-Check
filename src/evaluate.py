import joblib, glob, json
from sklearn.metrics import roc_auc_score
from src.data_prep import choose_dataset
from src.features import build_features

def evaluate():

    # -----------------------------
    # 1. Choose dataset again
    # -----------------------------
    dtype, df = choose_dataset()

    # -----------------------------
    # 2. Load the latest trained model + scaler
    # -----------------------------
    model_files = sorted(glob.glob("models/lgbm_*.pkl"))
    meta_files  = sorted(glob.glob("models/metadata_*.json"))
    scaler_files = sorted(glob.glob("models/scaler_*.pkl"))

    if not model_files:
        raise RuntimeError("❌ No trained LightGBM model found.")

    model_path = model_files[-1]
    meta_path  = meta_files[-1]

    print(f"\n📄 Using model: {model_path}")
    print(f"📄 Using metadata: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    features = meta["features"]

    model = joblib.load(model_path)

    # Load scaler only if applicable
    scaler = None
    if dtype != "amex" and scaler_files:
        scaler_path = scaler_files[-1]
        print(f"📄 Using scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)

    # -----------------------------
    # 3. Prepare features
    # -----------------------------
    X, y, _ = build_features(df)

    X = X[features]  # ensure correct column order

    if scaler is not None:
        X = scaler.transform(X)

    # -----------------------------
    # 4. Compute AUC
    # -----------------------------
    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)

    print(f"\n📊 Evaluation AUC on '{dtype}' dataset: {auc:.4f}")

if __name__ == "__main__":
    evaluate()
