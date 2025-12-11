import joblib, shap, glob, json
import matplotlib.pyplot as plt
from src.data_prep import choose_dataset
from src.features import build_features

def get_latest_model():
    model_files = sorted(glob.glob("models/lgbm_*.pkl"))
    scaler_files = sorted(glob.glob("models/scaler_*.pkl"))
    meta_files = sorted(glob.glob("models/metadata_*.json"))

    if not model_files or not scaler_files:
        raise RuntimeError("❌ No trained LGBM model or scaler found.")

    return model_files[-1], scaler_files[-1], meta_files[-1]

def explain():
    # User chooses dataset
    dtype, df = choose_dataset()

    # Build features
    X, y, _ = build_features(df)

    # Load model + metadata
    model_path, scaler_path, meta_path = get_latest_model()

    print(f"\n📄 Using model: {model_path}")
    print(f"📄 Using metadata: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Ensure correct feature ordering
    feature_order = meta["features"]
    X = X[feature_order]

    model = joblib.load(model_path)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("images/shap_summary.png")

    print("\n📊 SHAP explainability plot saved as shap_summary.png")

if __name__ == "__main__":
    explain()
