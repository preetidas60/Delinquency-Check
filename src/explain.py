import shap, joblib
from src.data_prep import load_synthetic
from src.features import build_features
import matplotlib.pyplot as plt

def explain():
    df = load_synthetic()
    X, y, _ = build_features(df)

    model = joblib.load("models/lgbm.pkl")
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    shap.summary_plot(values, X, show=False)
    plt.savefig("shap_summary.png")
    print("Saved to shap_summary.png")

if __name__ == "__main__":
    explain()
