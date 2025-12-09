import joblib, shap
import matplotlib.pyplot as plt
from src.data_prep import load_best_dataset
from src.features import build_features

def explain():
    dtype, data = load_best_dataset()
    df = data if dtype != "amex" else data["train_data.csv"]

    X, y, _ = build_features(df)

    model = joblib.load("models/lgbm.pkl")

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    shap.summary_plot(values, X, show=False)
    plt.savefig("shap_summary.png")

    print("Saved shap_summary.png")

if __name__ == "__main__":
    explain()
