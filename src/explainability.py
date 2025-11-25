import joblib, os
import shap
from src.features import base_tabular_features
from src.data_prep import load_synthetic

REPORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

def explain(model_name='lgbm.pkl'):
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', model_name)
    model = joblib.load(model_path)
    df = load_synthetic()
    df = df.dropna(subset=['dpd_next_month'])
    X, feat_names = base_tabular_features(df)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, X, feature_names=feat_names, show=False)
    plt.tight_layout()
    out = os.path.join(REPORT_DIR, 'shap_summary.png')
    plt.savefig(out, dpi=150)
    print("Saved SHAP summary to", out)
