import os, joblib, numpy as np
from sklearn.metrics import roc_auc_score
from src.data_prep import load_synthetic, load_user_sample
from src.features import base_tabular_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

def evaluate(use_synthetic=True):
    df = load_synthetic() if use_synthetic else load_user_sample()
    df = df.dropna(subset=['dpd_next_month'])
    X, feat_names = base_tabular_features(df)
    y = df['dpd_next_month'].astype(int).values
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    lr = joblib.load(os.path.join(MODEL_DIR, 'logistic.pkl'))
    gbm = joblib.load(os.path.join(MODEL_DIR, 'lgbm.pkl'))
    Xs = scaler.transform(X)
    probs_lr = lr.predict_proba(Xs)[:,1]
    print('LR AUC:', roc_auc_score(y, probs_lr))
    k = max(1, int(0.1 * len(probs_lr)))
    topk = np.argsort(probs_lr)[-k:]
    print('LR Precision@10%:', y[topk].mean())

if __name__ == "__main__":
    evaluate(use_synthetic=True)
