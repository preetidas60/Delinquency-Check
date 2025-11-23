import joblib, os
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.data_prep import load_synthetic

def evaluate():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    lr = joblib.load(os.path.join(models_dir, 'logistic.pkl'))
    gbm = joblib.load(os.path.join(models_dir, 'lgbm.pkl'))
    df = load_synthetic()
    df = df.dropna(subset=['dpd_next_month'])
    features = ['utilisation_pct','avg_payment_ratio','min_due_paid_freq',
                'merchant_mix_index','cash_withdrawal_pct','recent_spend_change_pct','credit_limit']
    X = df[features].astype(float)
    y = df['dpd_next_month'].astype(int)
    Xs = scaler.transform(X)
    probs = lr.predict_proba(Xs)[:,1]
    print('LR AUC', roc_auc_score(y, probs))
    k = max(1, int(0.10 * len(probs)))
    topk = probs.argsort()[-k:]
    print('Precision@10%:', y.values[topk].mean())

if __name__ == '__main__':
    evaluate()
