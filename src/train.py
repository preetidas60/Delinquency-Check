import os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.data_prep import load_synthetic, load_user_excel

def prepare_training_data(use_synthetic=True):
    if use_synthetic:
        df = load_synthetic()
    else:
        df = load_user_excel()
    # drop rows with missing label
    df = df.dropna(subset=['dpd_next_month'])
    # basic features
    features = ['utilisation_pct','avg_payment_ratio','min_due_paid_freq',
                'merchant_mix_index','cash_withdrawal_pct','recent_spend_change_pct','credit_limit']
    X = df[features].astype(float)
    y = df['dpd_next_month'].astype(int)
    return X, y

def train_baselines(use_synthetic=True):
    X, y = prepare_training_data(use_synthetic)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    probs_lr = lr.predict_proba(X_test_s)[:,1]
    auc_lr = roc_auc_score(y_test, probs_lr)
    print('Logistic AUC:', auc_lr)
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
    joblib.dump(lr, os.path.join(os.path.dirname(__file__), '..', 'models', 'logistic.pkl'))
    # LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    params = {'objective':'binary','metric':'auc','verbosity':-1,'boosting_type':'gbdt'}
    gbm = lgb.train(params, lgb_train, num_boost_round=200)
    preds = gbm.predict(X_test)
    auc_gbm = roc_auc_score(y_test, preds)
    print('LightGBM AUC:', auc_gbm)
    joblib.dump(gbm, os.path.join(os.path.dirname(__file__), '..', 'models', 'lgbm.pkl'))
    # Save scaler
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl'))

if __name__ == '__main__':
    train_baselines(use_synthetic=True)
