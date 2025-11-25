import os, joblib, json, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score
from src.data_prep import load_synthetic, load_user_sample
from src.features import base_tabular_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_eval(use_synthetic=True):
    df = load_synthetic() if use_synthetic else load_user_sample()
    if 'dpd_next_month' not in df.columns:
        raise ValueError("Label 'dpd_next_month' not found in data")
    df = df.dropna(subset=['dpd_next_month'])
    X, feat_names = base_tabular_features(df)
    y = df['dpd_next_month'].astype(int).values

    # time-based split if month present
    if 'month' in df.columns:
        max_month = int(df['month'].max())
        train_mask = df['month'] <= int(max_month * 0.8)
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[~train_mask]
        y_test = y[~train_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # standardize for LR
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    probs_lr = lr.predict_proba(X_test_s)[:,1]
    auc_lr = roc_auc_score(y_test, probs_lr)
    print("Logistic AUC:", auc_lr)
    joblib.dump(lr, os.path.join(MODEL_DIR, 'logistic.pkl'))

    # LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    params = {'objective':'binary','metric':'auc','verbosity':-1}
    gbm = lgb.train(params, lgb_train, num_boost_round=300)
    preds = gbm.predict(X_test)
    auc_gbm = roc_auc_score(y_test, preds)
    print("LightGBM AUC:", auc_gbm)
    joblib.dump(gbm, os.path.join(MODEL_DIR, 'lgbm.pkl'))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict_proba(X_test)[:,1]
    auc_rf = roc_auc_score(y_test, preds_rf)
    print("RandomForest AUC:", auc_rf)
    joblib.dump(rf, os.path.join(MODEL_DIR, 'rf.pkl'))

    # Persist scaler & features
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feat_names, f)

    # Precision@Top10%
    for name, probs_arr in [('LR', probs_lr), ('GBM', preds), ('RF', preds_rf)]:
        k = max(1, int(0.10 * len(probs_arr)))
        topk = np.argsort(probs_arr)[-k:]
        prec = y_test[topk].mean()
        print(f"{name} Precision@10%: {prec:.3f}")

if __name__ == "__main__":
    train_and_eval(use_synthetic=True)
