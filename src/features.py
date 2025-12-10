import pandas as pd
import numpy as np

HDFC_REQUIRED = [
    "utilisation_pct",
    "avg_payment_ratio",
    "min_due_paid_freq",
    "merchant_mix_index",
    "cash_withdrawal_pct",
    "recent_spend_change_pct",
    "credit_limit"
]

def normalize_columns(df):
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def build_features(df):
    df = normalize_columns(df)

    # -------------------------------------------------------
    # CASE 1 — HDFC / Synthetic Dataset (your custom features)
    # -------------------------------------------------------
    if all(col in df.columns for col in HDFC_REQUIRED):
        df["utilisation_frac"] = df["utilisation_pct"] / 100
        df["avg_payment_frac"] = df["avg_payment_ratio"] / 100

        features = [
            "utilisation_frac",
            "avg_payment_frac",
            "min_due_paid_freq",
            "merchant_mix_index",
            "cash_withdrawal_pct",
            "recent_spend_change_pct",
            "credit_limit",
        ]

        X = df[features].astype(float)
        y = df["dpd_next_month"].astype(int) if "dpd_next_month" in df.columns else None

        return X, y, features

    # -------------------------------------------------------
    # CASE 2 — AmEx Dataset (encoded Kaggle features)
    # -------------------------------------------------------
    if "target" in df.columns:

        # 1. Convert date column S_2 to numeric days
        if "s_2" in df.columns:
            df["s_2"] = pd.to_datetime(df["s_2"], errors="coerce")
            df["s_2"] = (df["s_2"] - df["s_2"].min()).dt.days

        # 2. Replace possible 'nan' strings with actual NaN
        df = df.replace("nan", np.nan)

        # 3. Convert all feature columns except these to numeric
        exclude_cols = ["customer_id", "target"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Convert safely: non-convertible values become NaN
        df[feature_cols] = df[feature_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        X = df[feature_cols]
        y = df["target"].astype(int)

        return X, y, feature_cols

    # -------------------------------------------------------
    # Unknown dataset schema → fail intentionally
    # -------------------------------------------------------
    raise ValueError("Unknown dataset schema. Cannot build features.")
