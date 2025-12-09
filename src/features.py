import pandas as pd

REQUIRED = [
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
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def validate(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

def build_features(df):
    df = normalize_columns(df)
    validate(df)

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
