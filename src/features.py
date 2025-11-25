import pandas as pd
import numpy as np

def sanitize_columns(df):
    df = df.copy()
    rename_map = {}
    for c in df.columns:
        c2 = c.strip().replace(' ', '_').replace('%', 'pct').replace('-', '_').lower()
        rename_map[c] = c2
    df.rename(columns=rename_map, inplace=True)
    return df

def base_tabular_features(df):
    df = sanitize_columns(df)
    # Standardize min_due naming
    if 'min_due_paid_frequency' in df.columns and 'min_due_paid_freq' not in df.columns:
        df['min_due_paid_freq'] = df['min_due_paid_frequency']
    # ensure numeric types
    for c in ['utilisation_pct','avg_payment_ratio','min_due_paid_freq','merchant_mix_index','cash_withdrawal_pct','recent_spend_change_pct','credit_limit']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    # derived fractions
    if 'utilisation_pct' in df.columns:
        df['utilisation_frac'] = df['utilisation_pct'] / 100.0
    if 'avg_payment_ratio' in df.columns:
        df['avg_payment_frac'] = df['avg_payment_ratio'] / 100.0
    features = []
    for c in ['utilisation_frac','avg_payment_frac','min_due_paid_freq','merchant_mix_index','cash_withdrawal_pct','recent_spend_change_pct','credit_limit']:
        if c in df.columns:
            features.append(c)
    X = df[features].fillna(0.0)
    return X, features
