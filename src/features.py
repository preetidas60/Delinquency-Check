def build_features(df):
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    df['utilisation_frac'] = df['utilisation_pct'] / 100
    df['avg_payment_frac'] = df['avg_payment_ratio'] / 100

    features = [
        'utilisation_frac',
        'avg_payment_frac',
        'min_due_paid_freq',
        'merchant_mix_index',
        'cash_withdrawal_pct',
        'recent_spend_change_pct',
        'credit_limit'
    ]

    X = df[features].fillna(0)
    y = df['dpd_next_month'].astype(int)

    return X, y, features
