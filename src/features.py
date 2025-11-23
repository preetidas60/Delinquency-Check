import pandas as pd
import numpy as np

def compute_features_monthly(df):
    """
    Simple example: compute a stable set of aggregates per customer.
    Input expects monthly-level rows (customer_id, month,...).
    """
    df = df.copy()
    df['utilisation_frac'] = df['utilisation_pct'] / 100.0
    df['avg_payment_frac'] = df['avg_payment_ratio'] / 100.0
    # Example aggregation per customer (can be extended)
    agg = df.sort_values(['customer_id','month']).groupby('customer_id').agg({
        'utilisation_frac':'mean',
        'avg_payment_frac':'mean',
        'min_due_paid_freq':'mean',
        'merchant_mix_index':'mean',
        'cash_withdrawal_pct':'mean',
        'recent_spend_change_pct':'mean'
    }).reset_index()
    return agg

if __name__ == '__main__':
    print('Feature helpers module. Import compute_features_monthly in train script.')
