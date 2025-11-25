import pandas as pd
from src.data_prep import load_synthetic, load_user_sample
from src.features import base_tabular_features

def build_feature_table(use_synthetic=True):
    df = load_synthetic() if use_synthetic else load_user_sample()
    # example: keep monthly rows, drop rows without label
    feat_df, feat_names = base_tabular_features(df)
    feat_df = feat_df.copy()
    # include label if present
    if 'dpd_next_month' in df.columns:
        feat_df['dpd_next_month'] = df['dpd_next_month']
    return feat_df, feat_names
