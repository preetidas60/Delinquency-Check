import os
import pandas as pd
from src.utils import path_user_excel, path_synthetic, path_amex_dir

def load_user_sample():
    p = path_user_excel()
    if not os.path.exists(p):
        raise FileNotFoundError(f"User sample not found at {p}")
    df = pd.read_excel(p, sheet_name='Sample')
    return df

def load_synthetic():
    p = path_synthetic()
    if not os.path.exists(p):
        raise FileNotFoundError(f"Synthetic CSV not found at {p}")
    return pd.read_csv(p)

def load_amex(amex_dir=None):
    if amex_dir is None:
        amex_dir = path_amex_dir()
    if not os.path.exists(amex_dir):
        raise FileNotFoundError(f"AmEx folder not found at {amex_dir}")
    files = [f for f in os.listdir(amex_dir) if f.endswith('.csv')]
    dfs = {}
    for f in files:
        dfs[f] = pd.read_csv(os.path.join(amex_dir, f))
    return dfs
