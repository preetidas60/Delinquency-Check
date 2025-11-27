import os
import pandas as pd
from src.utils import user_excel, synthetic_csv, amex_dir

def load_user():
    return pd.read_excel(user_excel(), sheet_name="Sample")

def load_synthetic():
    return pd.read_csv(synthetic_csv())

def load_amex():
    dfs = {}
    for f in os.listdir(amex_dir()):
        if f.endswith(".csv"):
            dfs[f] = pd.read_csv(os.path.join(amex_dir(), f))
    return dfs
