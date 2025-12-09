import os
import pandas as pd
from src.utils import user_excel, synthetic_csv, amex_dir

def _safe_read_csv(path):
    if not os.path.exists(path):
        print(f"⚠️ Missing CSV: {path}")
        return None
    return pd.read_csv(path)

def _safe_read_excel(path):
    if not os.path.exists(path):
        print(f"⚠️ Missing Excel file: {path}")
        return None
    return pd.read_excel(path, sheet_name="Sample")

def load_user():
    return _safe_read_excel(user_excel())

def load_synthetic():
    return _safe_read_csv(synthetic_csv())

def load_amex():
    d = amex_dir()
    if not os.path.exists(d):
        print(f"⚠️ AmEx folder missing: {d}")
        return None

    csvs = [f for f in os.listdir(d) if f.endswith(".csv")]
    if not csvs:
        print("⚠️ No AmEx CSVs found")
        return None

    dfs = {}
    for f in csvs:
        path = os.path.join(d, f)
        dfs[f] = pd.read_csv(path, low_memory=True)

    return dfs

def load_best_dataset():
    """
    Priority:
    1. AmEx (best, largest)
    2. Synthetic
    3. User Excel
    """

    amex = load_amex()
    if amex:
        print("📌 Using AmEx dataset")
        return "amex", amex

    syn = load_synthetic()
    if syn is not None:
        print("📌 Using Synthetic dataset")
        return "synthetic", syn

    usr = load_user()
    if usr is not None:
        print("📌 Using User Excel sample")
        return "user", usr

    raise FileNotFoundError("❌ No dataset available.")
