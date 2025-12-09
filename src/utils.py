import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def ensure_dirs():
    for d in ["user", "synthetic", "amex"]:
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)

def user_excel():
    return str(DATA_DIR / "user" / "Credit Card Delinquency Watch.xlsx")

def synthetic_csv():
    return str(DATA_DIR / "synthetic" / "synthetic_indian_credit.csv")

def amex_dir():
    return str(DATA_DIR / "amex")
