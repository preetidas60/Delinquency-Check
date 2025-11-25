import os

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")

def path_user_excel():
    return os.path.join(DATA_DIR, "user", "Credit Card Delinquency Watch.xlsx")

def path_synthetic():
    return os.path.join(DATA_DIR, "synthetic", "synthetic_indian_credit.csv")

def path_amex_dir():
    return os.path.join(DATA_DIR, "amex")

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "user"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "synthetic"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "amex"), exist_ok=True)
