import os

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")

def user_excel():
    return os.path.join(DATA_DIR, "user", "Credit Card Delinquency Watch.xlsx")

def synthetic_csv():
    return os.path.join(DATA_DIR, "synthetic", "synthetic_indian_credit.csv")

def amex_dir():
    return os.path.join(DATA_DIR, "amex")
