import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")

def load_user_excel(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'Credit Card Delinquency Watch.xlsx')
    print('Loading user data from', path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # the sample sheet in your Excel is 'Sample'
    df = pd.read_excel(path, sheet_name='Sample')
    return df

def load_synthetic(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'synthetic_indian_credit.csv')
    print('Loading synthetic data from', path)
    return pd.read_csv(path)

def main_preview():
    print('\n-- User Excel Preview --')
    try:
        df = load_user_excel()
        print(df.shape)
        print(df.head())
    except Exception as e:
        print('Could not load user excel:', e)
    print('\n-- Synthetic Preview --')
    try:
        sf = load_synthetic()
        print(sf.shape)
        print(sf.head())
    except Exception as e:
        print('Could not load synthetic:', e)

if __name__ == '__main__':
    main_preview()
