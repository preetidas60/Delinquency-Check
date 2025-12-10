import os
import pandas as pd
from src.utils import user_excel, synthetic_csv, amex_dir

# ------------------------------
# SAFE READERS
# ------------------------------

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


# ------------------------------
# DATASET LOADERS
# ------------------------------

def load_user():
    return _safe_read_excel(user_excel())


def load_synthetic():
    return _safe_read_csv(synthetic_csv())


# Load AmEx in CHUNKS (memory-safe)
def load_amex_sample(rows=100000):
    """
    Loads a random sample from the huge AmEx dataset using safe low-memory chunking.
    Designed for machines with 4–8GB RAM.
    """

    d = amex_dir()
    train_path = os.path.join(d, "train_data.csv")
    labels_path = os.path.join(d, "train_labels.csv")

    if not os.path.exists(train_path):
        print("⚠️ AmEx train_data.csv not found.")
        return None

    if not os.path.exists(labels_path):
        print("⚠️ AmEx train_labels.csv not found.")
        return None

    print("📥 Loading AmEx CHUNKED sample (RAM-safe)...")

    chunks = []
    chunk_size = 50000     # SAFE FOR LOW RAM
    sample_frac = 0.01     # 1% of each chunk
    current_rows = 0

    for chunk in pd.read_csv(train_path, chunksize=chunk_size, dtype=str):
        samp = chunk.sample(frac=sample_frac)
        chunks.append(samp)
        current_rows += len(samp)

        if current_rows >= rows:
            break

    df = pd.concat(chunks, ignore_index=True)

    labels = pd.read_csv(labels_path, dtype=str)
    df = df.merge(labels, on="customer_ID", how="inner")

    print(f"✅ Loaded AmEx sample safely with {len(df)} rows.")
    return df


# ------------------------------
# DATASET SELECTION MENU
# ------------------------------

def choose_dataset():
    print("\n📊 Select which dataset to use:")
    print("1. Synthetic")
    print("2. User Excel")
    print("3. AmEx (chunked sample)")
    choice = input("\nEnter choice (1/2/3): ")

    if choice == "1":
        df = load_synthetic()
        if df is None:
            raise FileNotFoundError("Synthetic dataset not found.")
        return "synthetic", df

    elif choice == "2":
        df = load_user()
        if df is None:
            raise FileNotFoundError("User Excel dataset not found.")
        return "user", df

    elif choice == "3":
        df = load_amex_sample()
        if df is None:
            raise FileNotFoundError("AmEx dataset not found.")
        return "amex", df

    else:
        raise ValueError("Invalid choice. Please enter 1, 2, or 3.")
