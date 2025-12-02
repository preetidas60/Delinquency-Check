import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

df = pd.DataFrame({
    "utilisation_pct": np.random.uniform(5, 95, n),
    "avg_payment_ratio": np.random.uniform(10, 100, n),
    "min_due_paid_freq": np.random.uniform(0, 100, n),
    "merchant_mix_index": np.random.uniform(0.1, 1.0, n),
    "cash_withdrawal_pct": np.random.uniform(0, 40, n),
    "recent_spend_change_pct": np.random.uniform(-50, 50, n),
    "credit_limit": np.random.uniform(5000, 200000, n),
})

prob = (
    0.02 * df["utilisation_pct"]
    - 0.03 * df["avg_payment_ratio"]
    - 0.01 * df["min_due_paid_freq"]
    + 0.5 * df["merchant_mix_index"]
    + 0.03 * df["cash_withdrawal_pct"]
    + 0.01 * np.maximum(0, df["recent_spend_change_pct"])
) / 10

prob = 1 / (1 + np.exp(-prob))
df["dpd_next_month"] = np.random.binomial(1, prob)

df.to_csv("data/synthetic/synthetic_indian_credit.csv", index=False)
print("Dataset saved as synthetic_indian_credit.csv")
