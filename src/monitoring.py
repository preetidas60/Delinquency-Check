# Monitoring hooks: compute PSI, drift alerts, and store metrics to logs or a monitoring system
import numpy as np
import pandas as pd

def psi(expected, actual, buckets=10):
    # population stability index
    def scale_range(vals, n):
        counts, _ = np.histogram(vals, bins=n)
        return counts / counts.sum()
    exp_pct = scale_range(expected, buckets)
    act_pct = scale_range(actual, buckets)
    eps = 1e-6
    psi_vals = (exp_pct - act_pct) * np.log((exp_pct + eps) / (act_pct + eps))
    return psi_vals.sum()
