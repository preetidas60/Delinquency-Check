import pandas as pd
from lifelines import CoxPHFitter

def train_cox(df, duration_col='duration', event_col='event'):
    # df must contain duration (time until event or censor) and event (1=delinquency occurred)
    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col)
    return cph

if __name__ == "__main__":
    print("Provide a time-to-event dataset per customer to use survival training.")
