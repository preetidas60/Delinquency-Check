# Early Risk Signals — Credit Card Delinquency Watch

This repo contains code to prototype an early-warning system for credit-card delinquency.
It includes data-prep, feature engineering, baseline training (Logistic Regression + LightGBM),
and evaluation.

## Datasets (existing on this environment)

- Your uploaded Excel: /mnt/data/Credit Card Delinquency Watch.xlsx
- Synthetic dataset created for development: /mnt/data/synthetic_indian_credit.csv

## Quickstart

1. Create virtualenv and install deps:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Place the AmEx Kaggle dataset (optional) in `data/amex/` if you want to run large-scale experiments.

3. Run the pipeline:
   bash run_all.sh

## Files

- src/data_prep.py : load sample & synthetic data
- src/features.py : basic feature helpers
- src/train.py : trains Logistic Regression and LightGBM (saves models to models/)
- src/evaluate.py : basic evaluation metrics and Precision@10%
- notebooks/01_exploration.ipynb : EDA and visualization notebook

## Algorithms used

- Baseline: Logistic Regression (interpretable)
- Production candidate: LightGBM (gradient boosting)
- Planned advanced: Survival models, Transformers over transactions, Graph neural nets, Uplift models

## Notes

- The provided synthetic dataset allows you to run the pipeline end-to-end while you obtain real datasets (e.g., AmEx Kaggle).
- I used the synthetic dataset to produce baseline results. Move to real data for production.
