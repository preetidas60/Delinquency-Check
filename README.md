# Credit Card Delinquency Prediction

This project predicts next-month delinquency using credit behavior data.

## Setup

1. Create virtual environment:
   python3 -m venv venv
   source venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Place datasets:
   data/user/
   data/synthetic/
   data/amex/ (optional)

## Train Model

python src/train.py

## Evaluate Model

python src/evaluate.py

## Explain Model

python src/explain.py

## Run API

uvicorn src.serve_api:app --reload --port 8000

## API Docs

http://localhost:8000/docs
