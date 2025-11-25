from src.models.train_baselines import train_and_eval
def test_train_runs():
    # smoke test: runs training on small slice
    train_and_eval(use_synthetic=True)
    import os
    assert os.path.exists('models/lgbm.pkl')
