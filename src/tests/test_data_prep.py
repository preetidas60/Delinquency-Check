from src.data_prep import load_synthetic
def test_load_synthetic():
    df = load_synthetic()
    assert df is not None
    assert 'dpd_next_month' in df.columns
