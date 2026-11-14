import pandas as pd
from comparisons import compare, school_holdout
from grade_lab import load_data, split_data


def test_school_cohorts_and_determinism():
    df = load_data()
    result = school_holdout(df)
    assert len(result) == 6
    assert set(zip(result.train_n, result.test_n)) == {(349, 46), (46, 349)}
    pd.testing.assert_frame_equal(result, school_holdout(df))
    assert result.mae.ge(0).all()


def test_identical_inputs_and_baseline():
    train, test = split_data(load_data())
    result = compare(train, test).set_index('model')
    assert abs(result.loc['Previous-grade baseline', 'mae'] - (test.G2-test.G3).abs().mean()) < 1e-12
