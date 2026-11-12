import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

from grade_lab import PreviousGrade, load_data, run_experiments, split_data
from robustness import baseline_ridge_intervals, compute_conformal_intervals


def test_baseline_ridge_intervals_match_fixed_resampling_and_selected_models():
    df = load_data()
    _, test = split_data(df)
    _, results, _, predictions = run_experiments(df)
    names = ['Earlier grades', 'Study context only']
    selected = results.set_index('experiment').loc[names]
    table = baseline_ridge_intervals(test.G3, predictions)
    pd.testing.assert_frame_equal(table, baseline_ridge_intervals(test.G3, predictions))
    assert table.experiment.tolist() == names
    assert table.selected_model.tolist() == selected.selected_model.tolist()
    assert table.test_rows.tolist() == [79, 79]
    assert table.resamples.tolist() == [1000, 1000]
    assert table.bootstrap_seed.tolist() == [2026, 2026]

    samples = np.random.default_rng(2026).integers(0, 79, size=(1000, 79))
    for row in table.itertuples():
        errors = np.abs(test.G3.to_numpy() - predictions[row.experiment])
        np.testing.assert_allclose(row.test_mae, selected.loc[row.experiment, 'test_mae'])
        np.testing.assert_allclose(
            [row.ci_low, row.ci_high], np.quantile(errors[samples].mean(axis=1), [.025, .975]),
        )


def test_conformal_intervals_use_the_correct_rank_and_have_synthetic_coverage():
    rng = np.random.default_rng(11)
    y = rng.uniform(0, 20, size=6000)
    X = np.zeros((len(y), 1))
    model = DummyRegressor(strategy='constant', constant=10)
    lower, upper, q_hat, coverage = compute_conformal_intervals(
        model, X[:4000], y[:4000], X[4000:], y_eval=y[4000:],
    )
    _, calibration = train_test_split(y[:4000], test_size=0.25, random_state=42)
    rank = int(np.ceil((len(calibration) + 1) * 0.90))
    assert q_hat == np.sort(abs(calibration - 10))[rank - 1]
    assert np.all((0 <= lower) & (lower <= upper) & (upper <= 20))
    assert 0.87 <= coverage <= 0.93
    assert not hasattr(model, 'constant_')


def test_conformal_real_split_uses_held_out_labels_only_for_coverage():
    train, test = split_data(load_data())
    cols = ['G1', 'G2']
    args = (PreviousGrade(), train[cols], train.G3, test[cols])
    lower, upper, q_hat, coverage = compute_conformal_intervals(*args, y_eval=test.G3)
    assert np.all((0 <= lower) & (lower <= upper) & (upper <= 20))
    assert coverage == np.mean((test.G3 >= lower) & (test.G3 <= upper))
    other_lower, other_upper, other_q, no_coverage = compute_conformal_intervals(*args)
    np.testing.assert_array_equal(lower, other_lower)
    np.testing.assert_array_equal(upper, other_upper)
    assert other_q == q_hat and no_coverage is None
    assert q_hat == 1
    assert coverage == 66 / 79


def test_conformal_tiny_calibration_returns_full_range():
    X = np.zeros((4, 1))
    lower, upper, q_hat, coverage = compute_conformal_intervals(DummyRegressor(), X, [1, 3, 9, 12], X)
    assert np.isinf(q_hat)
    assert coverage is None
    np.testing.assert_array_equal(lower, np.zeros(4))
    np.testing.assert_array_equal(upper, np.full(4, 20))
