import numpy as np
import pandas as pd

from grade_lab import load_data, run_experiments, split_data
from robustness import baseline_ridge_intervals


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
