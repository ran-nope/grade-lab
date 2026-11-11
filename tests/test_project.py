import numpy as np
import pytest
from grade_lab import FEATURES, load_data, split_data, run_experiments, predict_grade, fit_studio_models
from robustness import bootstrap_mae, SENSITIVITY_SEEDS


def test_data_and_no_target_leakage():
    df = load_data()
    assert len(df) == 395
    assert df[['G1', 'G2', 'G3']].notna().all().all()
    assert all('G3' not in cols and 'absences' not in cols for cols in FEATURES.values())
    train, test = split_data(df)
    assert len(train) == 316 and len(test) == 79
    assert not set(train.index) & set(test.index)


def test_selection_uses_training_scores_and_predictions_are_finite():
    df = load_data()
    cv, results, models, predictions = run_experiments(df)
    for row in results.itertuples():
        group = cv[cv.experiment == row.experiment]
        assert row.selected_model == group.loc[group.cv_mae.idxmin(), 'model']
        assert np.isfinite(predictions[row.experiment]).all()
    assert np.isfinite(predict_grade(models['Earlier grades'], 10, 12))
    with pytest.raises(ValueError):
        predict_grade(models['Earlier grades'], -1, 12)


def test_heldout_targets_cannot_change_model_selection():
    df = load_data()
    _, test = split_data(df)
    changed = df.copy()
    changed.loc[test.index, 'G3'] = 20 - changed.loc[test.index, 'G3']
    cv1, _, _, p1 = run_experiments(df)
    cv2, _, _, p2 = run_experiments(changed)
    assert cv1.equals(cv2)
    for key in p1:
        np.testing.assert_allclose(p1[key], p2[key])


def test_streamlit_app():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run(timeout=60)
    assert not app.exception
    assert app.selectbox[0].value == 'Previous-grade baseline'
    before = app.metric[3].value
    app.slider[0].set_value(18).run(timeout=60)
    assert app.metric[3].value == before
    app.slider[1].set_value(18)
    app.run(timeout=60)
    assert not app.exception
    assert app.metric[3].value != before
    app.selectbox[0].set_value('Ridge regression').run(timeout=60)
    ridge_before = app.metric[3].value
    app.slider[0].set_value(2).run(timeout=60)
    assert app.metric[3].value != ridge_before
    app.radio[0].set_value('G1').run(timeout=60)
    app.selectbox[0].set_value('Small decision tree').run(timeout=60)
    assert not app.exception


def test_studio_never_fits_heldout_labels():
    df = load_data()
    _, test = split_data(df)
    modified = df.copy()
    modified.loc[test.index, 'G3'] = 20 - modified.loc[test.index, 'G3']
    original, changed = fit_studio_models(df), fit_studio_models(modified)
    for name in original:
        np.testing.assert_allclose(original[name].predict(test[['G1', 'G2']]), changed[name].predict(test[['G1', 'G2']]))
    baseline = original['Previous-grade baseline']
    assert predict_grade(baseline, 2, 12) == predict_grade(baseline, 18, 12) == 12
    ridge = original['Ridge regression']
    assert predict_grade(ridge, 2, 12) != predict_grade(ridge, 18, 12)


def test_paired_bootstrap_known_errors():
    actual = np.array([1., 2., 3.])
    table, samples = bootstrap_mae(actual, {'perfect': actual, 'offset': actual + 2}, n_boot=100)
    np.testing.assert_array_equal(samples['perfect'], np.zeros(100))
    np.testing.assert_allclose(samples['offset'] - samples['perfect'], 2)
    assert table.loc[0, 'ci_low'] == table.loc[0, 'ci_high'] == 0
    table2, samples2 = bootstrap_mae(actual, {'variable': actual + [0, 1, 5]}, n_boot=100)
    table3, samples3 = bootstrap_mae(actual, {'variable': actual + [0, 1, 5]}, n_boot=100)
    assert table2.equals(table3)
    np.testing.assert_array_equal(samples2['variable'], samples3['variable'])
    with pytest.raises(ValueError):
        bootstrap_mae(actual, {'bad': [1, 2]}, n_boot=100)


def test_sensitivity_split_isolation():
    df = load_data()
    for seed in SENSITIVITY_SEEDS:
        train, test = split_data(df, seed)
        assert len(train) == 316 and len(test) == 79
        assert set(train.index).isdisjoint(test.index)
