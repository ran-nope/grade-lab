from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import StratifiedKFold
from streamlit.testing.v1 import AppTest

import zero_grades
from grade_lab import EXPANDED_FEATURES, PROSPECTIVE_CONTEXT, HurdleGradeEstimator, load_data, split_data

ROOT = Path(__file__).parents[1]


def test_spotlight_compares_both_groups():
    development = pd.DataFrame({'G1': [4, 8, 9, 13], 'G2': [0, 6, 0, 12], 'G3': [0, 0, 1, 13]})
    summary = zero_grades.summarize_zero_grades(development)
    assert summary.to_dict('records') == [
        {'Final grade': 'Zero', 'Records': 2, 'G1 median': 6.0, 'G1 range': '4 to 8',
         'G2 median': 3.0, 'G2 range': '0 to 6', 'G2 equals zero': 1},
        {'Final grade': 'Above zero', 'Records': 2, 'G1 median': 11.0, 'G1 range': '9 to 13',
         'G2 median': 6.0, 'G2 range': '0 to 12', 'G2 equals zero': 1},
    ]


def test_generated_report_matches_published_file_and_ignores_held_out_rows(tmp_path, monkeypatch):
    data = load_data()
    _, held_out = split_data(data)
    data.loc[held_out.index, ['G1', 'G2', 'G3']] = 0
    monkeypatch.setattr(zero_grades, 'load_data', lambda: data)
    generated = tmp_path / 'ZERO_GRADES.md'
    zero_grades.save_zero_grade_report(generated)
    assert generated.read_text(encoding='utf-8') == (
        ROOT / 'reports' / 'ZERO_GRADES.md'
    ).read_text(encoding='utf-8')


def test_evidence_displays_the_development_spotlight():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    assert not app.exception
    evidence = app.tabs[1]
    spotlight, = [item.value for item in evidence.dataframe if 'G2 equals zero' in item.value]
    development, _ = split_data(load_data())
    pd.testing.assert_frame_equal(spotlight, zero_grades.summarize_zero_grades(development))
    assert spotlight.Records.sum() == 316
    assert any(caption.value == zero_grades.SCOPE for caption in evidence.caption)


@pytest.mark.parametrize('mode', ['hard', 'expected'])
def test_hurdle_preserves_hard_zero_and_clips_predictions(mode):
    df = load_data()
    development, test = split_data(df)
    columns = ['G1', 'G2', 'absences', 'failures', 'higher']
    model = clone(HurdleGradeEstimator(
        base_regressor=DummyRegressor(strategy='constant', constant=25),
        mode=mode, threshold=1,
    )).fit(development[columns], development.G3)
    prediction = model.predict(test[columns])
    assert np.all(prediction[test.G2.eq(0)] == 0.0)
    assert np.all((0 <= prediction) & (prediction <= 20))
    if mode == 'hard':
        assert np.all(prediction[test.G2.gt(0)] == 20.0)


def test_hurdle_regressor_fits_only_positive_targets_and_handles_one_class():
    X = pd.DataFrame({'G1': [4, 6, 8], 'G2': [0, 6, 8], 'absences': [1, 0, 3], 'failures': [1, 0, 0]})
    model = HurdleGradeEstimator(base_regressor=DummyRegressor())
    np.testing.assert_allclose(model.fit(X, [0, 8, 12]).predict(X), [0, 10, 10])
    np.testing.assert_allclose(model.fit(X, [0, 0, 0]).predict(X), [0, 0, 0])
    with pytest.raises(ValueError, match='threshold'):
        HurdleGradeEstimator(threshold=2).fit(X, [0, 8, 12])


def test_auto_threshold_minimizes_development_out_of_fold_mae():
    development, _ = split_data(load_data())
    columns = EXPANDED_FEATURES[PROSPECTIVE_CONTEXT]
    X, y = development[columns].reset_index(drop=True), development.G3.to_numpy()
    active = np.flatnonzero(X.G2.gt(0))
    probabilities, grades = np.zeros(len(X)), np.zeros(len(X))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for _, validation in cv.split(active, y[active] == 0):
        validation_rows = active[validation]
        training_rows = np.setdiff1d(np.arange(len(X)), validation_rows)
        fixed = HurdleGradeEstimator(mode='hard', threshold=0.5).fit(X.iloc[training_rows], y[training_rows])
        probabilities[validation_rows] = fixed.classifier_.predict_proba(X.iloc[validation_rows])[:, 1]
        grades[validation_rows] = fixed.regressor_.predict(X.iloc[validation_rows])

    def loss(threshold):
        prediction = np.where(probabilities >= threshold, 0, np.clip(grades, 0, 20))
        errors = np.abs(y - prediction)
        return errors[active].mean(), errors[active][y[active] == 0].mean()

    expected = min(np.linspace(0.35, 0.95, 61), key=loss)
    model = HurdleGradeEstimator(mode='hard').fit(X, y)
    assert 0.35 <= model.threshold_ <= 0.95
    assert model.threshold_ == expected


def test_expected_mode_uses_noise_floor():
    X = pd.DataFrame({'G1': [8, 10, 14, 12], 'G2': [8, 10, 14, 0], 'failures': [1, 0, 0, 0]})
    y = [0, 10, 14, 0]
    guarded = HurdleGradeEstimator(base_regressor=DummyRegressor(), noise_floor=1).fit(X, y)
    blended = HurdleGradeEstimator(base_regressor=DummyRegressor(), noise_floor=0).fit(X, y)
    np.testing.assert_array_equal(guarded.predict(X), [12, 12, 12, 0])
    assert (blended.predict(X)[:3] < 12).all()
    assert blended.predict(X)[3] == 0
