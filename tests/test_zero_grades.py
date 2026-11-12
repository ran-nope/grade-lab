from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from streamlit.testing.v1 import AppTest

import zero_grades
from grade_lab import HurdleGradeEstimator, load_data, split_data

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
