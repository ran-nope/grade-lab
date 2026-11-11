import numpy as np
import pytest
from grade_lab import FEATURES, load_data, split_data, run_experiments, predict_grade


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
    before = app.metric[3].value
    app.slider[0].set_value(18)
    app.slider[1].set_value(18)
    app.run(timeout=60)
    assert not app.exception
    assert app.metric[3].value != before
