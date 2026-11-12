import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
import grade_lab
from grade_lab import FEATURES, load_data, split_data, run_experiments, predict_grade, fit_studio_models
from robustness import bootstrap_mae, SENSITIVITY_SEEDS
from grade_lab import EXPANDED_FEATURES, GradeVelocityTransformer, mixed_feature_pipeline, run_expanded_benchmarks


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


def test_grade_velocity_preserves_input_and_feature_names():
    grades = pd.DataFrame({'G1': [8, 12], 'G2': [11, 9]}, index=[6, 2])
    transformer = GradeVelocityTransformer()
    result = transformer.fit_transform(grades)
    assert result.index.tolist() == [6, 2]
    assert result.grade_delta.tolist() == [3, -3]
    assert transformer.get_feature_names_out().tolist() == ['G1', 'G2', 'grade_delta']
    assert list(grades) == ['G1', 'G2']


def test_subjects_and_mixed_feature_pipelines():
    math = load_data()
    pd.testing.assert_frame_equal(math, load_data('mat'))
    assert len(math) == 395
    assert len(load_data('por')) == 649
    with pytest.raises(ValueError, match='Choose'):
        load_data('unknown')
    for columns in EXPANDED_FEATURES.values():
        train, test = split_data(math)
        pipeline = mixed_feature_pipeline(columns, Ridge(alpha=10))
        pipeline.fit(train[columns], train.G3)
        preprocess = next(step for step in pipeline.named_steps.values() if isinstance(step, ColumnTransformer))
        names = preprocess.get_feature_names_out().tolist()
        assert any(name.startswith('categorical__higher_') for name in names)
        assert ('numeric__grade_delta' in names) == ('G2' in columns)
        assert preprocess.named_transformers_['numeric'].mean_[0] == pytest.approx(train[columns[0]].mean())
        changed = test[columns].copy()
        changed['higher'] = 'unseen-category'
        with pytest.warns(UserWarning, match='unknown categories'):
            assert np.isfinite(pipeline.predict(changed)).all()


def test_expanded_benchmarks_keep_all_rows_and_score_both_slices():
    df = load_data()
    benchmarks = run_expanded_benchmarks(df)
    hurdle = benchmarks[benchmarks.model.str.startswith('Hurdle')]
    assert len(hurdle) == 2
    assert set(hurdle.experiment) == {'Grades + expanded context'}
    assert benchmarks.zero_grade_rows.tolist() == [5] * len(benchmarks)
    assert benchmarks.nonzero_grade_rows.tolist() == [74] * len(benchmarks)
    np.testing.assert_allclose(
        benchmarks.test_mae,
        (5 * benchmarks.zero_grade_mae + 74 * benchmarks.nonzero_grade_mae) / 79,
    )


def test_portuguese_download_handles_nested_archive_and_uses_cache(tmp_path, monkeypatch):
    data = b'G1;G2;G3;studytime;failures;traveltime\n10;11;12;1;0;1\n'
    inner, outer = io.BytesIO(), io.BytesIO()
    with zipfile.ZipFile(inner, 'w') as archive:
        archive.writestr('student-por.csv', data)
    with zipfile.ZipFile(outer, 'w') as archive:
        archive.writestr('student.zip', inner.getvalue())
    monkeypatch.setattr(grade_lab, 'DATA', tmp_path / 'student-mat.csv')
    requests = []

    def fetch(request, timeout):
        requests.append(request.full_url)
        return io.BytesIO(outer.getvalue())

    monkeypatch.setattr(grade_lab.urllib.request, 'urlopen', fetch)
    assert load_data('por').G3.tolist() == [12]
    assert load_data('por').G3.tolist() == [12]
    assert requests == [grade_lab.URL]
    assert (tmp_path / 'student-por.csv').read_bytes() == data
