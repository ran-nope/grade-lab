"""Small, reproducible ML experiment. Read this after the notebook."""
from pathlib import Path
from typing import Literal
import hashlib
import io
import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'student-mat.csv'
URL = 'https://archive.ics.uci.edu/static/public/320/student+performance.zip'
FEATURES = {
    'Earlier grades': ['G1', 'G2'],
    'Study context only': ['studytime', 'failures', 'traveltime'],
    'Grades + context': ['G1', 'G2', 'studytime', 'failures', 'traveltime'],
}
EXPANDED_NUMERIC = ['studytime', 'failures', 'traveltime', 'absences', 'Medu', 'Fedu']
EXPANDED_CATEGORICAL = ['higher', 'schoolsup', 'famsup', 'internet', 'paid']
EXPANDED_FEATURES = {
    'Expanded context': EXPANDED_NUMERIC + EXPANDED_CATEGORICAL,
    'Grades + expanded context': ['G1', 'G2'] + EXPANDED_NUMERIC + EXPANDED_CATEGORICAL,
}


class PreviousGrade(RegressorMixin, BaseEstimator):
    """A strong baseline: predict that the final grade equals G2."""
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return X['G2'].to_numpy(dtype=float)


class GradeVelocityTransformer(TransformerMixin, BaseEstimator):
    """Add G2 minus G1 to a copy of a grade DataFrame."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return X.assign(grade_delta=X['G2'] - X['G1'])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        names = list(self.feature_names_in_ if input_features is None else input_features)
        if 'grade_delta' not in names:
            names.append('grade_delta')
        return np.asarray(names, dtype=object)


def mixed_feature_pipeline(columns, estimator):
    """Fit scaling and category encoding inside the caller's training fold."""
    numeric = [name for name in columns if name not in EXPANDED_CATEGORICAL]
    categorical = [name for name in columns if name in EXPANDED_CATEGORICAL]
    steps = []
    if {'G1', 'G2'}.issubset(numeric):
        steps.append(GradeVelocityTransformer())
        if 'grade_delta' not in numeric:
            numeric.append('grade_delta')
    preprocess = ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('categorical', OneHotEncoder(drop='if_binary', handle_unknown='ignore',
                                      sparse_output=False), categorical),
    ])
    return make_pipeline(*steps, preprocess, estimator)


class HurdleGradeEstimator(RegressorMixin, BaseEstimator):
    """Estimate zero risk and the grade conditional on a positive outcome."""

    def __init__(self, base_regressor=None, mode='expected', threshold=0.5):
        self.base_regressor = base_regressor
        self.mode = mode
        self.threshold = threshold

    def fit(self, X, y):
        if self.mode not in ('hard', 'expected') or not 0 <= self.threshold <= 1:
            raise ValueError('Use hard or expected mode and a threshold between 0 and 1.')
        required = {'G1', 'G2', 'absences', 'failures'}
        if not required.issubset(X.columns):
            raise ValueError(f'Missing hurdle inputs: {sorted(required - set(X.columns))}')
        columns = [name for name in EXPANDED_FEATURES['Grades + expanded context'] if name in X]
        X = X[columns]
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X),) or not len(y) or not np.all((0 <= y) & (y <= 20)):
            raise ValueError('Expected one final grade between 0 and 20 per row.')
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        active = X.G2.gt(0).to_numpy()
        labels = y[active] == 0
        self.classifier_ = None
        self.zero_probability_ = float(labels.mean()) if labels.size else 1.0
        if np.unique(labels).size == 2:
            self.classifier_ = mixed_feature_pipeline(
                X.columns, LogisticRegression(class_weight='balanced', random_state=42),
            ).fit(X.iloc[active], labels)
        self.regressor_ = None
        positive = y > 0
        if positive.any():
            base = Ridge(alpha=10.0) if self.base_regressor is None else clone(self.base_regressor)
            regressor = base if isinstance(base, PreviousGrade) else mixed_feature_pipeline(X.columns, base)
            self.regressor_ = regressor.fit(X.iloc[positive], y[positive])
        return self

    def predict(self, X):
        check_is_fitted(self, 'regressor_')
        active = X.G2.gt(0).to_numpy()
        estimates = np.zeros(len(X))
        if not active.any() or self.regressor_ is None:
            return estimates
        rows = X.iloc[active][self.feature_names_in_]
        probability = (self.classifier_.predict_proba(rows)[:, 1] if self.classifier_ is not None
                       else self.zero_probability_)
        positive_grade = self.regressor_.predict(rows)
        if self.mode == 'hard':
            estimates[active] = np.where(probability >= self.threshold, 0.0, positive_grade)
        else:
            estimates[active] = (1 - probability) * positive_grade
        return np.clip(estimates, 0.0, 20.0)


def download_data(subject: Literal['mat', 'por'] = 'mat'):
    if subject not in ('mat', 'por'):
        raise ValueError("Choose 'mat' or 'por'.")
    path = DATA if subject == 'mat' else DATA.with_name('student-por.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(URL, headers={'User-Agent': 'GradeLab/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        member = f'student-{subject}.csv'
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if member in archive.namelist():
                data = archive.read(member)
            else:
                with zipfile.ZipFile(io.BytesIO(archive.read('student.zip'))) as inner:
                    data = inner.read(member)
        path.write_bytes(data)
    return path


def load_data(subject: Literal['mat', 'por'] = 'mat'):
    df = pd.read_csv(download_data(subject), sep=';')
    required = sorted(set(sum(FEATURES.values(), []) + ['G3']))
    if df[required].isna().any().any():
        raise ValueError('Missing values found: inspect data before training.')
    if not df[['G1', 'G2', 'G3']].apply(lambda s: s.between(0, 20).all()).all():
        raise ValueError('Grades must lie between 0 and 20.')
    return df


def split_data(df, seed=42):
    # Identical split for every experiment; no subject datasets are combined.
    return train_test_split(df, test_size=0.2, random_state=seed)


def candidates(columns):
    models = {
        'Mean baseline': DummyRegressor(strategy='mean'),
        'Ridge regression': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'Small decision tree': DecisionTreeRegressor(max_depth=3, min_samples_leaf=15, random_state=42),
    }
    if 'G2' in columns:
        models['Previous-grade baseline'] = PreviousGrade()
    return models


def run_experiments(df, seed=42):
    train, test = split_data(df, seed)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    comparisons, results, fitted, predictions = [], [], {}, {}
    for experiment, columns in FEATURES.items():
        models = candidates(columns)
        scores = {}
        for name, model in models.items():
            errors = -cross_val_score(model, train[columns], train.G3, cv=cv, scoring='neg_mean_absolute_error')
            scores[name] = float(errors.mean())
            comparisons.append({'experiment': experiment, 'model': name, 'cv_mae': errors.mean(), 'cv_std': errors.std()})
        winner = min(scores, key=scores.get)
        model = models[winner].fit(train[columns], train.G3)
        pred = model.predict(test[columns])
        fitted[experiment] = model
        predictions[experiment] = pred
        results.append({'experiment': experiment, 'selected_model': winner,
                        'test_mae': mean_absolute_error(test.G3, pred),
                        'test_rmse': np.sqrt(mean_squared_error(test.G3, pred)),
                        'test_r2': r2_score(test.G3, pred)})
    return pd.DataFrame(comparisons), pd.DataFrame(results), fitted, predictions


def grade_slice_scores(actual, predicted):
    errors = np.abs(np.asarray(actual) - np.asarray(predicted))
    zero = np.asarray(actual) == 0
    return {
        'test_mae': errors.mean(),
        'zero_grade_mae': errors[zero].mean() if zero.any() else np.nan,
        'nonzero_grade_mae': errors[~zero].mean() if (~zero).any() else np.nan,
        'zero_grade_rows': int(zero.sum()),
        'nonzero_grade_rows': int((~zero).sum()),
    }


def run_expanded_benchmarks(df, seed=42):
    """Evaluate prespecified candidates separately from the original experiment."""
    train, test = split_data(df, seed)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for experiment, columns in EXPANDED_FEATURES.items():
        models = {
            'Mean baseline': DummyRegressor(strategy='mean'),
            'Ridge regression': mixed_feature_pipeline(columns, Ridge(alpha=10.0)),
            'Small decision tree': mixed_feature_pipeline(
                columns, DecisionTreeRegressor(max_depth=3, min_samples_leaf=15, random_state=42),
            ),
        }
        if 'G2' in columns:
            models.update({
                'Previous-grade baseline': PreviousGrade(),
                'Hurdle (expected)': HurdleGradeEstimator(),
                'Hurdle (hard, threshold 0.5)': HurdleGradeEstimator(mode='hard'),
            })
        for name, model in models.items():
            # DummyRegressor ignores the input values, so no encoding is needed.
            scores = -cross_val_score(
                model, train[columns], train.G3, cv=cv, scoring='neg_mean_absolute_error',
            )
            prediction = model.fit(train[columns], train.G3).predict(test[columns])
            rows.append({'experiment': experiment, 'model': name, 'cv_mae': scores.mean(),
                         **grade_slice_scores(test.G3, prediction)})
    return pd.DataFrame(rows)


def fit_studio_models(df):
    """Fit fixed alternatives on training rows only; never select on test data."""
    train, _ = split_data(df)
    columns = FEATURES['Earlier grades']
    return {name: model.fit(train[columns], train.G3)
            for name, model in candidates(columns).items() if name != 'Mean baseline'}


def predict_grade(model, g1, g2):
    if not (0 <= g1 <= 20 and 0 <= g2 <= 20):
        raise ValueError('Grades must be between 0 and 20.')
    # Keep raw estimates: silently clipping would change the evaluated estimator.
    return float(model.predict(pd.DataFrame({'G1': [g1], 'G2': [g2]}))[0])


def save_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = load_data()
    cv, results, models, predictions = run_experiments(df)
    train, test = split_data(df)
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    cv.to_csv(out / 'cross_validation.csv', index=False)
    results.to_csv(out / 'test_results.csv', index=False)
    errors = pd.DataFrame({'actual': test.G3, 'predicted': predictions['Earlier grades']})
    errors['absolute_error'] = (errors.actual - errors.predicted).abs()
    errors.to_csv(out / 'held_out_predictions.csv', index_label='source_row')
    metadata = {'rows': len(df), 'train_rows': len(train), 'test_rows': len(test),
                'seed': 42, 'sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
                'source': URL, 'features': FEATURES}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='#f5f7fb')
    df.G3.hist(bins=np.arange(-0.5, 21.5), ax=axes[0, 0], color='#4361ee', edgecolor='white')
    axes[0, 0].set(title='Final grades include zeros', xlabel='Final grade / 20', ylabel='Students')
    means = df.groupby('studytime').G3.agg(['mean', 'count'])
    axes[0, 1].bar(['<2 h', '2–5 h', '5–10 h', '>10 h'], means['mean'], color='#139c90')
    axes[0, 1].set(title='Association is not causation', xlabel='Reported weekly study time', ylabel='Mean final grade')
    zero = test.G3.eq(0)
    axes[1, 0].scatter(test.G3[~zero], predictions['Earlier grades'][~zero], alpha=.6, color='#4361ee', label='Nonzero final grade')
    axes[1, 0].scatter(test.G3[zero], predictions['Earlier grades'][zero], s=65, facecolors='none', edgecolors='#c66a24', linewidths=2, label='Zero final grade (n=5)')
    axes[1, 0].plot([0, 20], [0, 20], '--', color='#222')
    axes[1, 0].set(title='Unseen students: prediction vs reality', xlabel='Actual grade', ylabel='Predicted grade')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].barh(results.experiment, results.test_mae, color=['#4361ee', '#ef946c', '#139c90'])
    axes[1, 1].set(title='Do earlier grades matter?', xlabel='Test MAE · lower is better')
    fig.suptitle('GRADE LAB  /  Evidence before confidence', fontsize=19, fontweight='bold')
    fig.tight_layout(pad=2)
    fig.savefig(out / 'experiment.png', dpi=150)
    plt.close(fig)
    lines = ['# Results', '', 'Models were selected using training-only five-fold cross-validation. The fixed test set was used only after selection.', '', '| Experiment | Selected model | Test MAE | RMSE | R² |', '|---|---|---:|---:|---:|']
    for r in results.itertuples():
        lines.append(f'| {r.experiment} | {r.selected_model} | {r.test_mae:.2f} | {r.test_rmse:.2f} | {r.test_r2:.2f} |')
    lines += ['', 'MAE is the average absolute mistake in grade points. It is not a per-person confidence interval.', '', '## Error audit', '']
    for label, mask in [('Actual final grade = 0', test.G3 == 0), ('Actual final grade > 0', test.G3 > 0)]:
        e = errors.loc[mask]
        lines.append(f'- {label}: n={len(e)}, MAE={e.absolute_error.mean():.2f}.')
    lines += ['', 'These are descriptive post-evaluation slices, not new model-selection criteria. Zero grades are retained; their causes are not inferred.', '', 'Context-only results are retrospective associations. Questionnaire measurement timing is not established, so this is not an early-warning system.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(results.to_string(index=False))
    expanded = run_expanded_benchmarks(df)
    expanded.to_csv(out / 'expanded_benchmarks.csv', index=False, float_format='%.10g')
    print(expanded.to_string(index=False))


if __name__ == '__main__':
    save_report()
