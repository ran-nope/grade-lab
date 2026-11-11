"""Small, reproducible ML experiment. Read this after the notebook."""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'student-mat.csv'
URL = 'https://archive.ics.uci.edu/static/public/320/student+performance.zip'
FEATURES = {
    'Earlier grades': ['G1', 'G2'],
    'Study context only': ['studytime', 'failures', 'traveltime'],
    'Grades + context': ['G1', 'G2', 'studytime', 'failures', 'traveltime'],
}


class PreviousGrade(RegressorMixin, BaseEstimator):
    """A strong baseline: predict that the final grade equals G2."""
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return X['G2'].to_numpy(dtype=float)


def download_data():
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        request = urllib.request.Request(URL, headers={'User-Agent': 'GradeLab/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        archive = zipfile.ZipFile(io.BytesIO(payload))
        if 'student-mat.csv' not in archive.namelist():
            archive = zipfile.ZipFile(io.BytesIO(archive.read('student.zip')))
        DATA.write_bytes(archive.read('student-mat.csv'))
    return DATA


def load_data():
    df = pd.read_csv(download_data(), sep=';')
    required = sorted(set(sum(FEATURES.values(), []) + ['G3']))
    if df[required].isna().any().any():
        raise ValueError('Missing values found: inspect data before training.')
    if not df[['G1', 'G2', 'G3']].apply(lambda s: s.between(0, 20).all()).all():
        raise ValueError('Grades must lie between 0 and 20.')
    return df


def split_data(df):
    # Identical split for every experiment; no subject datasets are combined.
    return train_test_split(df, test_size=0.2, random_state=42)


def candidates(columns):
    models = {
        'Mean baseline': DummyRegressor(strategy='mean'),
        'Ridge regression': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'Small decision tree': DecisionTreeRegressor(max_depth=3, min_samples_leaf=15, random_state=42),
    }
    if 'G2' in columns:
        models['Previous-grade baseline'] = PreviousGrade()
    return models


def run_experiments(df):
    train, test = split_data(df)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
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
    axes[1, 0].scatter(test.G3, predictions['Earlier grades'], alpha=.6, color='#4361ee')
    axes[1, 0].plot([0, 20], [0, 20], '--', color='#222')
    axes[1, 0].set(title='Unseen students: prediction vs reality', xlabel='Actual grade', ylabel='Predicted grade')
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


if __name__ == '__main__':
    save_report()
