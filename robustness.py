"""Supplementary uncertainty checks; run with python robustness.py.

Fixed seeds are declared before running, never searched for a favorable result.
Outputs do not change the original seed-42 model choice.
"""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from grade_lab import FEATURES, ROOT, candidates, load_data, run_experiments, split_data

BOOTSTRAP_SEED = 2026
BOOTSTRAP_REPLICATES = 10000
BASELINE_RIDGE_REPLICATES = 1000
SENSITIVITY_SEEDS = (7, 17, 27, 37, 47, 57, 67, 77, 87, 97)


def fit_conformal_model(model, X_train, y_train, alpha=0.10, random_state=42):
    """Pool five development-fold residuals, then fit the center on all development rows."""
    if not 0 < alpha < 1 or len(X_train) < 5:
        raise ValueError('Use alpha between 0 and 1 and at least five development rows.')
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    prediction = cross_val_predict(model, X_train, y_train, cv=cv)
    residuals = np.sort(np.asarray(y_train) - prediction)
    n = len(residuals)
    low_rank = int(np.floor((n + 1) * (alpha / 2)))
    high_rank = int(np.ceil((n + 1) * (1 - alpha / 2)))
    q_low = residuals[low_rank - 1] if low_rank > 0 else -np.inf
    q_high = residuals[high_rank - 1] if high_rank <= n else np.inf
    return clone(model).fit(X_train, y_train), (float(q_low), float(q_high))


def conformal_bounds(predictions, offsets):
    predictions = np.asarray(predictions)
    q_low, q_high = offsets
    # Include the point prediction when both residual quantiles have the same sign.
    lower = np.clip(predictions + min(0, q_low), 0, 20)
    upper = np.clip(predictions + max(0, q_high), 0, 20)
    return lower, upper


def conformal_coverage(lower, upper, y_eval):
    actual = np.asarray(y_eval)
    if actual.shape != lower.shape or not actual.size or not np.isfinite(actual).all():
        raise ValueError('Evaluation grades must be finite and match the interval rows.')
    return float(np.mean((actual >= lower) & (actual <= upper)))


def compute_conformal_intervals(model, X_train, y_train, X_eval, alpha=0.10,
                                random_state=42, *, y_eval=None):
    """Return bounds, signed offsets, and coverage when evaluation labels are supplied."""
    fitted, offsets = fit_conformal_model(model, X_train, y_train, alpha, random_state)
    lower, upper = conformal_bounds(fitted.predict(X_eval), offsets)
    coverage = conformal_coverage(lower, upper, y_eval) if y_eval is not None else None
    return lower, upper, offsets, coverage


def studio_calibration(development, held_out):
    """Calibrate each fixed studio model and evaluate its intervals on held-out rows."""
    columns = FEATURES['Earlier grades']
    calibrated, rows = {}, []
    for name, model in candidates(columns).items():
        if name == 'Mean baseline':
            continue
        fitted, offsets = fit_conformal_model(model, development[columns], development.G3)
        calibrated[name] = (fitted, offsets)
        lower, upper = conformal_bounds(fitted.predict(held_out[columns]), offsets)
        rows.append({'model': name, 'nominal_coverage': 0.90,
                     'test_coverage': conformal_coverage(lower, upper, held_out.G3),
                     'test_rows': len(held_out), 'q_low': offsets[0], 'q_high': offsets[1],
                     'mean_width': (upper - lower).mean()})
    return calibrated, pd.DataFrame(rows)


def bootstrap_mae(actual, predictions, n_boot=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Paired percentile bootstrap over test rows, conditional on fitted models.

    Same sampled rows for every model; not individual prediction intervals.
    Retraining and model-selection uncertainty are not included.
    """
    actual = np.asarray(actual, dtype=float)
    if actual.ndim != 1 or len(actual) == 0 or not np.isfinite(actual).all():
        raise ValueError('Expected a nonempty finite target vector.')
    if n_boot < 2:
        raise ValueError('At least two bootstrap replicates are required.')
    indexes = np.random.default_rng(seed).integers(0, len(actual), size=(n_boot, len(actual)))
    rows, samples = [], {}
    for name, predicted in predictions.items():
        predicted = np.asarray(predicted, dtype=float)
        if predicted.shape != actual.shape or not np.isfinite(predicted).all():
            raise ValueError('Predictions must match the finite target vector.')
        errors = np.abs(actual - predicted)
        samples[name] = errors[indexes].mean(axis=1)
        lo, hi = np.quantile(samples[name], [.025, .975])
        rows.append({'experiment': name, 'test_mae': errors.mean(), 'ci_low': lo, 'ci_high': hi})
    return pd.DataFrame(rows), samples


def baseline_ridge_intervals(actual, predictions):
    """Summarize fixed-test MAE spread for the two selected approaches."""
    intervals, _ = bootstrap_mae(
        actual,
        {'Earlier grades': predictions['Earlier grades'],
         'Study context only': predictions['Study context only']},
        n_boot=BASELINE_RIDGE_REPLICATES,
    )
    intervals.insert(1, 'selected_model', ['Previous-grade baseline', 'Ridge regression'])
    return intervals.assign(
        test_rows=len(actual), resamples=BASELINE_RIDGE_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )


def save_robustness():
    df = load_data()
    _, _, _, predictions = run_experiments(df)
    _, test = split_data(df)
    intervals, samples = bootstrap_mae(test.G3, predictions)
    delta = samples['Study context only'] - samples['Earlier grades']
    paired = {'comparison': 'context-only minus earlier-grades MAE',
              'estimate': float(np.mean(np.abs(test.G3.to_numpy() - predictions['Study context only'])) - np.mean(np.abs(test.G3.to_numpy() - predictions['Earlier grades']))),
              'ci_low': float(np.quantile(delta, .025)), 'ci_high': float(np.quantile(delta, .975)),
              'replicates': BOOTSTRAP_REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED,
              'test_rows': len(test)}
    all_runs = []
    for seed in SENSITIVITY_SEEDS:
        _, result, _, _ = run_experiments(df, seed=seed)
        result.insert(0, 'split_seed', seed)
        result['zero_grade_test_rows'] = int(split_data(df, seed)[1].G3.eq(0).sum())
        all_runs.append(result)
    runs = pd.concat(all_runs, ignore_index=True)
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    development, _ = split_data(df)
    _, coverage = studio_calibration(development, test)
    por_development, por_test = split_data(load_data('por'))
    _, por_coverage = studio_calibration(por_development, por_test)
    pd.concat([coverage.assign(subject='mat'), por_coverage.assign(subject='por')], ignore_index=True).to_csv(
        out / 'conformal_coverage.csv', index=False, float_format='%.10g',
    )
    intervals.to_csv(out / 'bootstrap_intervals.csv', index=False)
    baseline_ridge_intervals(test.G3, predictions).to_csv(
        out / 'baseline_ridge_bootstrap.csv', index=False,
    )
    runs.to_csv(out / 'split_sensitivity.csv', index=False)
    (out / 'paired_comparison.json').write_text(json.dumps(paired, indent=2) + '\n', encoding='utf-8')
    lines = ['# How stable are the results?', '', '## Fixed-test bootstrap', '',
             '95% percentile intervals from 10,000 paired resamples of the 79 seed-42 test rows (bootstrap seed 2026). Models are held fixed. These intervals describe test-row sampling variability under an IID assumption; they omit retraining, model-selection, school clustering, and distribution-shift uncertainty. They are not intervals for individual students.', '',
             '| Experiment | Test MAE | 95% bootstrap interval |', '|---|---:|---:|']
    for r in intervals.itertuples():
        lines.append(f'| {r.experiment} | {r.test_mae:.2f} | {r.ci_low:.2f}–{r.ci_high:.2f} |')
    lines += ['', f"The paired context-minus-grades MAE difference is {paired['estimate']:.2f} points (95% interval {paired['ci_low']:.2f}–{paired['ci_high']:.2f}). This compares different information sets, not ridge against the baseline with identical inputs.", '',
              '## Additional split sensitivity', '',
              'This supplementary analysis was added after inspecting the original seed-42 results. All ten seeds were fixed before this analysis ran: ' + ', '.join(map(str, SENSITIVITY_SEEDS)) + '. Each split reruns training-only five-fold selection, then evaluates its selected model on that split’s held-out rows. No seed is selected as the best result.', '',
              'The splits overlap, so their scores are correlated. The range below is descriptive, not a confidence interval or ten independent replications. This is a sensitivity analysis, not new untouched external validation.', '',
              '| Experiment | Mean MAE across splits | Min–max MAE |', '|---|---:|---:|']
    for name, group in runs.groupby('experiment', sort=False):
        lines.append(f'| {name} | {group.test_mae.mean():.2f} | {group.test_mae.min():.2f}–{group.test_mae.max():.2f} |')
    lines += ['', 'All selected models, seeds, zero-grade counts, and scores are in [split_sensitivity.csv](split_sensitivity.csv). None of these checks changes the original winner or app defaults. With only five zero-grade cases in the original test set, that subgroup’s error remains especially uncertain.']
    (out / 'ROBUSTNESS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(intervals.to_string(index=False))
    print(runs.groupby('experiment').test_mae.agg(['mean', 'min', 'max']).to_string())


if __name__ == '__main__':
    save_robustness()
