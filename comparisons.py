"""Fixed-model comparisons; never tune using evaluation-school targets."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error
from grade_lab import candidates, load_data, split_data

COLUMNS = ['G1', 'G2']


def compare(train, test):
    rows = []
    for name, model in candidates(COLUMNS).items():
        if name == 'Mean baseline':
            continue
        model.fit(train[COLUMNS], train.G3)
        rows.append({'model': name, 'train_n': len(train), 'test_n': len(test),
                     'mae': mean_absolute_error(test.G3, model.predict(test[COLUMNS]))})
    return pd.DataFrame(rows)


def school_holdout(df):
    results = []
    for source, target in [('GP', 'MS'), ('MS', 'GP')]:
        train, test = df[df.school == source], df[df.school == target]
        if train.empty or test.empty:
            raise ValueError('Both GP and MS schools are required.')
        result = compare(train, test)
        result.insert(0, 'direction', f'{source} → {target}')
        results.append(result)
    return pd.concat(results, ignore_index=True)


def table(df):
    return '| ' + ' | '.join(df.columns) + ' |\n|' + '|'.join(['---'] * len(df.columns)) + '|\n' + '\n'.join(
        '| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in row) + ' |'
        for row in df.itertuples(index=False, name=None))


def main():
    root = Path(__file__).parent
    df = load_data()
    same = compare(*split_data(df))
    school = school_holdout(df)
    school.to_csv(root / 'reports/school_holdout.csv', index=False)
    (root / 'reports/SCHOOL_HOLDOUT.md').write_text(
        '# Cross-school holdout\n\nAll models receive G1 and G2. Fixed settings match the original experiment. '
        'Scaling is fitted only on the training school; no evaluation-school labels select models.\n\n' + table(school) +
        '\n\nMS has only 46 students versus 349 at GP. Training on MS is particularly unstable; '
        'these two historical schools do not establish generalization to other schools or years. '
        'Both directions are descriptive stress tests, not a new model-selection procedure.\n', encoding='utf-8')
    readme = root / 'README.md'
    text = readme.read_text(encoding='utf-8')
    start, end = '<!-- same-feature:start -->', '<!-- same-feature:end -->'
    section = start + '\n## Same inputs, different models\n\nAll three models receive G1 and G2 on the original seed-42 split. '
    section += 'The previous-grade rule ignores G1 by definition. These are fixed alternatives, not models selected using test scores.\n\n' + table(same)
    section += '\n\nThe earlier context-only comparison is a **feature ablation**, not a same-input model comparison. '
    section += '[Cross-school stress test](reports/SCHOOL_HOLDOUT.md).\n' + end
    if start in text:
        text = text[:text.index(start)] + section + text[text.index(end) + len(end):]
    else:
        text += '\n' + section + '\n'
    readme.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
