"""Describe development grades. Run with python zero_grades.py."""

import pandas as pd

from grade_lab import ROOT, load_data, split_data

SCOPE = (
    'This describes the development data. The comparison cannot establish causes. '
    'The five held-out zero-grade cases were excluded.'
)


def summarize_zero_grades(development):
    """Compare earlier grades within the supplied development rows."""
    zero = development.G3.eq(0)
    rows = []
    for label, group in [('Zero', development.loc[zero]), ('Above zero', development.loc[~zero])]:
        rows.append({
            'Final grade': label,
            'Records': len(group),
            'G1 median': group.G1.median(),
            'G1 range': f'{group.G1.min()} to {group.G1.max()}',
            'G2 median': group.G2.median(),
            'G2 range': f'{group.G2.min()} to {group.G2.max()}',
            'G2 equals zero': int(group.G2.eq(0).sum()),
        })
    return pd.DataFrame(rows)


def save_zero_grade_report(output=ROOT / 'reports' / 'ZERO_GRADES.md'):
    development, _ = split_data(load_data())
    summary = summarize_zero_grades(development)
    header = '| ' + ' | '.join(summary.columns) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(summary.columns)) + ' |'
    rows = ['| ' + ' | '.join(map(str, row)) + ' |'
            for row in summary.itertuples(index=False, name=None)]
    lines = [
        '# Zero-grade spotlight', '',
        f'The seed-42 development split contains {len(development)} mathematics records.',
        SCOPE, '', header, separator, *rows,
    ]
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    save_zero_grade_report()
