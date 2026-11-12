import json
import ast
from pathlib import Path

import pandas as pd
import pyarrow as pa
from streamlit.testing.v1 import AppTest

from grade_lab import load_data, split_data

APP = Path(__file__).parents[1] / 'app.py'


def test_evidence_chart_and_five_zero_grade_records():
    app = AppTest.from_file(str(APP)).run(timeout=60)
    assert not app.exception
    evidence = app.tabs[1]

    chart_proto = evidence.get('vega_lite_chart')[0].proto
    chart = json.loads(chart_proto.spec)
    reference, points = chart['layer']
    dataset = next(item for item in chart_proto.datasets if item.name == reference['data']['name'])
    reference_data = pa.ipc.open_stream(dataset.data.data).read_all().to_pylist()
    assert reference_data == [{'Actual': 0, 'Predicted': 0}, {'Actual': 20, 'Predicted': 20}]
    color = points['encoding']['color']
    assert color['field'] == 'Group'
    assert set(color['scale']['domain']) == {'Zero final grade', 'Nonzero final grade'}
    assert len(set(color['scale']['range'])) == 2

    _, test = split_data(load_data())
    expected = test.loc[test.G3.eq(0), ['G1', 'G2']].copy()
    expected['Predicted'] = expected.G2.astype(float)
    expected['Actual'] = 0
    assert len(expected) == 5
    table, = [item.value for item in evidence.dataframe
              if list(item.value.columns) == ['G1', 'G2', 'Predicted', 'Actual']]
    pd.testing.assert_frame_equal(
        table.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False,
    )


def test_all_tabs_render_and_evidence_labels_bootstrap_spread():
    app = AppTest.from_file(str(APP)).run(timeout=60)
    assert not app.exception
    studio, evidence, learn = app.tabs
    assert studio.selectbox[0].options == [
        'Ridge regression', 'Small decision tree', 'Previous-grade baseline',
    ]
    assert any('training examples have both grades' in text.value for text in studio.caption)
    assert len(learn.expander) == 6
    intervals, = [item.value for item in evidence.dataframe
                  if '95% lower' in item.value.columns]
    assert intervals['Selected model'].tolist() == ['Previous-grade baseline', 'Ridge regression']
    report_dir = APP.parent / 'reports'
    published = pd.read_csv(report_dir / 'bootstrap_intervals.csv').set_index('experiment')
    expected = published.loc[intervals.Inputs, ['test_mae', 'ci_low', 'ci_high']]
    expected.columns = ['Test MAE', '95% lower', '95% upper']
    pd.testing.assert_frame_equal(
        intervals[list(expected.columns)].reset_index(drop=True),
        expected.round(2).reset_index(drop=True),
    )
    metadata = json.loads((report_dir / 'paired_comparison.json').read_text(encoding='utf-8'))
    label = (f"{metadata['replicates']:,} paired resamples of the {metadata['test_rows']} "
             f"test records (seed {metadata['bootstrap_seed']})")
    assert any(label in text.value for text in evidence.caption)


def test_page_config_uses_plain_title_and_default_icon():
    module = ast.parse(APP.read_text(encoding='utf-8'))
    config, = [node.value for node in module.body
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute)
               and node.value.func.attr == 'set_page_config']
    options = {item.arg: ast.literal_eval(item.value) for item in config.keywords}
    assert options['page_title'] == 'Grade Lab'
    assert 'page_icon' not in options
