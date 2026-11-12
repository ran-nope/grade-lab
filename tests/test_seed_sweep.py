import pandas as pd

import seed_sweep
from grade_lab import FEATURES, run_experiments


def test_seed_sweep_writes_every_fixed_run(tmp_path, monkeypatch):
    observed = {}

    def record_run(df, seed):
        run = run_experiments(df, seed=seed)
        assert seed not in observed
        observed[seed] = run[1].copy()
        return run

    monkeypatch.setattr(seed_sweep, 'run_experiments', record_run)
    output = tmp_path / 'reports' / 'seed_sweep.csv'
    seed_sweep.save_seed_sweep(output)
    report = pd.read_csv(output)

    assert list(observed) == [1, 2, 3, 4, 5]
    assert report.split_seed.tolist() == [seed for seed in observed for _ in FEATURES]
    for seed, expected in observed.items():
        rows = report.loc[report.split_seed.eq(seed)].drop(columns='split_seed')
        pd.testing.assert_frame_equal(rows.reset_index(drop=True), expected)
