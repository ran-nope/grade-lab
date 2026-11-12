"""Run the full experiment for five fixed seeds with python seed_sweep.py."""

import pandas as pd

from grade_lab import ROOT, load_data, run_experiments

SWEEP_SEEDS = (1, 2, 3, 4, 5)


def save_seed_sweep(output=ROOT / 'reports' / 'seed_sweep.csv'):
    df = load_data()
    runs = []
    for seed in SWEEP_SEEDS:
        _, results, _, _ = run_experiments(df, seed=seed)
        runs.append(results.assign(split_seed=seed))

    sweep = pd.concat(runs, ignore_index=True)
    seed = sweep.pop('split_seed')
    sweep.insert(0, 'split_seed', seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output, index=False)


if __name__ == '__main__':
    save_seed_sweep()
