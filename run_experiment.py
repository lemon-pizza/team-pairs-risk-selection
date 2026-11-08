"""Replay the published selections or solve the same experiment afresh."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle

# Set numeric thread limits before NumPy is imported.
import extension_experiment as engine
from experiment import HERE, UPSTREAM, sha, verify_upstream, write
import numpy as np
import pandas as pd
from sp500_pairs.config import load_config
from sp500_pairs.data import load_data


MODELS = ['BASELINE_B1'] + [f'SCORE_RISK_B{b}' for b in range(1, 6)]
CASES = [(months, False) for months in range(1, 7)] + [(6, True)]


def input_receipt(args, reference):
    paths = {key: getattr(args, key).resolve() for key in ('prices', 'membership', 'sectors')}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if args.mode == 'replay' and sha(path) != reference['inputs'][key]:
            raise ValueError(f'{key} CSV differs from the published input. Use the matching file or --mode solve.')
    cfg = load_config(UPSTREAM / 'configs/baseline.toml').with_updates({
        f'data.{key}_file': str(path) for key, path in paths.items()
    })
    data = load_data(cfg)
    return {'config': cfg.values, 'data': data.metadata, 'upstream_commit': verify_upstream()['commit']}


def install_frozen_selections(cases, reference):
    bundle_path = HERE / 'frozen_selections.json.gz'
    if sha(bundle_path) != reference['frozen_selections_sha256']:
        raise ValueError('Frozen selection checksum mismatch')
    with gzip.open(bundle_path, 'rt', encoding='utf-8') as handle:
        bundle = json.load(handle)
    if bundle['format'] != 1:
        raise ValueError('Unknown frozen selection format')

    indexes = {}
    for date, records in bundle['formation'].items():
        rows = pd.DataFrame(records)
        target = engine.formation_path(date)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('wb') as handle:
            pickle.dump(rows, handle)
        target.with_suffix('.sha256').write_text(sha(target) + '\n')
        indexes[date] = {pair_id: idx for idx, pair_id in enumerate(rows.pair_id)}

    windows = []
    for months, uncapped in cases:
        for window in engine.windows_for(engine.DATA.prices.index, engine.CFG, months):
            folder = engine.case_path(months, uncapped)
            rows = engine.load_rows(window.id)
            for key in ['baseline'] + [f'b{b}' for b in range(1, 6)]:
                target = folder / 'selection' / f'{window.id}_{key}.json'
                record = bundle['selections'][target.relative_to(engine.OUT).as_posix()]
                ids = [indexes[window.id][pair_id] for pair_id in record['selected_ids']]
                picked = rows.iloc[ids]
                cap = 1 if key == 'baseline' else int(key[-1])
                stock_counts = Counter([*picked.x, *picked.y])
                if len(set(ids)) != len(ids) or max(stock_counts.values(), default=0) > cap:
                    raise ValueError(f'Invalid stock degree in {target.name}')
                if max(Counter(picked.sector).values(), default=0) > 4 or (not uncapped and len(ids) > 20):
                    raise ValueError(f'Invalid sector or portfolio cap in {target.name}')
                write(target, {**record, 'selected_indices': ids})
            windows.append((months, window.id))
    return sorted(set(windows))


def verify_results(results, reference):
    path = HERE / 'reference_results.csv'
    if sha(path) != reference['reference_results_sha256']:
        raise ValueError('Reference results checksum mismatch')
    keys = ['months', 'uncapped', 'model', 'one_way_bps']
    baseline = pd.read_csv(path).set_index(keys)
    observed = results.set_index(keys)
    if not observed.index.is_unique or not observed.index.isin(baseline.index).all():
        raise ValueError('Unexpected result cases')
    expected = baseline.loc[observed.index]
    for metric in ('cumulative_return', 'annualized_volatility', 'sharpe', 'max_drawdown', 'completed_trades'):
        np.testing.assert_allclose(observed[metric], expected[metric], rtol=1e-9, atol=1e-11, err_msg=metric)
    return {'matched_cases': len(observed), 'matches_published_results': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('prices', 'membership', 'sectors', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--mode', choices=['replay', 'solve'], default='replay')
    parser.add_argument('--arm', choices=['all', 'base', 'uncapped', 'cadence'], default='all')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--solver-seconds', type=float, default=15)
    args = parser.parse_args()
    if args.workers < 1 or args.solver_seconds <= 0:
        parser.error('Workers and solver time must be positive')
    args.output = args.output.resolve()
    args.source = args.output / '_inputs'
    reference = json.loads((HERE / 'reference.json').read_text())
    verify_upstream()
    fingerprint = input_receipt(args, reference)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.source / 'experiment.json', {'fingerprint': fingerprint})
    engine.initialize(str(args.source), str(args.output))
    cases = {'all': CASES, 'base': [(6, False)], 'uncapped': [(6, True)], 'cadence': CASES[:-1]}[args.arm]

    if args.mode == 'replay':
        unit_tasks = install_frozen_selections(cases, reference)
    else:
        tasks = [(months, window.id, uncapped, args.solver_seconds)
                 for months, uncapped in cases
                 for window in engine.windows_for(engine.DATA.prices.index, engine.CFG, months)]
        engine.pooled(engine.formation_work, sorted({task[1] for task in tasks}), args)
        engine.pooled(engine.selection_work, tasks, args)
        unit_tasks = sorted({(months, date) for months, date, *_ in tasks})

    engine.pooled(engine.execution_unit_work, unit_tasks, args)
    tasks = [(months, uncapped, model) for months, uncapped in cases for model in MODELS]
    groups = engine.pooled(engine.backtest_work, tasks, args)
    results = pd.DataFrame([row for group in groups for row in group])
    results = results.sort_values(['uncapped', 'months', 'model', 'one_way_bps'])
    results.to_csv(args.output / 'comparison.csv', index=False)
    validation = verify_results(results, reference) if args.mode == 'replay' else {'computed_cases': len(results)}
    write(args.output / 'validation.json', {'mode': args.mode, **validation})
    print(json.dumps(validation), flush=True)


if __name__ == '__main__':
    main()
