"""Original eligibility, with safe rejection before the expensive EG test.

With multiple_testing=none and require_i1=false, a pair failing the unchanged
correlation/hedge/half-life screens cannot enter the eligible pool regardless of
its EG result. Every survivor is fitted by the unmodified upstream function.
No top-N screen, new filter, or estimated EG statistic is introduced.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from sp500_pairs.data import normalize_ticker
from sp500_pairs.formation import fit_direction, Pair


def pair_objects(rows, prices, formation_end):
    pairs = []
    for r in rows.itertuples():
        px, py = float(prices[r.x].iloc[-1]), float(prices[r.y].iloc[-1])
        z = (np.log(py)-r.alpha-r.beta*np.log(px)-r.mean)/r.std
        pairs.append(Pair(r.pair_id, r.x, r.y, r.sector, r.alpha, r.beta, r.mean, r.std,
                          formation_end, z, px, py))
    return pairs


def eligible_formation(data, window, cfg):
    f, d = cfg['formation'], cfg['data']
    assert f['orientation'] == 'fixed' and f['multiple_testing'] == 'none' and not f['require_i1']
    prices = data.prices.loc[window.formation_dates]
    universe, info = data.universe(window.formation_dates[-1])
    unknown = {s.casefold() for s in d['unknown_sectors']}
    excluded = {normalize_ticker(t) for t in d['excluded_tickers']}
    usable, stock_audit = [], []
    for item in universe.itertuples():
        reason = ''
        if item.ticker in excluded: reason = 'explicit_exclusion'
        elif item.ticker not in prices: reason = 'no_price_column'
        elif item.sector.casefold() in unknown: reason = 'unknown_sector'
        elif prices[item.ticker].notna().mean() < d['min_formation_coverage'] or prices[item.ticker].count() < d['min_formation_observations']:
            reason = 'formation_coverage'
        stock_audit.append(dict(ticker=item.ticker, sector=item.sector, exclusion=reason))
        if not reason: usable.append((item.ticker, item.sector))
    logs = np.log(prices[[t for t, _ in usable]])
    correlations = logs.diff().corr()
    groups = pd.DataFrame(usable, columns=['ticker', 'sector'])
    records, eligible = [], []
    for sector, group in groups.groupby('sector', sort=True):
        for x, y in combinations(sorted(group.ticker), 2):
            pid = f'{x}__{y}'
            reason = ''
            # A wide numerical margin defers boundary cases to the exact fit.
            if f['correlation_filter'] and (not np.isfinite(correlations.loc[x,y]) or correlations.loc[x,y] < f['min_return_correlation']-1e-10):
                reason = 'correlation'
            if not reason:
                frame = logs[[x,y]].dropna()
                if len(frame) < d['min_formation_observations'] or len(frame)/len(logs) < d['min_formation_coverage']:
                    reason = 'pair_coverage'
                else:
                    xv, yv = frame[x].to_numpy(), frame[y].to_numpy()
                    design = np.column_stack([np.ones(len(frame)), xv])
                    try:
                        alpha, beta = np.linalg.lstsq(design, yv, rcond=None)[0]
                        spread = yv-alpha-beta*xv
                        phi = float(np.linalg.lstsq(np.column_stack([np.ones(len(spread)-1), spread[:-1]]), spread[1:], rcond=None)[0][1])
                        hl = -np.log(2)/np.log(phi) if 0 < phi < 1 else np.nan
                        if beta < f['beta_min']-1e-10 or beta > f['beta_max']+1e-10:
                            reason = 'hedge_ratio'
                        elif f['half_life_filter'] and (not np.isfinite(hl) or hl < f['half_life_min']-1e-8 or hl > f['half_life_max']+1e-8):
                            reason = 'half_life'
                    except (ValueError, np.linalg.LinAlgError):
                        pass  # Let the original fitter determine failures.
            if reason:
                records.append(dict(pair_id=pid,x=x,y=y,sector=sector,exclusion=reason,eg_tested=False))
                continue
            row = dict(pair_id=pid, sector=sector, **fit_direction(logs, x, y, cfg))
            reasons = []
            if row['error']: reasons.append('fit_error')
            else:
                if row['n_observations'] < d['min_formation_observations'] or row['n_observations']/len(logs) < d['min_formation_coverage']: reasons.append('pair_coverage')
                if row['pvalue'] >= f['cointegration_alpha']: reasons.append('cointegration')
                if f['correlation_filter'] and (not np.isfinite(row['return_correlation']) or row['return_correlation'] <= f['min_return_correlation']): reasons.append('correlation')
                if not f['beta_min'] <= row['beta'] <= f['beta_max']: reasons.append('hedge_ratio')
                if f['half_life_filter'] and (not np.isfinite(row['half_life']) or not f['half_life_min'] <= row['half_life'] <= f['half_life_max']): reasons.append('half_life')
                if not np.isfinite(prices[x].iloc[-1]) or not np.isfinite(prices[y].iloc[-1]): reasons.append('missing_formation_end_price')
            row.update(exclusion=';'.join(reasons), eg_tested=True)
            records.append(row)
            if not reasons: eligible.append(row)
    rows = pd.DataFrame(eligible).sort_values(['tstat','pair_id','x'],kind='stable').reset_index(drop=True)
    return rows, pd.DataFrame(records), pd.DataFrame(stock_audit), {**info, 'candidate_pairs':len(records),
        'eligible_pairs':len(rows), 'eg_tests':sum(r['eg_tested'] for r in records),
        'i1_diagnostic_omitted':True, 'i1_filter_enabled':False}
