"""Full eligible-pool, score-improving selection under exact replay NAV variance."""
from __future__ import annotations
from collections import Counter
from dataclasses import replace
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

SELECTION_ONLY = {"", "stock_cap", "sector_cap", "portfolio_full"}


def eligible_rows(tested):
    return tested.loc[tested.exclusion.fillna("").isin(SELECTION_ONLY)].sort_values(
        ["tstat", "pair_id", "x"], kind="stable").copy()


def incidence(rows):
    stocks = sorted(set(rows.x) | set(rows.y))
    sectors = sorted(set(rows.sector))
    lookup = {s:i for i,s in enumerate(stocks)}
    sector_lookup = {s:i+len(stocks) for i,s in enumerate(sectors)}
    matrix = np.zeros((len(stocks)+len(sectors), len(rows)), dtype=np.int8)
    for j,row in enumerate(rows.itertuples()):
        matrix[lookup[row.x],j] = matrix[lookup[row.y],j] = 1
        matrix[sector_lookup[row.sector],j] = 1
    return matrix, np.array([2]*len(stocks)+[4]*len(sectors))


def nav_risk(paths, chosen, slots=20, cash_path=None, return_mask=None):
    """paths includes initial NAV=1 row; each column is a separate unit sleeve."""
    cash = np.ones(len(paths)) if cash_path is None else cash_path
    nav = paths[:, list(chosen)].sum(axis=1) + (slots-len(chosen))*cash
    if (nav <= 0).any() or not np.isfinite(nav).all():
        raise ValueError("Invalid aggregate calibration NAV")
    returns=nav[1:]/nav[:-1]-1
    if return_mask is not None: returns=returns[return_mask]
    return float(np.var(returns, ddof=1))


def solve(rows, paths, baseline, slots=20, cash_path=None, max_moves=100, return_mask=None, exact_limit=100000):
    rows = rows.reset_index(drop=True)
    scores = -rows.tstat.to_numpy(float)
    n,k = len(rows),len(baseline)
    assert paths.shape[1] == n and paths.shape[0] >= 3
    assert np.isfinite(scores).all() and np.isfinite(paths).all()
    matrix,caps = incidence(rows)
    baseline = tuple(sorted(baseline))
    assert len(set(baseline)) == k and (matrix[:,baseline].sum(axis=1) <= caps).all()
    budget = nav_risk(paths, baseline, slots, cash_path,return_mask)
    allowed = budget*(1+1e-10)+1e-18
    def feasible(ids):
        return (matrix[:,list(ids)].sum(axis=1) <= caps).all() and nav_risk(paths,ids,slots,cash_path,return_mask) <= allowed
    best = baseline
    baseline_score = float(scores[list(best)].sum())
    current_score = baseline_score
    upper = None
    status = "empty_baseline" if not k else "local_search"
    moves = 0
    if k and comb(n,k) <= exact_limit:
        for ids in combinations(range(n),k):
            score = float(scores[list(ids)].sum())
            if score > current_score+1e-12 and feasible(ids):
                best,current_score = ids,score
        status = "exact_enumeration"
        upper = current_score
    elif k:
        # The linear relaxation here drops ONLY risk, not incidence/cardinality.
        a = np.vstack([matrix, np.ones(n)])
        result = milp(-scores, integrality=np.ones(n), bounds=Bounds(0,1),
                      constraints=LinearConstraint(a,np.r_[np.full(len(caps),-np.inf),k],np.r_[caps,k]),
                      options={"time_limit":30., "mip_rel_gap":0.})
        if getattr(result,"mip_dual_bound",None) is not None and np.isfinite(result.mip_dual_bound):
            upper = float(-result.mip_dual_bound)
        if result.x is not None:
            proposal = tuple(np.flatnonzero(result.x>.5))
            if len(proposal)==k and feasible(proposal):
                value = float(scores[list(proposal)].sum())
                if value > current_score+1e-12:
                    best,current_score = proposal,value
                if result.status == 0:
                    status = "global_score_optimum_also_satisfies_risk"
        if status != "global_score_optimum_also_satisfies_risk":
            cash = np.ones(len(paths)) if cash_path is None else cash_path
            pair_lookup={tuple(sorted((r.x,r.y))):j for j,r in enumerate(rows.itertuples())}
            for _ in range(max_moves):
                ids = np.array(best,dtype=int)
                counts = matrix[:,ids].sum(axis=1)
                total = paths[:,ids].sum(axis=1)+(slots-k)*cash
                outside = np.setdiff1d(np.arange(n),ids,assume_unique=True)
                gain_best = 1e-12
                replacement = None
                for old in ids:
                    candidates = outside[scores[outside] > scores[old]+gain_best]
                    if not len(candidates):
                        continue
                    ok = (counts[:,None]-matrix[:,old,None]+matrix[:,candidates] <= caps[:,None]).all(axis=0)
                    candidates = candidates[ok]
                    for begin in range(0,len(candidates),256):
                        js = candidates[begin:begin+256]
                        nav = total[:,None]-paths[:,old,None]+paths[:,js]
                        returns=nav[1:]/nav[:-1]-1
                        if return_mask is not None: returns=returns[return_mask]
                        risks = np.var(returns,axis=0,ddof=1)
                        gains = scores[js]-scores[old]
                        good = (risks <= allowed) & (gains > gain_best)
                        if good.any():
                            j = int(np.argmax(np.where(good,gains,-np.inf)))
                            gain_best = float(gains[j])
                            replacement = {int(old)},{int(js[j])}
                # Degree-preserving four-stock reconnections escape a saturated
                # stock-cap trap without removing any candidate from the pool.
                selected_set=set(best)
                for left,right in combinations(ids,2):
                    a,b=rows.iloc[left][['x','y']]
                    c,d=rows.iloc[right][['x','y']]
                    if len({a,b,c,d})<4: continue
                    for edges in [((a,c),(b,d)),((a,d),(b,c))]:
                        proposed=[pair_lookup.get(tuple(sorted(e))) for e in edges]
                        if any(j is None or j in selected_set for j in proposed): continue
                        gain=float(scores[proposed].sum()-scores[[left,right]].sum())
                        if gain<=gain_best: continue
                        candidate=tuple(sorted((selected_set-{int(left),int(right)})|set(proposed)))
                        if len(candidate)==k and feasible(candidate):
                            gain_best=gain
                            replacement={int(left),int(right)},set(proposed)
                if replacement is None:
                    status = "one_exchange_and_four_stock_local_optimum"
                    break
                removed,added = replacement
                best = tuple(sorted((set(best)-removed)|added))
                current_score = float(scores[list(best)].sum())
                moves += 1
            else:
                status = "move_limit"
    risk = nav_risk(paths,best,slots,cash_path,return_mask)
    assert len(best)==k and feasible(best) and current_score >= baseline_score-1e-10
    return list(best), {"eligible_candidates":n,"selected_count":k,"baseline_score":baseline_score,
        "selected_score":current_score,"baseline_variance":budget,"selected_variance":risk,
        "variance_ratio":risk/budget if budget>0 else None,"score_upper_bound":upper,
        "score_gap_to_upper_bound":max(0.,upper-current_score) if upper is not None else None,
        "termination":status,"accepted_exchanges":moves,"candidate_top_n_limit":None,
        "risk_feasible":True,"per_stock_cap":2,"per_sector_cap":4}


def calibration(pair, prices, cfg):
    """Formation-end parameter calibration, not OOS trading performance."""
    from sp500_pairs.engine import simulate_pair
    start = prices.index[0]
    px,py = prices.iloc[0][[pair.x,pair.y]]
    # With no position, the ORIGINAL engine permits missing quotes and remains
    # in cash until a finite close can generate a next-close entry signal.
    # IPOs may pass the 99% coverage filter despite missing a block's first quote.
    # Do not add an unintended stricter candidate filter at calibration start.
    anchor = replace(pair, formation_end=start, last_z=pair.zscore(px,py),last_x=float(px),last_y=float(py))
    gross = cfg.with_updates({"costs.commission_bps":0.,"costs.slippage_bps":0.,"costs.borrow_annual_bps":0.})
    daily,_ = simulate_pair(anchor,prices.iloc[1:][[pair.x,pair.y]],1.,gross,"formation_risk")
    return np.r_[1.,daily.nav.to_numpy()]


def calibration_blocks(pairs,prices,cfg,window):
    """Independent original-duration sleeves; exclude artificial cash reset jumps.

    Fitted parameters use the full formation sample (explicit in-sample risk
    estimate). Within each historical six-month block the original state machine
    resets, so block_window does not accidentally become a twelve-month ban.
    """
    matrices,cash_paths,masks=[],[],[]
    begin=window.nominal_start-pd.DateOffset(months=cfg['schedule']['formation_months'])
    records=[]
    while begin<window.nominal_start:
        end=min(begin+pd.DateOffset(months=cfg['schedule']['trading_months']),window.nominal_start)
        block=prices.loc[(prices.index>=begin)&(prices.index<end)]
        if len(block)<3: raise ValueError('Insufficient observations in calibration block')
        matrix=np.column_stack([calibration(p,block,cfg) for p in pairs]) if pairs else np.empty((len(block),0))
        if matrices: masks.append(np.array([False]))
        masks.append(np.ones(len(matrix)-1,dtype=bool))
        matrices.append(matrix)
        days=(block.index-block.index[0]).days.to_numpy()
        cash_paths.append((1+cfg['portfolio']['cash_annual_rate'])**(days/365))
        records.append({'start':str(block.index[0].date()),'end':str(block.index[-1].date()),'returns':len(block)-1})
        begin=end
    return np.vstack(matrices),np.concatenate(cash_paths),np.concatenate(masks),records
