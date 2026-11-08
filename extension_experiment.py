"""Two separate extensions of the verified nonoverlap-risk experiment.

1. No global pair cap; six-month cadence, fully invested NAV/K.
2. One-through-six month cadence; original 20-slot capital and pair cap.
Every cadence has its own contemporaneous nonoverlap formation risk benchmark.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import pickle
import shutil
import time
import numpy as np
import pandas as pd
from experiment import HERE, sha, write, verify_upstream
import prefetch
from selection import eligible_rows, calibration_blocks
from extension_formation import eligible_formation, pair_objects
from extension_solver import greedy, optimize, risk, structure
from extension_replay import fast_calibration_blocks
from sp500_pairs.schedule import Window
from sp500_pairs.engine import simulate_pair
from sp500_pairs.runner import metrics

DATA=CFG=OUT=SOURCE=None
DEFAULT_OUT=HERE/'output/extensions_20261106'


def initialize(source, out):
    global DATA,CFG,OUT,SOURCE
    SOURCE,OUT=Path(source),Path(out)
    prefetch.initialize(source)
    DATA,CFG=prefetch.DATA,prefetch.CFG


def windows_for(calendar, cfg, months):
    # Same Jan-2016 anchor and Dec-2024 end for every cadence. Include the final
    # partial window rather than silently dropping 2024Q4 for the five-month arm.
    start=pd.Timestamp(cfg['schedule']['start']); end=pd.Timestamp(cfg['schedule']['end'])+pd.Timedelta(days=1)
    result=[]; begin=start
    while begin<end:
        stop=min(begin+pd.DateOffset(months=months),end)
        first=begin-pd.DateOffset(months=cfg['schedule']['formation_months'])
        formation=calendar[(calendar>=first)&(calendar<begin)]
        trading=calendar[(calendar>=begin)&(calendar<stop)]
        assert len(formation)>=cfg['data']['min_formation_observations'] and len(trading)>=2
        result.append(Window(begin.strftime('%Y-%m-%d'),formation,trading,begin,stop-pd.Timedelta(days=1)))
        begin=stop
    return result


def formation_path(date): return OUT/'formation'/date/'eligible.pkl'


def formation_work(date):
    target=formation_path(date)
    if target.exists():
        assert sha(target)==target.with_suffix('.sha256').read_text().strip()
        return dict(stage='formation',window=date,status='cached')
    window=next(w for w in windows_for(DATA.prices.index,CFG,1) if w.id==date)
    started=time.monotonic()
    rows,tested,stocks,audit=eligible_formation(DATA,window,CFG)
    original=SOURCE/'cache'/date/'formation.pkl'
    if original.exists():
        assert sha(original)==original.with_suffix('.sha256').read_text().strip()
        with original.open('rb') as f: old=pickle.load(f)
        expected=eligible_rows(old[1]).reset_index(drop=True)
        assert list(rows.pair_id)==list(expected.pair_id),'Eligibility shortcut changed original candidate pool'
        for column in ['alpha','beta','mean','std','tstat','pvalue','half_life','return_correlation']:
            np.testing.assert_allclose(rows[column],expected[column],rtol=0,atol=1e-12)
        audit['parity_against_original_cache']=True
    else: audit['parity_against_original_cache']=None
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as f:pickle.dump(rows,f)
    target.with_suffix('.sha256').write_text(sha(target)+'\n')
    tested.to_csv(target.parent/'screen_audit.csv',index=False)
    stocks.to_csv(target.parent/'stock_audit.csv',index=False)
    audit.update(window=date,formation_start=str(window.formation_dates[0].date()),
        formation_end=str(window.formation_dates[-1].date()),first_trading_date=str(window.trading_dates[0].date()),
        formation_sha256=sha(target),elapsed_seconds=time.monotonic()-started)
    write(target.parent/'audit.json',audit)
    return dict(stage='formation',**audit)


def load_rows(date):
    path=formation_path(date)
    if not path.exists() or not path.with_suffix('.sha256').exists():
        original=SOURCE/'cache'/date/'formation.pkl'
        if original.exists():
            assert sha(original)==original.with_suffix('.sha256').read_text().strip()
            with original.open('rb') as f:return eligible_rows(pickle.load(f)[1]).reset_index(drop=True)
        raise FileNotFoundError(path)
    assert sha(path)==path.with_suffix('.sha256').read_text().strip()
    with path.open('rb') as f:return pickle.load(f)


def case_path(months,uncapped):
    return OUT/('uncapped_6m' if uncapped else f'cadence_{months}m')


def pack_risk(window,months,rows):
    target=OUT/'risk'/f'{months}m'/f'{window.id}.npz'
    if not target.exists():
        original=SOURCE/'cache'/window.id/'risk_paths.npz'
        if months==6 and original.exists():
            assert sha(original)==original.with_suffix('.sha256').read_text().strip()
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original,target)
        else:
            current=CFG.with_updates({'schedule.trading_months':months})
            prices=DATA.prices.loc[window.formation_dates]
            pairs=pair_objects(rows,prices,window.formation_dates[-1])
            paths,cash,mask,blocks,parity=fast_calibration_blocks(pairs,prices,current,window)
            target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,paths=paths,cash_path=cash,return_mask=mask,
                pair_ids=np.array(list(rows.pair_id),dtype=str),blocks=np.array(json.dumps(blocks)))
            write(target.with_suffix('.parity.json'),parity)
        target.with_suffix('.sha256').write_text(sha(target)+'\n')
    assert sha(target)==target.with_suffix('.sha256').read_text().strip()
    with np.load(target,allow_pickle=False) as p:
        assert list(p['pair_ids'])==list(rows.pair_id)
        return p['paths'],p['cash_path'],p['return_mask']


def selection_work(task):
    months,date,uncapped,limit=task
    window=next(w for w in windows_for(DATA.prices.index,CFG,months) if w.id==date)
    rows=load_rows(date);paths,cash,mask=pack_risk(window,months,rows)
    cap=None if uncapped else 20;folder=case_path(months,uncapped)
    baseline=greedy(rows,1,cap);warm=baseline
    write(folder/'selection'/f'{date}_baseline.json',dict(selected_indices=baseline,
        selected_ids=list(rows.pair_id.iloc[baseline]),variance=risk(paths,baseline,cash,mask,cap),
        score=float(-rows.tstat.iloc[baseline].sum())))
    audits=[]
    for b in range(1,6):
        target=folder/'selection'/f'{date}_b{b}.json'
        if target.exists():
            record=json.loads(target.read_text());ids,audit=record['selected_indices'],record['audit']
        elif not uncapped and months==6 and (HERE/'output/nonoverlap_risk/selection'/f'{date}_b{b}.json').is_file():
            old=HERE/'output/nonoverlap_risk/selection'/f'{date}_b{b}.json'
            record=json.loads(old.read_text());ids,audit=record['selected_indices'],record['audit']
            audit={**audit,'reused_from':str(old.relative_to(HERE)),'pair_cap':20}
        elif b==5:
            ids=list(warm);audit={**audits[-1],'b':5,'termination':'identical_feasible_set_to_b4','copied_from_b':4}
        else:
            log=folder/'solver_logs'/f'{date}_b{b}.log';log.parent.mkdir(parents=True,exist_ok=True)
            ids,audit=optimize(rows,paths,cash,mask,b,baseline,warm,pair_cap=cap,time_limit=limit,log_path=log)
        audit={**audit,'window':date,'months':months,'uncapped':uncapped,
            'baseline_count':len(baseline),'baseline_ids':list(rows.pair_id.iloc[baseline]),
            'selected_ids':list(rows.pair_id.iloc[ids])}
        matrix,caps=structure(rows,b,cap)
        assert (matrix[:,ids].sum(axis=1)<=caps).all()
        v=risk(paths,ids,cash,mask,cap);budget=risk(paths,baseline,cash,mask,cap)
        np.testing.assert_allclose(v,audit['selected_variance'],rtol=1e-11,atol=1e-18)
        np.testing.assert_allclose(budget,audit['baseline_variance'],rtol=1e-11,atol=1e-18)
        assert v<=budget*(1+1e-10)+1e-18
        assert -rows.tstat.iloc[ids].sum()>=-rows.tstat.iloc[warm].sum()-1e-7
        write(target,dict(selected_indices=ids,audit=audit));warm=ids;audits.append(audit)
    return dict(stage='selection',months=months,uncapped=uncapped,window=date,
        global_cases=sum(a['certified_global'] for a in audits),worst_gap=max(a['relative_score_gap'] for a in audits),
        counts=[len(baseline)]+[a['selected_count'] for a in audits])


def run_case(months,uncapped,model,bps,use_units=True):
    folder=case_path(months,uncapped)/f'{model}_{bps}bp'
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    folder.mkdir(parents=True,exist_ok=True)
    cfg=CFG.with_updates({'schedule.trading_months':months,'costs.commission_bps':float(bps)})
    capital=cfg['portfolio']['initial_capital'];daily_frames=[];pair_frames=[];trade_frames=[];selected=[];window_rows=[]
    columns=['nav_start','nav','gross_pnl','trading_cost','borrow_cost','cash_interest','turnover','gross_before','gross_notional']
    for window in windows_for(DATA.prices.index,cfg,months):
        key='baseline' if model=='BASELINE_B1' else f'b{model[-1]}'
        record=json.loads((case_path(months,uncapped)/'selection'/f'{window.id}_{key}.json').read_text())
        rows=load_rows(window.id);ids=record['selected_indices']
        pairs=pair_objects(rows.iloc[ids],DATA.prices.loc[window.formation_dates],window.formation_dates[-1])
        count=len(pairs);start=capital
        slots=count if uncapped else 20
        slot=start/slots if slots else 0.
        unused=0. if uncapped and count else start-count*slot
        frames=[]
        if use_units:
            cache=OUT/'execution_units'/f'{months}m'/f'{window.id}.pkl'
            assert sha(cache)==cache.with_suffix('.sha256').read_text().strip()
            with cache.open('rb') as f:units=pickle.load(f)
        for pair in pairs:
            if use_units:
                unit_daily,unit_trades=units[bps][pair.id]
                d=unit_daily.copy();t=unit_trades.copy()
                # No share rounding or fixed cash fees: the unchanged engine is
                # homogeneous in initial capital. Reuse units, scale money only.
                money=columns+['quantity_x','quantity_y','notional_x','notional_y']
                d[money]*=slot
                trade_money=['quantity_x','quantity_y','entry_nav','entry_gross','entry_cost','gross_pnl','borrow_cost','exit_cost','exit_nav','net_pnl']
                if len(t):t[trade_money]*=slot
                if not frames:
                    direct,_=simulate_pair(pair,DATA.prices.loc[window.trading_dates,[pair.x,pair.y]],slot,cfg,window.id)
                    np.testing.assert_allclose(d[columns],direct[columns],rtol=2e-12,atol=1e-8)
            else:
                d,t=simulate_pair(pair,DATA.prices.loc[window.trading_dates,[pair.x,pair.y]],slot,cfg,window.id)
            frames.append(d);pair_frames.append(d)
            if len(t):trade_frames.append(t)
            selected.append(dict(window=window.id,**asdict(pair)))
        agg=pd.DataFrame(0.,index=window.trading_dates,columns=columns)
        if frames:agg+=pd.concat(frames).groupby('date')[columns].sum().reindex(window.trading_dates)
        # cash_annual_rate is unchanged and zero in this experiment.
        assert cfg['portfolio']['cash_annual_rate']==0.
        agg[['nav_start','nav']]+=unused
        agg['unused_slot_cash']=unused;agg['window']=window.id
        agg['net_return']=agg.nav/agg.nav_start-1;agg.index.name='date'
        daily_frames.append(agg.reset_index());capital=float(agg.nav.iloc[-1])
        window_rows.append(dict(window=window.id,selected_pairs=count,capital_start=start,capital_end=capital,
            slot_initial_capital=slot,allocated_initial_capital=slot*count,unused_initial_cash=unused,
            trading_start=str(window.trading_dates[0].date()),trading_end=str(window.trading_dates[-1].date()),
            formation_end=str(window.formation_dates[-1].date()),partial_window=window.nominal_end < window.nominal_start+pd.DateOffset(months=months)-pd.Timedelta(days=1)))
    daily=pd.concat(daily_frames,ignore_index=True)
    pair_daily=pd.concat(pair_frames,ignore_index=True)
    trades=pd.concat(trade_frames,ignore_index=True) if trade_frames else pd.DataFrame(columns=['net_pnl','exit_reason','holding_days'])
    summary=metrics(daily,trades,cfg)
    daily.to_csv(folder/'portfolio_daily.csv',index=False)
    keep=['date','window','pair_id','x','y','nav_start','nav','gross_pnl','trading_cost','turnover','quantity_x','quantity_y']
    pair_daily[keep].to_csv(folder/'pair_daily.csv',index=False)
    trades.to_csv(folder/'trades.csv',index=False)
    pd.DataFrame(selected).to_csv(folder/'selected_pairs.csv',index=False)
    pd.DataFrame(window_rows).to_csv(folder/'windows.csv',index=False)
    write(folder/'summary.json',summary)
    write(folder/'manifest.json',dict(status='complete',months=months,uncapped=uncapped,model=model,
        bps=bps,engine='unmodified upstream simulate_pair at unit capital; homogeneous dollar scaling, checked at actual slot capital every window' if use_units else 'unmodified upstream simulate_pair at actual slot capital',allocation='NAV/K' if uncapped else 'NAV/20'))
    return summary


def backtest_work(task):
    months,uncapped,model=task
    results=[dict(months=months,uncapped=uncapped,model=model,one_way_bps=bps,**run_case(months,uncapped,model,bps)) for bps in [0,5]]
    print(json.dumps(dict(stage='backtest',months=months,uncapped=uncapped,model=model,
        gross=results[0]['cumulative_return'],net=results[1]['cumulative_return'])),flush=True)
    return results


def execution_unit_work(task):
    months,date=task[:2]
    prefetched=task[2] if len(task)>2 else None
    window=next(w for w in windows_for(DATA.prices.index,CFG,months) if w.id==date)
    paths=[];chosen=set()
    if prefetched is not None:chosen=set(prefetched)
    else:
        for uncapped in ([False,True] if months==6 else [False]):
            folder=case_path(months,uncapped)/'selection'
            for key in ['baseline']+[f'b{b}' for b in range(1,6)]:
                p=folder/f'{date}_{key}.json'
                if not p.exists():continue
                record=json.loads(p.read_text());chosen.update(record['selected_indices']);paths.append(p)
    hashes={str(p.relative_to(OUT)):sha(p) for p in paths}
    target=OUT/'execution_units'/f'{months}m'/f'{date}.pkl'
    audit=target.with_suffix('.json')
    fingerprint=dict(formation_sha256=sha(formation_path(date)),engine_sha256=sha(HERE/'vendor/sp500-pairs-trading/src/sp500_pairs/engine.py'),
        strategy=CFG['strategy'],prices_sha256=DATA.metadata['inputs']['prices']['sha256'])
    units={0:{},5:{}}
    if audit.exists():
        assert sha(target)==target.with_suffix('.sha256').read_text().strip()
        assert json.loads(audit.read_text())['fingerprint']==fingerprint
        with target.open('rb') as f:units=pickle.load(f)
    rows=load_rows(date);pairs=pair_objects(rows.iloc[sorted(chosen)],DATA.prices.loc[window.formation_dates],window.formation_dates[-1])
    missing=[p for p in pairs if any(p.id not in units[bps] for bps in [0,5])]
    if not missing:return dict(stage='execution_units',months=months,window=date,status='cached')
    failures=[]
    for bps in [0,5]:
        cfg=CFG.with_updates({'costs.commission_bps':float(bps),'schedule.trading_months':months})
        for pair in missing:
            if pair.id in units[bps]:continue
            try:units[bps][pair.id]=simulate_pair(pair,DATA.prices.loc[window.trading_dates,[pair.x,pair.y]],1.,cfg,date)
            except Exception as exc:
                if prefetched is None:raise
                failures.append(dict(pair_id=pair.id,bps=bps,error=f'{type(exc).__name__}: {exc}'))
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as f:pickle.dump(units,f)
    target.with_suffix('.sha256').write_text(sha(target)+'\n')
    write(audit,dict(selection_hashes=hashes,pairs=len(units[0]),engine='unmodified upstream simulate_pair',fingerprint=fingerprint,prefetch_failures=failures))
    return dict(stage='execution_units',months=months,window=date,pairs=len(pairs))


def pooled(fn,tasks,args):
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(args.source),str(args.output))) as pool:
        futures=[pool.submit(fn,t) for t in tasks]
        result=[]
        for future in as_completed(futures):
            value=future.result();result.append(value)
            if fn!=backtest_work:print(json.dumps(value,ensure_ascii=False),flush=True)
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=HERE/'output/actual')
    p.add_argument('--output',type=Path,default=DEFAULT_OUT);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--solver-seconds',type=float,default=15.);p.add_argument('--phase',choices=['all','formation','selection','backtest'],default='all')
    p.add_argument('--arm',choices=['both','uncapped','cadence'],default='both')
    p.add_argument('--ready-only',action='store_true',help='Process only verified formation caches currently available')
    args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    initialize(str(args.source),str(args.output));verify_upstream();OUT.mkdir(parents=True,exist_ok=True)
    source_receipt=json.loads((SOURCE/'experiment.json').read_text())
    protocol=dict(experiments=['Remove only global pair cap at 6m cadence; NAV/K for all models','Cadence 1,2,3,4,5,6 months; max20 and NAV/20 retained'],
        models=['Original nonoverlap greedy benchmark']+[f'Score maximum with degree <= {b} and common nonoverlap formation risk ceiling' for b in range(1,6)],
        common=dict(period='2016-01-04 to 2024-12-31; identical calendar, initial NAV 1000000',
            formation_months=12,sector_pairs=4,full_candidate_pool=True,original_filters_unchanged=True,
            fees='0 and 5bp ONE WAY per executed leg; no netting, slippage, borrow or cash yield',
            trading='Unmodified upstream engine, fixed shares, next-close fills, all positions closed at each window end',
            risk='Original-engine gross NAV sample variance on prior12m replay blocks matching cadence; exclude reset transitions; full-formation coefficients, in-sample only',
            risk_reference='Each experiment/window has its OWN greedy NONOVERLAP benchmark under the SAME capital and replay rules; shared across b1..5',
            final_partial_window='5m last window Oct-Dec2024 included to preserve common evaluation end',
            formation_acceleration='Reject only pairs already failing unchanged deterministic screens before EG, valid because multiple_testing=none and require_i1=false. Original fitter on every surviving pair; parity against all18 old eligible pools.',
            limits='Exploratory, static sector map, provided NBL merger/security identity issue remains; no price edits'),
        b4_b5_equivalent=True,all_inputs=source_receipt['fingerprint']['data']['inputs'],upstream_commit=source_receipt['fingerprint']['upstream_commit'])
    write(OUT/'protocol.json',protocol)
    write(OUT/'run.json',dict(status='running',phase=args.phase,source=str(SOURCE),source_fingerprint=source_receipt['fingerprint'],
        code_sha256={n:sha(HERE/n) for n in ['extension_experiment.py','extension_solver.py','extension_formation.py']},solver_seconds=args.solver_seconds))
    if args.phase in ['all','formation']:
        dates=[w.id for w in windows_for(DATA.prices.index,CFG,1)]
        pooled(formation_work,dates,args)
    cases=([(h,False) for h in range(1,7)] if args.arm!='uncapped' else [])+([(6,True)] if args.arm!='cadence' else [])
    if args.phase in ['all','selection']:
        tasks=[(h,w.id,uncapped,args.solver_seconds) for h,uncapped in cases for w in windows_for(DATA.prices.index,CFG,h)]
        if args.ready_only:
            tasks=[t for t in tasks if formation_path(t[1]).with_suffix('.sha256').exists() or (SOURCE/'cache'/t[1]/'formation.sha256').exists()]
        pooled(selection_work,tasks,args)
    if args.phase in ['all','backtest']:
        unit_tasks=sorted({(h,w.id) for h,_ in cases for w in windows_for(DATA.prices.index,CFG,h)})
        pooled(execution_unit_work,unit_tasks,args)
        models=['BASELINE_B1']+[f'SCORE_RISK_B{b}' for b in range(1,6)]
        tasks=[(h,uncapped,model) for h,uncapped in cases for model in models]
        results=pooled(backtest_work,tasks,args)
        pd.DataFrame([r for case in results for r in case]).sort_values(['uncapped','months','model','one_way_bps']).to_csv(OUT/f'comparison_{args.arm}.csv',index=False)
    receipt=json.loads((OUT/'run.json').read_text());receipt.update(status='complete' if args.phase=='all' else 'phase_complete')
    write(OUT/'run.json',receipt)


if __name__=='__main__':main()
