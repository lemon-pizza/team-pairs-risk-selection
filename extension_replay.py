"""Vectorized zero-cost formation replay, checked against the original engine.

Only calibration is accelerated. Actual backtests always use simulate_pair.
Supported settings deliberately match this experiment; others fail explicitly.
"""
import json
import numpy as np
import pandas as pd
from selection import calibration_blocks


def block_paths(pairs, block, cfg):
    settings=cfg['strategy']
    assert cfg['portfolio']['cash_annual_rate']==0
    assert settings['reentry_after_stop']=='block_window' and settings['reentry_after_time_exit']=='block_window'
    n=len(pairs);days=len(block)
    if not n:return np.empty((days,0))
    px=block[[p.x for p in pairs]].to_numpy(float)
    py=block[[p.y for p in pairs]].to_numpy(float)
    alpha=np.array([p.alpha for p in pairs]);beta=np.array([p.beta for p in pairs])
    mean=np.array([p.mean for p in pairs]);std=np.array([p.std for p in pairs])
    finite=np.isfinite(px)&np.isfinite(py)&(px>0)&(py>0)
    z=(np.log(py)-alpha-beta*np.log(px)-mean)/std
    z=np.where(finite,z,np.nan)
    def direction(values):
        accepted=np.isfinite(values)&(np.abs(values)>=settings['entry_z'])
        if settings['stop_enabled']:accepted&=np.abs(values)<settings['stop_z']
        return np.where(accepted,np.where(values>0,-1,1),0)
    nav=np.ones(n);paths=np.ones((days,n));qx=np.zeros(n);qy=np.zeros(n)
    position=np.zeros(n,int);entry_index=np.zeros(n,int);blocked=np.zeros(n,bool)
    pending_direction=direction(z[0]);pending=np.where(pending_direction!=0,1,0)
    pending_reason=np.zeros(n,int)  # 1=mean, 2=stop, 3=time, 4=window end
    for i in range(1,days):
        held=position!=0
        if (held&~finite[i]).any():raise ValueError('Missing held calibration quote')
        nav[held]+=qx[held]*(px[i,held]-px[i-1,held])+qy[held]*(py[i,held]-py[i-1,held])
        if not np.isfinite(nav).all() or (nav<=0).any():raise ValueError('Invalid calibration NAV')
        execution=pending.copy();why=pending_reason.copy();event=np.zeros(n,bool)
        final=i==days-1
        if final:
            forced=held&(execution!=2);execution[forced]=2;why[forced]=4
            execution[~held]=0
        enter=(execution==1)&~held&~blocked&finite[i]
        event[(execution==1)&~finite[i]]=True
        qx[enter]=-pending_direction[enter]*nav[enter]*beta[enter]/(1+beta[enter])/px[i,enter]
        qy[enter]=pending_direction[enter]*nav[enter]/(1+beta[enter])/py[i,enter]
        position[enter]=pending_direction[enter];entry_index[enter]=i-1;event[enter]=True
        exit=(execution==2)&held
        position[exit]=0;qx[exit]=0.;qy[exit]=0.;event[exit]=True
        blocked[exit&((why==2)|(why==3))]=True
        pending=np.zeros(n,int);pending_reason=np.zeros(n,int);pending_direction=np.zeros(n,int)
        if not final:
            held=position!=0
            holding=(i-1)-entry_index+1
            mean_exit=held&(((position==1)&(z[i]>=-settings['exit_z']))|((position==-1)&(z[i]<=settings['exit_z'])))
            stop_exit=held&~mean_exit&settings['stop_enabled']&(((position==1)&(z[i]<=-settings['stop_z']))|((position==-1)&(z[i]>=settings['stop_z'])))
            time_exit=held&~mean_exit&~stop_exit&settings['max_holding_enabled']&(holding>=settings['max_holding_days']-1)
            pending[mean_exit|stop_exit|time_exit]=2
            pending_reason[mean_exit]=1;pending_reason[stop_exit]=2;pending_reason[time_exit]=3
            if i+1<days-1:
                flat=~held&~event&finite[i]&~blocked
                new=direction(z[i]);pending_direction=np.where(flat,new,0)
                pending[(pending_direction!=0)]=1
        paths[i]=nav
    return paths


def fast_calibration_blocks(pairs,prices,cfg,window):
    matrices=[];masks=[];records=[]
    begin=window.nominal_start-pd.DateOffset(months=cfg['schedule']['formation_months'])
    while begin<window.nominal_start:
        end=min(begin+pd.DateOffset(months=cfg['schedule']['trading_months']),window.nominal_start)
        block=prices.loc[(prices.index>=begin)&(prices.index<end)]
        if len(block)<3:raise ValueError('Insufficient calibration observations')
        matrix=block_paths(pairs,block,cfg)
        if matrices:masks.append(np.array([False]))
        masks.append(np.ones(len(matrix)-1,bool));matrices.append(matrix)
        records.append(dict(start=str(block.index[0].date()),end=str(block.index[-1].date()),returns=len(block)-1))
        begin=end
    paths=np.vstack(matrices);cash=np.ones(len(paths));mask=np.concatenate(masks)
    # Deterministic spread through every candidate pool, plus missing-anchor pairs.
    sample=set(np.linspace(0,max(0,len(pairs)-1),min(5,len(pairs)),dtype=int).tolist())
    if pairs:
        first=prices.index[0]
        sample.update(j for j,p in enumerate(pairs) if not np.isfinite(prices.loc[first,[p.x,p.y]]).all())
    sample=sorted(sample)
    check,check_cash,check_mask,check_blocks=calibration_blocks([pairs[j] for j in sample],prices,cfg,window)
    np.testing.assert_allclose(paths[:,sample],check,rtol=0,atol=2e-13)
    np.testing.assert_array_equal(mask,check_mask);np.testing.assert_array_equal(cash,check_cash)
    assert records==check_blocks
    return paths,cash,mask,records,dict(original_engine_parity_pairs=len(sample),max_abs_NAV_error=float(np.max(np.abs(paths[:,sample]-check))) if sample else 0.)
