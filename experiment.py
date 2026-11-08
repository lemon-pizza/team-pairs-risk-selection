"""Run unchanged team baseline and the b=2 score/risk selection experiment."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys
import platform
import os
from dataclasses import asdict

HERE=Path(__file__).resolve().parent
UPSTREAM=HERE/'vendor/sp500-pairs-trading'
sys.path.insert(0,str(UPSTREAM/'src'))
import numpy as np
import pandas as pd
from sp500_pairs.config import load_config
from sp500_pairs.data import load_data
from sp500_pairs.formation import Pair
import sp500_pairs.runner as runner
from selection import eligible_rows,calibration_blocks,solve


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def verify_upstream():
    info=json.loads((HERE/'upstream.json').read_text())
    for item in info['files']:
        b=(UPSTREAM/item['path']).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==item['sha'],item['path']
    return info


class Selectors:
    def __init__(self,output):
        self.original=runner.select_pairs
        self.output=output
        self.formations={}
        self.optimized={}
        self.audits=[]

    def base(self,data,window,cfg,progress=None):
        key=window.id
        if key not in self.formations:
            # Persistent formation cache is keyed by sources, data, and settings.
            folder=self.output/'cache'/key
            folder.mkdir(parents=True,exist_ok=True)
            p=folder/'formation.pkl'
            receipt=folder/'formation.sha256'
            if p.exists() and receipt.exists():
                assert sha(p)==receipt.read_text().strip()
                with p.open('rb') as f: self.formations[key]=pickle.load(f)
            else:
                self.formations[key]=self.original(data,window,cfg,progress)
                with p.open('wb') as f: pickle.dump(self.formations[key],f)
                receipt.write_text(sha(p)+'\n')
        return self.formations[key]

    def risk(self,data,window,cfg,progress=None):
        if window.id in self.optimized:
            return self.optimized[window.id]
        baseline,tested,stocks,directions,info=self.base(data,window,cfg,progress)
        rows=eligible_rows(tested).reset_index(drop=True)
        if progress: progress(f'{window.id}: ALL {len(rows)} eligible candidates; no top-N filter')
        prices=data.prices.loc[window.formation_dates]
        pairs=[]
        for row in rows.itertuples():
            px,py=float(prices[row.x].iloc[-1]),float(prices[row.y].iloc[-1])
            z=(np.log(py)-row.alpha-row.beta*np.log(px)-row.mean)/row.std
            pairs.append(Pair(row.pair_id,row.x,row.y,row.sector,row.alpha,row.beta,row.mean,row.std,window.formation_dates[-1],z,px,py))
        path=self.output/'cache'/window.id/'risk_paths.npz'
        if path.exists():
            assert path.with_suffix('.sha256').read_text().strip()==sha(path)
            with np.load(path,allow_pickle=False) as packed:
                assert list(packed['pair_ids'])==[p.id for p in pairs]
                paths=packed['paths']
                cash=packed['cash_path']; mask=packed['return_mask']
                blocks=json.loads(str(packed['blocks']))
        else:
            paths,cash,mask,blocks=calibration_blocks(pairs,prices,cfg,window)
            np.savez_compressed(path,paths=paths,pair_ids=np.array([p.id for p in pairs],dtype=str),
                                cash_path=cash,return_mask=mask,blocks=np.array(json.dumps(blocks)))
            path.with_suffix('.sha256').write_text(sha(path)+'\n')
        mapping={p.id:i for i,p in enumerate(pairs)}
        baseline_ids=[mapping[p.id] for p in baseline]
        chosen,audit=solve(rows,paths,baseline_ids,cfg['selection']['max_pairs'],cash,return_mask=mask)
        chosen_pairs=[pairs[i] for i in chosen]
        audit.update(window=window.id,formation_start=str(prices.index[0].date()),
                     formation_end=str(prices.index[-1].date()),first_trading_date=str(window.trading_dates[0].date()),
                     baseline_ids=[p.id for p in baseline],selected_ids=[p.id for p in chosen_pairs],
                     calibration_observations=int(mask.sum()),calibration_blocks=blocks)
        assert prices.index[-1]<window.trading_dates[0]
        selected=set(audit['selected_ids'])
        changed=tested.copy()
        changed['baseline_selected']=changed.selected
        changed['formation_filter_pass']=changed.exclusion.fillna('').isin({'','stock_cap','sector_cap','portfolio_full'})
        changed['selected']=changed.pair_id.isin(selected)
        changed.loc[changed.formation_filter_pass,'exclusion']='optimizer_not_selected'
        changed.loc[changed.selected,'exclusion']=''
        changed.loc[changed.selected,'selection_rank']=np.nan
        for rank,pair in enumerate(chosen_pairs,1): changed.loc[changed.pair_id.eq(pair.id),'selection_rank']=rank
        self.audits.append(audit)
        write(self.output/'selection_audit'/f'{window.id}.json',audit)
        rows.assign(baseline_selected=rows.pair_id.isin(audit['baseline_ids']),optimized_selected=rows.pair_id.isin(selected)).to_csv(self.output/'selection_audit'/f'{window.id}_all_eligible.csv',index=False)
        self.optimized[window.id]=(chosen_pairs,changed,stocks,directions,{**info,'selected_pairs':len(chosen_pairs),'empty_slots':cfg['selection']['max_pairs']-len(chosen_pairs)})
        if progress: progress(f'{window.id}: score {audit["baseline_score"]:.4f} -> {audit["selected_score"]:.4f}; risk ratio {audit["variance_ratio"]}')
        return self.optimized[window.id]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prices',type=Path)
    parser.add_argument('--membership',type=Path)
    parser.add_argument('--sectors',type=Path)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--max-windows',type=int,default=0)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    verify_upstream()
    cfg=load_config(UPSTREAM/'configs/baseline.toml')
    updates={'run.max_windows':args.max_windows}
    if args.demo:
        updates.update({'data.mode':'synthetic','schedule.end':'2016-12-31'})
    elif not all([args.prices,args.membership,args.sectors]):
        parser.error('Actual runs require --prices, --membership, --sectors; synthetic checks require --demo')
    else:
        for key in ('prices','membership','sectors'):
            p=getattr(args,key).resolve()
            if not p.is_file(): parser.error(f'Missing input: {p}')
            updates['data.'+key+'_file']=str(p)
    cfg=cfg.with_updates(updates)
    data=load_data(cfg)
    output=(args.output or HERE/'output'/('synthetic' if args.demo else 'actual')).resolve()
    output.mkdir(parents=True,exist_ok=True)
    fingerprint={'config':cfg.values,'data':data.metadata,'protocol_sha256':sha(HERE/'protocol.json'),
                 'selection_code_sha256':sha(HERE/'selection.py'),'experiment_code_sha256':sha(__file__),
                 'upstream_commit':json.loads((HERE/'upstream.json').read_text())['commit']}
    manifest_path=output/'experiment.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous['fingerprint']!=fingerprint: raise ValueError('Output settings/inputs changed; use a different --output directory')
    manifest={'status':'running','fingerprint':fingerprint,'python':platform.python_version(),
              'mode':'synthetic_software_check' if args.demo else 'actual_exploratory',
              'numpy':np.__version__,'pandas':pd.__version__}
    write(manifest_path,manifest)
    selectors=Selectors(output)
    summaries=[]
    try:
        for model,selector in [('BASELINE_B2',selectors.base),('SCORE_RISK_B2',selectors.risk)]:
            runner.select_pairs=selector
            for bps in (5.,0.):
                name=f'{model}_{bps:g}bp'
                print(f'RUN {name}',flush=True)
                current=cfg.with_updates({'costs.commission_bps':bps})
                case=output/name
                if case.exists():
                    state=json.loads((case/'manifest.json').read_text())
                    if state['status']!='complete' or json.loads((case/'effective_config.json').read_text())!=current.values:
                        raise ValueError(f'Incomplete/different existing case: {case}; use a new output directory')
                    summary=json.loads((case/'summary.json').read_text())
                else:
                    result=runner.run_backtest(current,data=data,output_dir=case,progress=lambda s:print(s,flush=True))
                    summary=result['summary']
                summaries.append({'model':model,'one_way_bps':bps,**summary})
        pd.DataFrame(summaries).to_csv(output/'comparison.csv',index=False)
        audits=[json.loads(p.read_text()) for p in sorted((output/'selection_audit').glob('*.json'))]
        write(output/'selection_summary.json',audits)
        manifest.update(status='complete',cases=4,windows=len(audits),source_files_verified=28)
        verify_upstream()
        write(manifest_path,manifest)
        print('COMPLETED all four evaluations; see comparison.csv',flush=True)
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write(manifest_path,manifest)
        raise
    finally:
        runner.select_pairs=selectors.original


if __name__=='__main__': main()
