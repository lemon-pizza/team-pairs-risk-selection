"""Precompute independent formation caches with unchanged upstream functions.

Publish a completed directory atomically. Never replace a cache directory that
the primary experiment already owns. This is only a compute-time optimization.
"""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import time

from experiment import HERE,UPSTREAM,verify_upstream,sha,write
from sp500_pairs.config import load_config
from sp500_pairs.data import load_data
from sp500_pairs.formation import select_pairs
from sp500_pairs.schedule import make_windows

DATA=CFG=OUTPUT=None


def initialize(output):
    global DATA,CFG,OUTPUT
    OUTPUT=Path(output)
    receipt=json.loads((OUTPUT/'experiment.json').read_text())
    values=receipt['fingerprint']['config']
    default=load_config(UPSTREAM/'configs/baseline.toml')
    updates={f'{section}.{key}':value for section,settings in values.items() for key,value in settings.items()
             if default.values[section][key]!=value}
    CFG=default.with_updates(updates)
    assert CFG.values==values
    DATA=load_data(CFG)
    assert DATA.metadata==receipt['fingerprint']['data']
    verify_upstream()


def work(window_id):
    destination=OUTPUT/'cache'/window_id
    if destination.exists(): return {'window':window_id,'status':'primary_owns_cache'}
    window=next(w for w in make_windows(DATA.prices.index,CFG) if w.id==window_id)
    started=time.monotonic()
    result=select_pairs(DATA,window,CFG)
    temporary=Path(tempfile.mkdtemp(prefix='formation_',dir=OUTPUT))
    try:
        path=temporary/'formation.pkl'
        with path.open('wb') as handle: pickle.dump(result,handle)
        (temporary/'formation.sha256').write_text(sha(path)+'\n')
        record={'window':window_id,'elapsed_seconds':time.monotonic()-started,
                'tested_pairs':len(result[1]),'selected_pairs':len(result[0]),
                'upstream_commit':receipt_commit(),'source':'unchanged upstream select_pairs',
                'formation_end':str(window.formation_dates[-1].date()),
                'first_trading_date':str(window.trading_dates[0].date()),
                'cache_sha256':sha(path),'prefetch_code_sha256':sha(__file__)}
        write(temporary/'prefetch_audit.json',record)
        if destination.exists():
            record['status']='primary_owns_cache'
        else:
            try:
                os.rename(temporary,destination)
                record['status']='published'
            except OSError:
                if not destination.exists(): raise
                record['status']='primary_owns_cache'
        return record
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def receipt_commit():
    return json.loads((HERE/'upstream.json').read_text())['commit']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    initialize(str(args.output.resolve()))
    windows=make_windows(DATA.prices.index,CFG)
    pending=[w.id for w in reversed(windows) if not (OUTPUT/'cache'/w.id).exists()]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(OUTPUT),)) as pool:
        for result in pool.map(work,pending): print(json.dumps(result),flush=True)


if __name__=='__main__': main()
