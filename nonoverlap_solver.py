"""Score maximization with exact original-engine NAV-return variance.

Integer separation excludes only selections rejected by the exact NAV oracle.
The optional fractional MINLP formulation is retained for independent checks.
"""
import time
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from pyscipopt import Model, quicksum, Conshdlr, SCIP_RESULT
from selection import incidence, nav_risk


def structure(rows,b):
    matrix,caps=incidence(rows)
    caps[:len(set(rows.x)|set(rows.y))]=b
    return np.vstack([matrix,np.ones(len(rows),dtype=int)]),np.r_[caps,20]


def greedy(rows,b=1):
    matrix,caps=structure(rows,b)
    counts=np.zeros(len(caps),int);ids=[]
    for j in range(len(rows)):
        if (counts+matrix[:,j]<=caps).all():
            ids.append(j);counts+=matrix[:,j]
    return ids


class ExactRiskHandler(Conshdlr):
    """Finite exact integer separation: forbid only risk-infeasible selections."""
    def __init__(self,z,paths,cash,mask,limit,matrix,caps,score,score_floor):
        self.z=z;self.paths=paths;self.cash=cash;self.mask=mask;self.limit=limit
        self.matrix=matrix;self.caps=caps;self.score=score;self.floor=score_floor
        self.cache={};self.cuts=0
        self.cover_cuts=0

    def risk_cover(self,ids):
        # A 19-edge cover is valid only after checking EVERY feasible completion
        # (zero or one more edge) at the model's explicit score lower bound.
        # Thus no monotonicity of variance or heuristic risk approximation is used.
        if len(ids)!=20:return None
        ids=list(ids)
        total=self.paths[:,ids].sum(axis=1)
        cores=total[:,None]-self.paths[:,ids]+self.cash[:,None]
        variances=np.var((cores[1:]/cores[:-1]-1)[self.mask],axis=0,ddof=1)
        for k in np.argsort(-variances)[:4]:
            core=[j for j in ids if j!=ids[k]]
            core_score=float(self.score[core].sum())
            if core_score>=self.floor-1e-8 and variances[k]<=self.limit:continue
            outside=np.setdiff1d(np.arange(len(self.z)),core)
            candidates=outside[self.score[outside]+core_score>=self.floor-1e-8]
            counts=self.matrix[:,core].sum(axis=1)
            candidates=candidates[(counts[:,None]+self.matrix[:,candidates]<=self.caps[:,None]).all(axis=0)]
            feasible=False
            for start in range(0,len(candidates),128):
                new=candidates[start:start+128]
                nav=cores[:,k,None]+self.paths[:,new]-self.cash[:,None]
                risks=np.var((nav[1:]/nav[:-1]-1)[self.mask],axis=0,ddof=1)
                if (risks<=self.limit).any():feasible=True;break
            if not feasible:return core
        return None

    def check(self,solution=None):
        values=np.array([self.model.getSolVal(solution,v) for v in self.z])
        if np.max(np.abs(values-np.rint(values)))>1e-6:return None
        ids=tuple(np.flatnonzero(values>.5))
        if ids not in self.cache:
            self.cache[ids]=nav_risk(self.paths,ids,20,self.cash,self.mask)<=self.limit
        return ids,self.cache[ids]

    def conscheck(self,constraints,solution,check_integrality,check_lp_rows,print_reason,completely,**results):
        answer=self.check(solution)
        return {'result':SCIP_RESULT.FEASIBLE if answer is None or answer[1] else SCIP_RESULT.INFEASIBLE}

    def consenfolp(self,constraints,n_useful_conss,sol_infeasible):
        answer=self.check()
        if answer is None or answer[1]:return {'result':SCIP_RESULT.FEASIBLE}
        ids=set(answer[0])
        cover=self.risk_cover(ids)
        if cover is not None:
            self.model.addCons(quicksum(self.z[j] for j in cover)<=len(cover)-1)
            self.cover_cuts+=1
        elif len(ids)==20:
            # With sum(z)<=20 this excludes exactly this 20-element set.
            self.model.addCons(quicksum(self.z[j] for j in ids)<=19)
        else:
            self.model.addCons(quicksum(1-self.z[j] if j in ids else self.z[j] for j in range(len(self.z)))>=1)
        self.cuts+=1
        return {'result':SCIP_RESULT.CONSADDED}

    def conslock(self,constraint,locktype,nlockspos,nlocksneg):
        for var in self.z:
            self.model.addVarLocksType(var,locktype,nlockspos+nlocksneg,nlockspos+nlocksneg)


def lazy_solve(rows,paths,cash,mask,b,baseline,warm=None,time_limit=30.,log_path=None):
    started=time.monotonic();matrix,caps=structure(rows,b);score=-rows.tstat.to_numpy(float)
    budget=nav_risk(paths,baseline,20,cash,mask);limit=budget*(1+1e-10)+1e-18
    model=Model(f'exact_risk_b{b}');model.hideOutput()
    if log_path:model.setLogfile(str(log_path))
    model.setRealParam('limits/time',time_limit);model.setRealParam('limits/gap',0.)
    model.setRealParam('numerics/feastol',1e-9)
    model.setBoolParam('misc/allowstrongdualreds',False)
    model.setBoolParam('misc/allowweakdualreds',False)
    model.setIntParam('randomization/randomseedshift',20261106)
    z=[model.addVar(vtype='B',name=f'z_{j}') for j in range(len(rows))]
    for k,row in enumerate(matrix):
        model.addCons(quicksum(z[j] for j in np.flatnonzero(row))<=int(caps[k]))
    model.setObjective(quicksum(float(score[j])*z[j] for j in range(len(rows))),'maximize')
    seeds=[list(baseline)]+([list(warm)] if warm is not None else [])
    best=local_improve(rows,paths,cash,mask,b,budget,seeds)
    floor=float(score[best].sum())-1e-8
    model.addCons(quicksum(float(score[j])*z[j] for j in range(len(rows)))>=floor)
    handler=ExactRiskHandler(z,paths,cash,mask,limit,matrix,caps,score,floor)
    model.includeConshdlr(handler,'exact_nav_variance','Exact formation replay variance',chckpriority=-10,enfopriority=-10)
    model.addPyCons(model.createCons(handler,'risk'))
    for ids in [baseline,best]:
        sol=model.createSol();ids=set(ids)
        for j,var in enumerate(z):model.setSolVal(sol,var,float(j in ids))
        model.addSol(sol,free=True)
    model.optimize()
    for sol in model.getSols():
        ids=[j for j,var in enumerate(z) if model.getSolVal(sol,var)>.5]
        if nav_risk(paths,ids,20,cash,mask)<=limit and (matrix[:,ids].sum(axis=1)<=caps).all():
            if score[ids].sum()>score[best].sum()+1e-10:best=ids
    value=float(score[best].sum());upper=float(model.getDualbound())
    return best,dict(b=b,selected_score=value,baseline_score=float(score[baseline].sum()),
        baseline_variance=budget,selected_variance=nav_risk(paths,best,20,cash,mask),
        variance_ratio=nav_risk(paths,best,20,cash,mask)/budget if budget else None,
        score_upper_bound=upper,absolute_score_gap=max(0.,upper-value),
        relative_score_gap=max(0.,upper-value)/max(abs(value),1e-12),
        certified_global=bool(upper-value<1e-6),termination=str(model.getStatus()),
        risk_no_good_cuts=handler.cuts,checked_integer_solutions=len(handler.cache),
        exact_risk_cover_cuts=handler.cover_cuts,
        candidate_count=len(rows),selected_count=len(best),elapsed_seconds=time.monotonic()-started,
        candidate_top_n_cap=None,cardinality_fixed=False)


def optimize(rows,paths,cash,mask,b,baseline,warm=None,time_limit=60.,log_path=None):
    matrix,caps=structure(rows,b);score=-rows.tstat.to_numpy(float)
    budget=nav_risk(paths,baseline,20,cash,mask)
    result=milp(-score,integrality=np.ones(len(rows)),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix,-np.inf,caps),options={'mip_rel_gap':0.,'time_limit':30.})
    if result.status!=0 or result.x is None:raise RuntimeError('Structural score optimum was not certified')
    proposal=np.flatnonzero(result.x>.5).astype(int).tolist()
    upper=float(-result.mip_dual_bound)
    pv=nav_risk(paths,proposal,20,cash,mask)
    seeds=[list(baseline)]+([list(warm)] if warm is not None else [])
    if pv<=budget*(1+1e-10)+1e-18:seeds.append(proposal)
    best=max(seeds,key=lambda ids:score[ids].sum())
    if abs(upper-score[best].sum())<1e-7:
        value=float(score[best].sum());variance=nav_risk(paths,best,20,cash,mask)
        return best,dict(b=b,selected_score=value,baseline_score=float(score[baseline].sum()),
            baseline_variance=budget,selected_variance=variance,variance_ratio=variance/budget if budget else None,
            score_upper_bound=upper,absolute_score_gap=max(0.,upper-value),relative_score_gap=0.,
            certified_global=True,termination='linear_score_optimum_feasible',
            selected_count=len(best),candidate_count=len(rows),candidate_top_n_cap=None,
            cardinality_fixed=False,linear_optimum_variance_ratio=pv/budget if budget else None,
            risk_no_good_cuts=0,checked_integer_solutions=1)
    best,audit=lazy_solve(rows,paths,cash,mask,b,baseline,warm,time_limit,log_path)
    audit['linear_optimum_variance_ratio']=pv/budget if budget else None
    audit['score_upper_bound']=min(audit['score_upper_bound'],upper)
    audit['absolute_score_gap']=max(0.,audit['score_upper_bound']-audit['selected_score'])
    audit['relative_score_gap']=audit['absolute_score_gap']/audit['selected_score']
    audit['certified_global']=audit['absolute_score_gap']<1e-6
    return best,audit


def local_improve(rows,paths,cash,mask,b,budget,seeds,max_moves=100):
    matrix,caps=structure(rows,b)
    score=-rows.tstat.to_numpy()
    best=tuple(seeds[0]);value=float(score[list(best)].sum())
    limit=budget*(1+1e-10)+1e-18
    for seed in seeds:
        if (matrix[:,seed].sum(axis=1)<=caps).all() and nav_risk(paths,seed,20,cash,mask)<=limit:
            if score[seed].sum()>value:best=tuple(seed);value=float(score[seed].sum())
    lookup={tuple(sorted((r.x,r.y))):j for j,r in enumerate(rows.itertuples())}
    for _ in range(max_moves):
        ids=np.array(best,int);outside=np.setdiff1d(np.arange(len(rows)),ids)
        counts=matrix[:,ids].sum(axis=1)
        total=paths[:,ids].sum(axis=1)+(20-len(ids))*cash
        change=None;gain=1e-10
        # Single replacements, additions and removals retain free cardinality.
        for old in list(ids)+[None]:
            candidates=outside if old is None else outside[score[outside]>score[old]+gain]
            minus=np.zeros(len(caps)) if old is None else matrix[:,old]
            good=(counts[:,None]-minus[:,None]+matrix[:,candidates]<=caps[:,None]).all(axis=0)
            candidates=candidates[good]
            for start in range(0,len(candidates),128):
                new=candidates[start:start+128]
                if not len(new):continue
                nav=total[:,None]+paths[:,new]-cash[:,None]
                if old is not None:nav-=paths[:,old,None]-cash[:,None]
                v=np.var((nav[1:]/nav[:-1]-1)[mask],axis=0,ddof=1)
                gains=score[new]-(0 if old is None else score[old])
                accepted=(v<=limit)&(gains>gain)
                if accepted.any():
                    j=int(np.argmax(np.where(accepted,gains,-np.inf)))
                    gain=float(gains[j]);change=({int(old)} if old is not None else set(),{int(new[j])})
        # Degree-preserving reconnections improve nonoverlap matchings too.
        selected=set(best)
        for left,right in combinations(ids,2):
            a,c=rows.iloc[left][['x','y']];d,e=rows.iloc[right][['x','y']]
            if len({a,c,d,e})<4:continue
            for edges in [((a,d),(c,e)),((a,e),(c,d))]:
                proposed=[lookup.get(tuple(sorted(edge))) for edge in edges]
                if any(j is None or j in selected for j in proposed):continue
                improved=float(score[proposed].sum()-score[[left,right]].sum())
                if improved<=gain:continue
                candidate=list((selected-{int(left),int(right)})|set(proposed))
                if (matrix[:,candidate].sum(axis=1)<=caps).all() and nav_risk(paths,candidate,20,cash,mask)<=limit:
                    gain=improved;change=({int(left),int(right)},set(proposed))
        if change is None:break
        best=tuple(sorted((set(best)-change[0])|change[1]));value=float(score[list(best)].sum())
    return list(best)


def solve(rows,paths,cash,mask,b,baseline,warm=None,time_limit=30.,log_path=None):
    started=time.monotonic()
    matrix,caps=structure(rows,b);score=-rows.tstat.to_numpy(float)
    budget=nav_risk(paths,baseline,20,cash,mask)
    allowed=budget*(1+1e-10)+1e-18
    linear=milp(-score,integrality=np.ones(len(rows)),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix,-np.inf,caps),options={'mip_rel_gap':0.,'time_limit':30.})
    if linear.x is None:raise RuntimeError('No structural integer solution')
    upper=float(-linear.mip_dual_bound)
    proposal=np.flatnonzero(linear.x>.5).astype(int).tolist()
    value=float(score[proposal].sum())
    variance=nav_risk(paths,proposal,20,cash,mask)
    base_score=float(score[baseline].sum())
    seeds=[list(baseline)]+([list(warm)] if warm is not None else [])
    if variance<=allowed:
        seeds.append(proposal)
    best=max(seeds,key=lambda ids:score[ids].sum())
    status='linear_score_optimum_feasible'
    scip_bound=None;scip_status=None
    if abs(upper-float(score[best].sum()))>1e-7:
        best=local_improve(rows,paths,cash,mask,b,budget,seeds)
    if abs(upper-float(score[best].sum()))>1e-7:
        model=Model(f'nonoverlap_b{b}')
        model.hideOutput()
        if log_path:model.setLogfile(str(log_path))
        model.setRealParam('limits/time',time_limit)
        model.setRealParam('numerics/feastol',1e-9)
        model.setRealParam('limits/gap',0.)
        model.setIntParam('randomization/randomseedshift',20261106)
        z=[model.addVar(vtype='B',name=f'z_{i}') for i in range(len(rows))]
        for k,row in enumerate(matrix):
            model.addCons(quicksum(float(row[j])*z[j] for j in np.flatnonzero(row))<=float(caps[k]))
        obj=quicksum(float(score[j])*z[j] for j in range(len(rows)))
        model.setObjective(obj,'maximize')
        model.addCons(obj>=float(score[best].sum())-1e-8)
        sigma=float(np.sqrt(budget))
        if sigma<=0:raise ValueError('Zero baseline variance requires separate treatment')
        transitions=np.flatnonzero(mask)
        n=len(transitions)
        xvars=[];nvars=[];bounds=[];delta_coeffs=[]
        for t in transitions:
            coeff=(paths[t]-cash[t])/20
            delta=((paths[t+1]-cash[t+1])-(paths[t]-cash[t]))/(20*sigma)
            low=float(cash[t]+np.sort(np.minimum(coeff,0))[:20].sum())
            high=float(cash[t]+np.sort(np.maximum(coeff,0))[-20:].sum())
            if low<=0:raise ValueError('Cannot prove positive NAV for entire candidate universe')
            d0=float((cash[t+1]-cash[t])/sigma)
            dlow=float(d0+np.sort(np.minimum(delta,0))[:20].sum())
            dhigh=float(d0+np.sort(np.maximum(delta,0))[-20:].sum())
            xlow=min(dlow/low,dlow/high);xhigh=max(dhigh/low,dhigh/high)
            nv=model.addVar(lb=low,ub=high,name=f'nav_{t}')
            xv=model.addVar(lb=xlow,ub=xhigh,name=f'r_{t}')
            model.addCons(nv==float(cash[t])+quicksum(float(coeff[j])*z[j] for j in range(len(rows))))
            model.addCons(nv*xv==d0+quicksum(float(delta[j])*z[j] for j in range(len(rows))))
            nvars.append(nv);xvars.append(xv);bounds.append((xlow,xhigh))
        mean=model.addVar(lb=min(v[0] for v in bounds),ub=max(v[1] for v in bounds),name='mean_return')
        model.addCons(n*mean==quicksum(xvars))
        deviations=[]
        for k,xv in enumerate(xvars):
            d=model.addVar(lb=bounds[k][0]-max(v[1] for v in bounds),
                           ub=bounds[k][1]-min(v[0] for v in bounds),name=f'dev_{k}')
            model.addCons(d==xv-mean);deviations.append(d)
        model.addCons(quicksum(d*d for d in deviations)<=float(n-1)*(1+1e-10))
        for seed in [baseline,best]:
            sol=model.createSol()
            selected=set(seed)
            for j,var in enumerate(z):model.setSolVal(sol,var,float(j in selected))
            fund=cash+(paths[:,seed]-cash[:,None]).sum(axis=1)/20
            returns=((fund[1:]/fund[:-1]-1)[mask])/sigma
            avg=float(returns.mean())
            for k,t in enumerate(transitions):
                model.setSolVal(sol,nvars[k],float(fund[t]))
                model.setSolVal(sol,xvars[k],float(returns[k]))
                model.setSolVal(sol,deviations[k],float(returns[k]-avg))
            model.setSolVal(sol,mean,avg)
            model.addSol(sol,free=True)
        model.optimize()
        scip_status=str(model.getStatus());scip_bound=float(model.getDualbound())
        upper=min(upper,scip_bound)
        for sol in model.getSols():
            ids=[j for j,var in enumerate(z) if model.getSolVal(sol,var)>.5]
            if (matrix[:,ids].sum(axis=1)<=caps).all() and nav_risk(paths,ids,20,cash,mask)<=allowed:
                if score[ids].sum()>score[best].sum()+1e-10:best=ids
        status='global_optimum' if upper-score[best].sum()<1e-6 else 'time_limited_feasible'
    final_score=float(score[best].sum());final_var=nav_risk(paths,best,20,cash,mask)
    assert (matrix[:,best].sum(axis=1)<=caps).all() and final_var<=allowed
    assert final_score>=base_score-1e-7
    return best,dict(b=b,baseline_score=base_score,selected_score=final_score,
        baseline_variance=budget,selected_variance=final_var,variance_ratio=final_var/budget,
        score_upper_bound=upper,absolute_score_gap=max(0.,upper-final_score),
        relative_score_gap=max(0.,upper-final_score)/max(abs(final_score),1e-12),
        certified_global=bool(upper-final_score<1e-6),termination=status,
        scip_status=scip_status,scip_bound=scip_bound,
        selected_count=len(best),elapsed_seconds=time.monotonic()-started,
        candidate_count=len(rows),linear_optimum_variance_ratio=variance/budget,
        candidate_top_n_cap=None,cardinality_fixed=False)
