"""Exact portfolio-NAV risk selection for fixed slots or fully invested NAV/K.

No candidate truncation. A missing pair cap removes the total-count constraint,
and does not remove the unchanged sector cap. Numerical bounds are reported.
"""
import time
from itertools import combinations
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
from nonoverlap_solver import Model, quicksum, ExactRiskHandler, SCIP_RESULT
from selection import incidence, nav_risk


def structure(rows, b, pair_cap=20):
    a, caps = incidence(rows)
    caps[:len(set(rows.x) | set(rows.y))] = b
    if pair_cap is not None:
        a = np.vstack([a, np.ones(len(rows), int)])
        caps = np.r_[caps, pair_cap]
    return a, caps


def greedy(rows, b=1, pair_cap=20):
    a, caps = structure(rows, b, pair_cap)
    counts = np.zeros(len(caps), int)
    ids = []
    for j in range(len(rows)):
        if (counts + a[:, j] <= caps).all():
            ids.append(j)
            counts += a[:, j]
    return ids


def risk(paths, ids, cash, mask, pair_cap=20):
    if pair_cap is None and not len(ids):
        return float(np.var((cash[1:] / cash[:-1] - 1)[mask], ddof=1))
    return nav_risk(paths, ids, len(ids) if pair_cap is None else pair_cap, cash, mask)


def local_improve(rows, paths, cash, mask, b, baseline, warm, pair_cap, budget, proposal=None):
    a, caps = structure(rows, b, pair_cap)
    scores = -rows.tstat.to_numpy(float)
    limit = budget * (1 + 1e-10) + 1e-18
    best = list(baseline)
    endpoints=list(zip(rows.x,rows.y))
    lookup={tuple(sorted(edge)):j for j,edge in enumerate(endpoints)}
    for seed in [baseline, warm]:
        if seed is not None and (a[:, seed].sum(axis=1) <= caps).all() and risk(paths, seed, cash, mask, pair_cap) <= limit:
            if scores[seed].sum() > scores[best].sum():
                best = list(seed)
    for _ in range(100):
        outside = np.setdiff1d(np.arange(len(rows)), best)
        counts = a[:, best].sum(axis=1)
        n = len(best)
        total = paths[:, best].sum(axis=1)
        if pair_cap is not None:
            total = total + (pair_cap - n) * cash
        gain, change = 1e-9, None
        for old in best + [None]:
            minus = np.zeros(len(caps)) if old is None else a[:, old]
            possible = outside[(counts[:, None] - minus[:, None] + a[:, outside] <= caps[:, None]).all(axis=0)]
            possible = possible[scores[possible] - (0 if old is None else scores[old]) > gain]
            for start in range(0, len(possible), 128):
                new = possible[start:start + 128]
                nav = total[:, None] + paths[:, new]
                if old is not None:
                    nav -= paths[:, old, None]
                elif pair_cap is not None:
                    nav -= cash[:, None]
                risks = np.var((nav[1:] / nav[:-1] - 1)[mask], axis=0, ddof=1)
                gains = scores[new] - (0 if old is None else scores[old])
                accepted = (risks <= limit) & (gains > gain) & (nav > 0).all(axis=0)
                if accepted.any():
                    k = int(np.argmax(np.where(accepted, gains, -np.inf)))
                    gain = float(gains[k]); change = old, int(new[k])
        if change is not None:
            old,new=change;change=({old} if old is not None else set(),{new})
        else:
            # Degree-preserving reconnections cover matching-specific traps.
            selected=set(best)
            for left,right in combinations(best,2):
                x,y=endpoints[left];u,v=endpoints[right]
                if len({x,y,u,v})<4:continue
                for edges in [((x,u),(y,v)),((x,v),(y,u))]:
                    new=[lookup.get(tuple(sorted(e))) for e in edges]
                    if any(j is None or j in selected for j in new):continue
                    improvement=float(scores[new].sum()-scores[[left,right]].sum())
                    if improvement<=gain:continue
                    candidate=list((selected-{left,right})|set(new))
                    if (a[:,candidate].sum(axis=1)<=caps).all() and risk(paths,candidate,cash,mask,pair_cap)<=limit:
                        gain=improvement;change=({left,right},set(new))
            # Two-for-two neighborhood drawn from the full-universe structural
            # optimum. This is only a warm-start search; the global model still
            # includes every original eligible candidate.
            priority=sorted(set(proposal or [])-selected)
            if len(priority)>=2:
                i,j=np.triu_indices(len(priority),1)
                new1=np.array(priority)[i];new2=np.array(priority)[j]
                sums=scores[new1]+scores[new2]
                new_counts=a[:,new1]+a[:,new2]
                for left,right in combinations(best,2):
                    improvements=sums-scores[left]-scores[right]
                    accepted=(improvements>gain)&(counts[:,None]-a[:,left,None]-a[:,right,None]+new_counts<=caps[:,None]).all(axis=0)
                    choices=np.flatnonzero(accepted)
                    for start in range(0,len(choices),128):
                        k=choices[start:start+128]
                        nav=total[:,None]-paths[:,left,None]-paths[:,right,None]+paths[:,new1[k]]+paths[:,new2[k]]
                        risks=np.var((nav[1:]/nav[:-1]-1)[mask],axis=0,ddof=1)
                        valid=(risks<=limit)&(nav>0).all(axis=0)
                        if valid.any():
                            m=int(np.argmax(np.where(valid,improvements[k],-np.inf)))
                            if improvements[k[m]]>gain:
                                q=k[m];gain=float(improvements[q]);change=({left,right},{int(new1[q]),int(new2[q])})
        if change is None:break
        best=list((set(best)-change[0])|change[1])
        best.sort()
    return best


class ExtensionRiskHandler(ExactRiskHandler):
    def __init__(self, *args, pair_cap=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_cap = pair_cap

    def check(self, solution=None):
        values = np.array([self.model.getSolVal(solution, v) for v in self.z])
        if len(values) and np.max(np.abs(values - np.rint(values))) > 1e-6:
            return None
        ids = tuple(np.flatnonzero(values > .5))
        if ids not in self.cache:
            self.cache[ids] = risk(self.paths, ids, self.cash, self.mask, self.pair_cap) <= self.limit
        return ids, self.cache[ids]

    def consenfolp(self, constraints, n_useful_conss, sol_infeasible):
        answer = self.check()
        if answer is None or answer[1]:
            return {'result': SCIP_RESULT.FEASIBLE}
        ids = set(answer[0])
        # Only the fixed-cap case permits the saturated-set/19-core cuts.
        cover = self.risk_cover(ids) if self.pair_cap == 20 else None
        if cover is not None:
            self.model.addCons(quicksum(self.z[j] for j in cover) <= len(cover) - 1)
            self.cover_cuts += 1
        elif self.pair_cap is not None and len(ids) == self.pair_cap:
            self.model.addCons(quicksum(self.z[j] for j in ids) <= self.pair_cap - 1)
        else:
            self.model.addCons(quicksum(1-self.z[j] if j in ids else self.z[j] for j in range(len(self.z))) >= 1)
        self.cuts += 1
        return {'result': SCIP_RESULT.CONSADDED}


def add_valid_risk_outer_bound(model,z,rows,paths,cash,mask,pair_cap,budget,best,fixed_zero,fixed_one,floor,
                             tighten=False,structure_matrix=None,structure_caps=None):
    """A proven convex relaxation; exact integer risk remains mandatory.

    L=delta NAV/N_ref is linear in z. Uniform positive NAV/delta bounds give
    |R-L|<=epsilon componentwise. Minkowski therefore implies
    sd(L)<=sd(R)+||epsilon||/sqrt(T-1). Every truly feasible portfolio is in
    this outer set; no proxy-risk acceptance or feasible candidate pruning.
    """
    if budget<=0:return None
    scores=-rows.tstat.to_numpy(float)
    max_count=min(len(rows),4*rows.sector.nunique(),pair_cap or len(rows))
    active=np.flatnonzero(~fixed_zero&~fixed_one);forced=np.flatnonzero(fixed_one)
    remain=min(max_count-len(forced),len(active))
    if pair_cap is None:
        levels=paths/max_count;offsets=np.zeros(len(cash))
        residual=floor-float(scores[forced].sum())
        required=max(0,int(np.searchsorted(np.cumsum(np.sort(scores[active])[::-1]),residual-1e-9)+1)) if residual>0 else 0
        constant=levels[:,forced].sum(axis=1)
        low=constant+np.sort(levels[:,active],axis=1)[:,:required].sum(axis=1)
        high=constant+(np.sort(levels[:,active],axis=1)[:,-remain:].sum(axis=1) if remain else 0.)
    else:
        levels=(paths-cash[:,None])/pair_cap;offsets=cash
        constant=offsets+levels[:,forced].sum(axis=1)
        low=constant+np.sort(np.minimum(levels[:,active],0),axis=1)[:,:remain].sum(axis=1)
        high=constant+(np.sort(np.maximum(levels[:,active],0),axis=1)[:,-remain:].sum(axis=1) if remain else 0.)
    ts=np.flatnonzero(mask);sigma=float(np.sqrt(budget))
    reference=offsets+levels[:,best].sum(axis=1)
    if (low[ts]<=0).any():return None
    delta=(levels[ts+1]-levels[ts])/sigma
    delta0=(offsets[ts+1]-offsets[ts])/sigma
    dc=delta0+delta[:,forced].sum(axis=1)
    dl=dc+np.sort(np.minimum(delta[:,active],0),axis=1)[:,:remain].sum(axis=1)
    dh=dc+(np.sort(np.maximum(delta[:,active],0),axis=1)[:,-remain:].sum(axis=1) if remain else 0.)
    ref=reference[ts]
    inverse_error=np.maximum(np.abs(1/low[ts]-1/ref),np.abs(1/high[ts]-1/ref))
    epsilon=np.maximum(np.abs(dl),np.abs(dh))*inverse_error
    if tighten and structure_matrix is not None:
        # Relaxed linear bounds under ALL structural constraints and the known
        # feasible score floor. Tightening an outer bound never imposes a new
        # investment rule. Concentrate solves on the largest error contributions.
        aa=np.vstack([structure_matrix,-scores]);bb=np.r_[structure_caps,-floor]
        bounds=list(zip(fixed_one.astype(int),(~fixed_zero).astype(int)))
        def extrema(coeff):
            values=[]
            scale=1e5
            for sign in [1,-1]:
                result=linprog(sign*coeff*scale,A_ub=aa,b_ub=bb,bounds=bounds,method='highs')
                if not result.success:raise RuntimeError('NAV interval relaxation failed')
                values.append(float(result.fun/scale)*sign)
            return values[0]-1e-9,values[1]+1e-9
        for k in np.argsort(-epsilon)[:min(40,len(ts))]:
            if epsilon[k]<1e-10:continue
            t=ts[k];lo,hi=extrema(levels[t])
            low[t]=max(low[t],float(offsets[t])+lo)
            high[t]=min(high[t],float(offsets[t])+hi)
            lo,hi=extrema(delta[k])
            dl[k]=max(dl[k],float(delta0[k])+lo)
            dh[k]=min(dh[k],float(delta0[k])+hi)
        inverse_error=np.maximum(np.abs(1/low[ts]-1/ref),np.abs(1/high[ts]-1/ref))
        epsilon=np.maximum(np.abs(dl),np.abs(dh))*inverse_error
    error=float(np.linalg.norm(epsilon)/np.sqrt(len(ts)-1))
    lower,upper=dl/ref,dh/ref
    variables=[]
    for k in range(len(ts)):
        rv=model.addVar(lb=float(lower[k])-1e-10,ub=float(upper[k])+1e-10,name=f'outer_r{k}')
        model.addCons(rv==float(dc[k]/ref[k])+quicksum(float(delta[k,j]/ref[k])*z[j] for j in active))
        variables.append(rv)
    ml=float(lower.min())-1e-10;mu=float(upper.max())+1e-10
    mean=model.addVar(lb=ml,ub=mu,name='outer_mean')
    model.addCons(len(ts)*mean==quicksum(variables))
    deviations=[]
    for k,rv in enumerate(variables):
        dv=model.addVar(lb=float(lower[k])-mu-1e-10,ub=float(upper[k])-ml+1e-10,name=f'outer_dev{k}')
        model.addCons(dv==rv-mean);deviations.append(dv)
    tolerance=float(np.sqrt((budget*(1+1e-10)+1e-18)/budget))
    model.addCons(quicksum(d*d for d in deviations)<=(len(ts)-1)*(tolerance+error+1e-9)**2)
    return dict(variables=variables,mean=mean,deviations=deviations,
        delta=delta,delta0=delta0,reference=ref,error=error)


def optimize(rows, paths, cash, mask, b, baseline, warm=None, pair_cap=20,
             time_limit=30., log_path=None, fractional=False, linearized=False,tighten_outer=False):
    started = time.monotonic()
    a, caps = structure(rows, b, pair_cap)
    scores = -rows.tstat.to_numpy(float)
    budget = risk(paths, baseline, cash, mask, pair_cap)
    allowed = budget * (1 + 1e-10) + 1e-18
    if not len(rows):
        return [], dict(b=b, selected_score=0., baseline_score=0., baseline_variance=budget,
            selected_variance=budget, variance_ratio=1., score_upper_bound=0., absolute_score_gap=0.,
            relative_score_gap=0., certified_global=True, termination='empty_candidate_pool',
            selected_count=0, baseline_count=0, candidate_count=0, pair_cap=pair_cap)
    linear = milp(-scores, integrality=np.ones(len(rows)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(a, -np.inf, caps), options={'mip_rel_gap': 0., 'time_limit': 60.})
    if linear.status != 0 or linear.x is None:
        raise RuntimeError('Structural MILP did not finish globally')
    proposal = np.flatnonzero(linear.x > .5).tolist()
    upper = float(-linear.mip_dual_bound)
    pv = risk(paths, proposal, cash, mask, pair_cap)
    seeds = [list(baseline)] + ([list(warm)] if warm is not None else [])
    if pv <= allowed:
        seeds.append(proposal)
    best = max(seeds, key=lambda ids: scores[ids].sum())
    cuts = checked = covers = 0
    status = 'linear_score_optimum_feasible'
    if upper - scores[best].sum() > 1e-7:
        best = local_improve(rows, paths, cash, mask, b, baseline, best, pair_cap, budget,proposal)
    if upper - scores[best].sum() > 1e-7:
        model = Model(f'extension_b{b}'); model.hideOutput()
        if log_path:
            model.setLogfile(str(log_path))
        model.setRealParam('limits/time', time_limit)
        model.setRealParam('limits/gap', 0.)
        model.setRealParam('numerics/feastol', 1e-9)
        model.setIntParam('randomization/randomseedshift', 20261106)
        model.setBoolParam('misc/allowstrongdualreds', bool(fractional and budget>0))
        model.setBoolParam('misc/allowweakdualreds', bool(fractional and budget>0))
        floor = float(scores[best].sum()) - 1e-8
        # Safe reduced-cost fixing from a relaxation of the full candidate pool.
        # A variable is fixed only when its opposite value cannot attain the
        # explicit incumbent score floor even after dropping the risk constraint.
        lp=linprog(-scores,A_ub=a,b_ub=caps,bounds=(0,1),method='highs')
        if not lp.success:raise RuntimeError('Structural LP bound failed')
        lp_upper=float(-lp.fun)
        fixed_zero=(lp_upper-lp.lower.marginals < floor-1e-6)
        fixed_one=(lp_upper+lp.upper.marginals < floor-1e-6)
        assert not (fixed_zero&fixed_one).any()
        z = [model.addVar(vtype='B',lb=int(fixed_one[j]),ub=int(not fixed_zero[j]),name=f'z_{j}') for j in range(len(rows))]
        for row, cap in zip(a, caps):
            model.addCons(quicksum(z[j] for j in np.flatnonzero(row)) <= int(cap))
        objective = quicksum(float(scores[j]) * z[j] for j in range(len(rows)))
        model.setObjective(objective, 'maximize')
        model.addCons(objective >= floor)
        if fractional and budget > 0:
            # Exact returns: n_t * x_t = delta NAV / sigma, not a covariance proxy.
            max_count = min(len(rows), 4 * rows.sector.nunique(), pair_cap or len(rows))
            if pair_cap is None:
                levels = paths / max_count
                offsets = np.zeros(len(cash))
                # Any k-edge score is bounded by the k largest individual scores.
                # This gives a valid positive NAV bound even with variable K.
                min_count=max(1,int(np.searchsorted(np.cumsum(np.sort(scores)[::-1]),floor-1e-9)+1))
                model.addCons(quicksum(z) >= min_count)
            else:
                levels = (paths - cash[:, None]) / pair_cap
                offsets = cash
            active=np.flatnonzero(~fixed_zero&~fixed_one);forced=np.flatnonzero(fixed_one)
            remaining=min(max_count-len(forced),len(active))
            constant=offsets+levels[:,forced].sum(axis=1)
            if pair_cap is None:
                residual_floor=floor-float(scores[forced].sum())
                required=max(0,int(np.searchsorted(np.cumsum(np.sort(scores[active])[::-1]),residual_floor-1e-9)+1)) if residual_floor>0 else 0
                lows=constant+np.sort(levels[:,active],axis=1)[:,:required].sum(axis=1)
                highs=constant+np.sort(levels[:,active],axis=1)[:,-remaining:].sum(axis=1) if remaining else constant.copy()
            else:
                lows=constant+np.sort(np.minimum(levels[:,active],0),axis=1)[:,:remaining].sum(axis=1)
                highs=constant+np.sort(np.maximum(levels[:,active],0),axis=1)[:,-remaining:].sum(axis=1) if remaining else constant.copy()
            sigma = float(np.sqrt(budget)); transitions = np.flatnonzero(mask)
            xvars = []; nvars = []; bounds = []; product_vars=[]
            for t in transitions:
                delta = (levels[t+1] - levels[t]) / sigma
                d0 = float((offsets[t+1] - offsets[t]) / sigma)
                dc=d0+float(delta[forced].sum())
                dl = dc + np.sort(np.minimum(delta[active], 0))[:remaining].sum()
                dh = dc + (np.sort(np.maximum(delta[active], 0))[-remaining:].sum() if remaining else 0.)
                lo, hi = float(lows[t]), float(highs[t])
                if lo <= 0:
                    raise ValueError('Unproven positive NAV bound')
                xl = min(dl/lo, dl/hi); xh = max(dh/lo, dh/hi)
                nv = model.addVar(lb=lo, ub=hi, name=f'nav_{t}')
                xv = model.addVar(lb=float(xl), ub=float(xh), name=f'ret_{t}')
                model.addCons(nv == float(offsets[t]) + quicksum(float(levels[t,j])*z[j] for j in range(len(rows))))
                products=[]
                if linearized:
                    # Exact binary-times-continuous linearization. The sole
                    # remaining nonlinear constraint is convex sample variance.
                    for j in active:
                        yv=model.addVar(lb=float(min(0,xl)),ub=float(max(0,xh)),name=f'rz_{t}_{j}')
                        model.addCons(yv>=float(xl)*z[j]);model.addCons(yv<=float(xh)*z[j])
                        model.addCons(yv>=xv-float(xh)*(1-z[j]));model.addCons(yv<=xv-float(xl)*(1-z[j]))
                        products.append((int(j),yv))
                    model.addCons(float(constant[t])*xv+quicksum(float(levels[t,j])*yv for j,yv in products)
                        == dc+quicksum(float(delta[j])*z[j] for j in active))
                else:
                    model.addCons(nv*xv == d0 + quicksum(float(delta[j])*z[j] for j in range(len(rows))))
                product_vars.append(products)
                nvars.append(nv); xvars.append(xv); bounds.append((xl, xh))
            n = len(transitions)
            mean_lo = min(v[0] for v in bounds); mean_hi = max(v[1] for v in bounds)
            mean = model.addVar(lb=float(mean_lo), ub=float(mean_hi), name='mean_return')
            model.addCons(n*mean == quicksum(xvars))
            deviations = []
            for k, xv in enumerate(xvars):
                dv = model.addVar(lb=float(bounds[k][0]-mean_hi), ub=float(bounds[k][1]-mean_lo))
                model.addCons(dv == xv-mean); deviations.append(dv)
            model.addCons(quicksum(d*d for d in deviations) <= (n-1)*(1+1e-10))
            for seed in [baseline, best]:
                sol = model.createSol(); chosen = set(seed)
                for j, var in enumerate(z): model.setSolVal(sol, var, int(j in chosen))
                fund = offsets + levels[:, seed].sum(axis=1)
                returns = (fund[1:]/fund[:-1]-1)[mask]/sigma
                avg = float(returns.mean())
                for k, t in enumerate(transitions):
                    model.setSolVal(sol, nvars[k], float(fund[t]))
                    model.setSolVal(sol, xvars[k], float(returns[k]))
                    model.setSolVal(sol, deviations[k], float(returns[k]-avg))
                    for j,yv in product_vars[k]:model.setSolVal(sol,yv,float(returns[k]) if j in chosen else 0.)
                model.setSolVal(sol, mean, avg); model.addSol(sol, free=True)
        else:
            outer=add_valid_risk_outer_bound(model,z,rows,paths,cash,mask,pair_cap,budget,best,fixed_zero,fixed_one,floor,
                tighten=tighten_outer,structure_matrix=a,structure_caps=caps)
            handler = ExtensionRiskHandler(z, paths, cash, mask, allowed, a, caps, scores, floor, pair_cap=pair_cap)
            model.includeConshdlr(handler, 'exact_nav_variance', 'Exact original-engine variance', chckpriority=-10, enfopriority=-10)
            model.addPyCons(model.createCons(handler, 'risk'))
            for seed in [baseline, best]:
                sol = model.createSol(); chosen = set(seed)
                for j, var in enumerate(z): model.setSolVal(sol, var, int(j in chosen))
                if outer is not None:
                    values=(outer['delta0']+outer['delta'][:,seed].sum(axis=1))/outer['reference']
                    mean_value=float(values.mean())
                    model.setSolVal(sol,outer['mean'],mean_value)
                    for k,var in enumerate(outer['variables']):
                        model.setSolVal(sol,var,float(values[k]))
                        model.setSolVal(sol,outer['deviations'][k],float(values[k]-mean_value))
                model.addSol(sol, free=True)
        model.optimize()
        upper = min(upper, float(model.getDualbound()))
        status = str(model.getStatus())
        for sol in model.getSols():
            ids = [j for j, var in enumerate(z) if model.getSolVal(sol, var) > .5]
            if (a[:, ids].sum(axis=1) <= caps).all() and risk(paths, ids, cash, mask, pair_cap) <= allowed:
                if scores[ids].sum() > scores[best].sum() + 1e-10:
                    best = ids
        if not fractional or budget <= 0:
            cuts, checked, covers = handler.cuts, len(handler.cache), handler.cover_cuts
    value = float(scores[best].sum()); variance = risk(paths, best, cash, mask, pair_cap)
    assert (a[:, best].sum(axis=1) <= caps).all() and variance <= allowed
    return best, dict(b=b, selected_score=value, baseline_score=float(scores[baseline].sum()),
        baseline_variance=budget, selected_variance=variance, variance_ratio=variance/budget if budget else None,
        score_upper_bound=upper, absolute_score_gap=max(0., upper-value),
        relative_score_gap=max(0., upper-value)/max(abs(value), 1e-12),
        certified_global=upper-value < 1e-6, termination=status, selected_count=len(best),
        baseline_count=len(baseline), candidate_count=len(rows), pair_cap=pair_cap,
        linear_optimum_variance_ratio=pv/budget if budget else None,
        risk_no_good_cuts=cuts, exact_risk_cover_cuts=covers, checked_integer_solutions=checked,
        elapsed_seconds=time.monotonic()-started, fractional=fractional,
        solver_time_limit_seconds=time_limit,linearized_binary_products=linearized)
