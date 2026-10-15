"""CPU32 physical-profile and adversarial certification contracts.

These are locally authored software fixtures.  They do not open registered
raws, run native256 inversions, or establish scientific performance.
"""
from .. import storage as s
from .. import measurement as m
from .. import numerical_strategy as ns
from ..joint_estimator import _DeadlineLinearProfile
from config import default_config
from physics import SpectralPhysics
from scipy.optimize import OptimizeResult
from unittest.mock import patch
import argparse
import ast
import copy
import json
import subprocess
import time
import numpy as np
import torch


class FixtureClock:
    def __init__(self): self.value=100.
    def __call__(self): return self.value


def run(output=None):
    started=time.perf_counter();torch.set_num_threads(2)
    p=SpectralPhysics(default_config(32),dtype=torch.float64)
    z=p.M.new_zeros((1,5,32,32));z[:,0]=.1023;z[:,1]=np.log(.53)
    C=m.continuum(p,z[:,:2])
    yy,xx=torch.meshgrid(torch.arange(32,dtype=p.M.dtype),torch.arange(32,dtype=p.M.dtype),indexing='ij')
    template=12.*torch.cos(2*torch.pi*xx/32)
    raw_e=p.op(C)+p.op.encode_images(p.M[:,2,None,None]*template[None])[None]
    raw=raw_e/p.adc_step+p.cfg['black_level_DN']
    raw_e=p.raw_to_electrons(raw)
    physical={'grid':4,'continuum_multiplier':1.,'template_multiplier':1.}
    options={'cost_limit_s':15.,'stages':[
        {**item,'maxiter':8,'maxfun':24} for item in ns.REPAIRED_NUMERICS['stages'][:3]]}
    checks=[];invocations=0

    def check(name,passed,**details):
        checks.append({'name':name,'passed':bool(passed),**s.plain(details)})

    a=ns.RepairedObjective(p,raw_e,z[:,:2],physical_options=physical)
    smooth=ns.initialize(p,raw,'smooth_demod')
    b=ns.RepairedObjective(p,raw_e,smooth,physical_options=physical)
    eval_z=torch.nn.functional.interpolate(z[:,:2],size=(4,4),mode='bilinear',align_corners=False)
    eval_z[:,0]+=.0005*torch.sin(torch.arange(4,dtype=p.M.dtype))[None,:,None]
    eval_z[:,1]+=.004*torch.cos(torch.arange(4,dtype=p.M.dtype))[None,None,:]
    flat=eval_z.numpy().ravel();pa=a.evaluate(flat);pb=b.evaluate(flat)
    objective_error=abs(pa[0]-pb[0]);gradient_error=float(np.max(np.abs(pa[1]-pb[1])))
    check('same_raw_two_initializers_share_physical_regularizer_objective_and_gradient',
        a.physical_contract==b.physical_contract and torch.equal(a.W,b.W)
        and torch.equal(a.regscale,b.regscale) and torch.equal(a.linear.reg,b.linear.reg)
        and objective_error<1e-10 and gradient_error<1e-10,
        absolute_objective_difference=objective_error,maximum_gradient_difference=gradient_error,
        distinct_initial_states=not torch.equal(a.z0,b.z0),real_template_modes=a.linear.U.dimension)

    scaled=ns.RepairedObjective(p,raw_e,smooth,physical_options=physical)
    scaled.coordinate_scale=scaled.coordinate_scale*np.r_[np.full(16,.37),np.full(16,2.4)]
    fv,gv=scaled.scaled_fun(flat/scaled.coordinate_scale)
    chain=float(np.linalg.norm(gv-pa[1]*scaled.coordinate_scale)/max(np.linalg.norm(gv),1e-30))
    check('changing_numerical_coordinates_preserves_physical_objective_and_chain_rule',
        abs(fv-pa[0])<1e-9 and chain<1e-9,
        absolute_objective_difference=abs(fv-pa[0]),gradient_chain_relative_error=chain)

    rng=np.random.default_rng(20261015);derivatives=[]
    for channel in (0,1,None):
        direction=rng.normal(size=flat.shape)
        if channel is not None: direction[(1-channel)*16:(2-channel)*16]=0
        direction/=np.linalg.norm(direction)
        step=2e-6
        fd=(a.evaluate(flat+step*direction)[0]-a.evaluate(flat-step*direction)[0])/(2*step)
        analytic=float(pa[1]@direction)
        error=abs(fd-analytic)/max(abs(analytic),1.)
        derivatives.append({'channel':channel,'finite_difference':fd,'analytic':analytic,'relative_error':error})
    qr,_,_,_=a.linear.factor_reference(raw_e-p.op(pa[3]))
    qr_point=a.evaluate_linear_solution(flat,qr,{'numerical_certified':True,'solver':'small-mode QR'})
    qr_gradient=float(np.linalg.norm(qr_point[1]-pa[1])/max(np.linalg.norm(qr_point[1]),1.))
    check('actual_physical_profile_gradient_matches_directional_differences_and_QR',
        max(d['relative_error'] for d in derivatives)<2e-5 and qr_gradient<1e-7,
        directions=derivatives,QR_gradient_relative_difference=qr_gradient,
        nonlinear_shape=[4,4],raw_shape=[32,32],linear_columns=a.linear.n)

    constant_e=p.op(C);constant_raw=constant_e/p.adc_step+p.cfg['black_level_DN']
    def invoke(raw_dn=constant_raw,state=z,numerical=None):
        nonlocal invocations
        invocations+=1
        return ns.repaired_infer(raw_dn,p,initializer='v3',v3_state=state,
            physical_options=physical,numerical_options=numerical or options)

    constant=invoke();ci=constant['info']
    check('actual_CPU32_constant_fixture_requires_three_qualified_endpoints_and_two_tighter_plateaus',
        constant['numerical_valid'] and len(ci['stages'])==3
        and all(row['qualified_endpoint'] for row in ci['stages'])
        and len(ci['output_comparisons'])==2 and all(row['passed'] for row in ci['output_comparisons'])
        and ci['returned_stage']==3 and ci['returned_result_role']=='tightest_certified_endpoint',
        returned_stage=ci['returned_stage'],qualified=[r['qualified_endpoint'] for r in ci['stages']],
        actual_projected_gradients=[r['gradient']['scaled_projected_linf'] for r in ci['stages']],
        comparisons=ci['output_comparisons'],wall_s=ci['wall_s'])

    def fake_optimizer(fun,start,**kwargs):
        value,_=fun(start)
        return OptimizeResult(x=np.asarray(start).copy(),fun=value,success=True,
            message='adversarial success without optimization',nit=0,nfev=1)

    wrong=z.clone();wrong[:,0]+=.01;wrong[:,1]+=.02
    with patch.object(ns,'minimize',fake_optimizer): false_success=invoke(state=wrong)
    fi=false_success['info']
    check('optimizer_success_cannot_replace_actual_stage_gradient',
        not false_success['numerical_valid'] and all(r['optimizer']['success'] for r in fi['stages'])
        and all(not r['qualified_endpoint'] for r in fi['stages'])
        and all(r['gradient']['scaled_projected_linf']>r['configuration']['gtol'] for r in fi['stages']),
        actual_projected_gradients=[r['gradient']['scaled_projected_linf'] for r in fi['stages']],
        requested_gtol=[r['configuration']['gtol'] for r in fi['stages']])

    flags=iter([True,False,True,True])
    def intermittent_optimizer(fun,start,**kwargs):
        fit=fake_optimizer(fun,start,**kwargs);fit.success=next(flags);return fit
    four=copy.deepcopy(options);four['stages'].append({**ns.REPAIRED_NUMERICS['stages'][3],'maxiter':8,'maxfun':24})
    with patch.object(ns,'minimize',intermittent_optimizer): intermittent=invoke(numerical=four)
    ii=intermittent['info']
    check('three_nonconsecutive_qualified_endpoints_do_not_certify',
        not intermittent['numerical_valid'] and sum(r['qualified_endpoint'] for r in ii['stages'])==3
        and ii['observed_consecutive_stable_comparisons']==1,
        qualified=[r['qualified_endpoint'] for r in ii['stages']],comparisons=ii['output_comparisons'])

    shifted=[]
    def moving_optimizer(fun,start,**kwargs):
        coordinate_scale=fun.__self__.coordinate_scale
        nxt=np.asarray(start).copy();nxt[0]+=.00001/coordinate_scale[0]
        shifted.append(nxt*coordinate_scale)
        value,_=fun(nxt)
        return OptimizeResult(x=nxt,fun=value,success=True,message='controlled finite endpoint',nit=1,nfev=1)
    with patch.object(ns,'minimize',moving_optimizer): moving=invoke(state=wrong)
    returned=moving['z_grid'].cpu().numpy().ravel()
    check('returns_latest_tightened_endpoint_instead_of_original_base',
        moving['info']['returned_stage']==3 and np.allclose(returned,shifted[-1],rtol=0,atol=1e-15)
        and not np.allclose(returned,shifted[0],rtol=0,atol=1e-8),
        returned_stage=moving['info']['returned_stage'],maximum_error=float(np.max(np.abs(returned-shifted[-1]))))

    clock=FixtureClock()
    def timed_optimizer(fun,start,**kwargs):
        fit=fake_optimizer(fun,start,**kwargs);clock.value+=.4;return fit
    short=copy.deepcopy(options);short['cost_limit_s']=1.
    with patch.object(ns.time,'perf_counter',clock),patch.object(ns,'minimize',timed_optimizer):
        timeout=invoke(numerical=short)
    ti=timeout['info']
    check('single_total_deadline_is_not_reset_between_stages_and_finite_endpoint_survives',
        not timeout['numerical_valid'] and ti['timed_out'] and ti['finite_output']
        and ti['returned_stage']==2 and len(ti['stages'])==3,
        returned_stage=ti['returned_stage'],timed_out=ti['timed_out'],finite_output=ti['finite_output'],errors=ti['errors'])

    clock=FixtureClock();real_initialize=ns.initialize
    def slow_initialize(*args,**kwargs):
        result=real_initialize(*args,**kwargs);clock.value+=2.;return result
    with patch.object(ns.time,'perf_counter',clock),patch.object(ns,'initialize',slow_initialize),patch.object(ns,'minimize') as optimizer:
        init_timeout=invoke(numerical=short)
    iti=init_timeout['info']
    check('initializer_time_is_inside_total_deadline_and_finite_unsolved_state_is_retained',
        not init_timeout['numerical_valid'] and iti['timed_out'] and iti['finite_output']
        and iti['returned_result_role']=='initializer_only' and not optimizer.called,
        timed_out=iti['timed_out'],returned_result_role=iti['returned_result_role'],wall_s=iti['wall_s'])

    original_solve=_DeadlineLinearProfile.solve;solve_calls=0
    def failed_inner(self,*args,**kwargs):
        nonlocal solve_calls
        solve_calls+=1;x,info=original_solve(self,*args,**kwargs)
        if solve_calls==2: info={**info,'numerical_certified':False,'status':'injected inner failure'}
        return x,info
    with patch.object(_DeadlineLinearProfile,'solve',failed_inner),patch.object(ns,'minimize',fake_optimizer):
        failed=invoke(state=wrong)
    failure=failed['info']
    check('inner_failure_retains_last_finite_iterate_without_certification',
        not failed['numerical_valid'] and failure['finite_output'] and failure['errors']
        and failure['errors'][0]['type']=='ArithmeticError'
        and failure['returned_result_role']=='interrupted_stage_finite_iterate',
        returned_result_role=failure['returned_result_role'],finite_output=failure['finite_output'],errors=failure['errors'])

    rejected=[]
    loosened=copy.deepcopy(ns.REPAIRED_NUMERICS['stages']);loosened[0]['gtol']=1e-3
    not_tightening=copy.deepcopy(ns.REPAIRED_NUMERICS['stages']);not_tightening[1]['gtol']=not_tightening[0]['gtol']
    for bad in ({'cost_limit_s':181.},{'numeric_template_budget_e':.171},
                {'numeric_color_budget_K':.201},{'required_stable_comparisons':1},
                {'inner_tolerance':1e-9},{'reference_inner_tolerance':1e-10},
                {'stages':loosened},{'stages':not_tightening},
                {'stages':copy.deepcopy(ns.REPAIRED_NUMERICS['stages'][:2])},
                {'grid':4},{'unregistered_option':True}):
        try: ns._repair_options(bad);rejected.append(False)
        except ValueError: rejected.append(True)
    check('registered_time_output_inner_and_outer_accuracy_contracts_cannot_be_relaxed',
        all(rejected),rejected=len([x for x in rejected if x]),attempts=len(rejected))

    def function_tree(text,name):
        tree=ast.parse(text)
        return ast.dump(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)
    legacy=[]
    for file,names in [('research_closeout/joint_estimator.py',('ProfileObjective','infer')),
                       ('research_closeout/numerical_strategy.py',('RecoveredObjective','recovered_infer'))]:
        base=subprocess.check_output(['git','show','HEAD:'+file],cwd=s.ROOT,text=True,encoding='utf8')
        current=(s.ROOT/file).read_text(encoding='utf8')
        legacy.extend(function_tree(base,name)==function_tree(current,name) for name in names)
    check('historical_objective_and_inverse_function_bodies_preserved',all(legacy),compared_definitions=4)

    result={'time':s.now(),'all_passed':all(row['passed'] for row in checks),'tests':checks,
        'scope':'locally authored CPU32 synthetic physical-profile software contracts; no native256 inversion, registered raw, scientific selection, GPU or final payload access',
        'raw_shape':[32,32],'nonlinear_shape':[4,4],'real_template_modes':a.linear.U.dimension,
        'formal_inverse_calls':0,'bounded_synthetic_profile_invocations':invocations,
        'formal_raws_generated':False,'scientific_validation':False,'GPU_used':False,
        'wall_s':time.perf_counter()-started,
        'source_sha256':{name:s.sha(s.ROOT/name) for name in (
            'research_closeout/joint_estimator.py','research_closeout/numerical_strategy.py',
            'research_closeout/tests/numerical_repair.py','research_closeout/linear_profile.py',
            'research_closeout/measurement.py','physics.py','config.py')}}
    s.write(output or s.REPORTS/'repair_001/NUMERICAL_REPAIR_CONTRACTS.json',result,replace=True)
    print(json.dumps(result,ensure_ascii=True,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output')
    raise SystemExit(0 if run(parser.parse_args().output)['all_passed'] else 2)
