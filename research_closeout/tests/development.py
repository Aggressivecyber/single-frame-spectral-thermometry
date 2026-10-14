"""Numerical recovery must preserve the objective and strong B1 definition."""
from .. import storage as s
from .. import measurement as m
from ..baselines import FrozenBaselines
from ..numerical_strategy import RecoveredObjective,RECOVERY,recovered_infer
from ..joint_estimator import ProfileObjective,DEFAULTS
from ..linear_profile import LinearProfile
from ..observable_basis import RealFourierBasis
from ..metrics import aggregate,resolution_scores
from .contracts import FiniteBasis
import json
import numpy as np
import torch

def run():
    torch.set_num_threads(2);torch.manual_seed(406018);p=m.physics('cpu');rng=np.random.default_rng(406018)
    basis=RealFourierBasis(p.op.object_otf);finite=FiniteBasis(basis,torch.linspace(0,basis.dimension-1,25).long())
    z=torch.zeros((1,2,4,4),dtype=torch.float64);z[:,0]=.003;z[:,1]=-.4
    raw_e=p.op(m.continuum(p,z,grid=4))+.04
    opt={**DEFAULTS,**RECOVERY,'grid':4,'cost_limit_s':90.,'base_limit_s':90.}
    normal=ProfileObjective(p,raw_e,z,opt);recovered=RecoveredObjective(p,raw_e,z,opt)
    for obj in (normal,recovered): obj.linear=LinearProfile(p,obj.W,basis=finite)
    flat=z.numpy().ravel();a=normal.evaluate(flat,forced_tolerance=1e-12)
    recovered.force_tolerance=1e-12;b=recovered.scaled_fun(flat/recovered.coordinate_scale)
    objective_error=abs(a[0]-b[0]);gradient_error=np.linalg.norm(a[1]-b[1]/recovered.coordinate_scale)/max(np.linalg.norm(a[1]),1)
    d=rng.normal(size=flat.size);d/=np.linalg.norm(d);v=flat/recovered.coordinate_scale;step=1e-4
    fd=(recovered.scaled_fun(v+step*d)[0]-recovered.scaled_fun(v-step*d)[0])/(2*step)
    derivative_error=abs(fd-b[1]@d)/max(abs(fd),1)
    from evaluation_v2.methods import demod_fit
    suite=FrozenBaselines('cpu');p32=suite.p
    raw=raw_e.float()/p32.adc_step+p32.cfg['black_level_DN']
    checks=[]
    def check(name,passed,**evidence): checks.append({'name':name,'passed':bool(passed),**evidence})
    check('scaled_LBFGSB_preserves_original_profile_objective',objective_error<1e-8,absolute_error=objective_error)
    check('scaled_coordinates_exact_gradient_chain_rule',gradient_error<1e-10,relative_error=gradient_error)
    check('scaled_physical_profile_finite_difference',derivative_error<1e-5,relative_error=derivative_error)
    constant=torch.zeros((1,5,256,256),dtype=torch.float64);constant[:,0]=.003;constant[:,1]=-.4
    constant_raw=p.op(m.continuum(p,constant[:,:2]))/p.adc_step+p.cfg['black_level_DN']
    fixture=recovered_infer(constant_raw,p,{'grid':4,'outer_maxiter':8,'reference_maxiter':8,
        'base_limit_s':45.,'cost_limit_s':60.},initializer='v3',v3_state=constant)
    check('whole_recovered_inverse_tighter_outer_finite_fixture',fixture['numerical_valid']
        and fixture['info']['regularizer_frozen_during_tighter_outer']
        and fixture['info']['nonnegative_predicted_measurement'],
        status=fixture['info']['status'],color_outer_difference_RMS_K=fixture['info']['color_outer_difference_RMS_K'],
        template_outer_difference_RMS_e=fixture['info']['template_outer_difference_RMS_e'],
        scope='constant synthetic numerical fixture with known initial state; not a scientific performance result')
    for lam in (.1,10.,100.,1000.):
        out=suite.traditional(raw,lam);reference=demod_fit(p32,p32.raw_to_electrons(raw),lam)
        error=float((out['template_postblur_e_ref']-reference['species_signed_e'][:,2]).abs().max())
        color=float((p32.color_temperature(out['continuum_postblur_bands'])[0]-out['temperature_color_K']).abs().max())
        check(f'B1_lambda{lam:g}_original_implementation_and_continuum_domain',error==0 and color<.2,
            template_maximum_difference_e=error,color_roundtrip_maximum_K=color,
            color_roundtrip_budget_K=.2,color_budget_source='registered numerical output budget; LUT interpolation is not an exact inverse')
    all_rows=[{'metrics':{'template_SSE':4.,'temperature_SSE_K2':9.,'domain_pixels':1}}, {'metrics':None}]
    total=aggregate(all_rows)
    check('failed_scientific_case_retains_denominator',total['failed_or_unavailable_cases']==1 and not total['all_cases_finite'])
    result={'time':s.now(),'all_passed':all(c['passed'] for c in checks),'tests':checks,
        'native_grid':[256,256],'profile_nonlinear_grid':[4,4],'profile_real_template_modes':25,
        'formal_raws_generated':False,'formal_performance_or_numerical_qualification_claimed':False}
    s.write(s.REPORTS/'DEVELOPMENT_CONTRACTS.json',result,replace=True)
    print(json.dumps(result,ensure_ascii=True),flush=True)
    return result

if __name__=='__main__': raise SystemExit(0 if run()['all_passed'] else 2)
