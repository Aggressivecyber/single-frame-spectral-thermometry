"""Independent CPU contracts for the explicit local-Hermite calibration.

No registered raw is inverted.  Full-range calibration tests, a scalar coded
raw cusp fixture and a CPU32/grid4 profile derivative test have distinct scopes.
"""
from .. import storage as s
from .. import measurement as m
from ..smooth_calibration import (SmoothCalibrationPhysics,physics,contract,
    HALF_WINDOW_K,CALIBRATION_VERSION,tensor_sha)
from ..joint_estimator import SharedProfileObjective
from physics import SpectralPhysics
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import argparse
import copy
import json
import time
import numpy as np
import torch


def _matrix(p,temperatures):
    values,derivative=p.continuum(torch.as_tensor(temperatures,dtype=p.M.dtype)[None,None])
    return values[0,:,0].T,derivative[0,:,0].T


def _integrate_planck(cfg,temperatures,order):
    """Independent float64 Gauss-Legendre integration of supplied band/QE model."""
    nodes,weights=np.polynomial.legendre.leggauss(order)
    temperatures=np.asarray(temperatures,dtype=np.float64)
    all_T=np.r_[temperatures,float(cfg['reference_temperature_K'])]
    counts=[]
    sampled=cfg.get('sampled_instrument_response')
    for k,(center,width,qe) in enumerate(zip(cfg['centers_nm'],cfg['widths_nm'],cfg['quantum_efficiencies'])):
        if sampled is None:
            lam=(center+nodes*width/2)*1e-9
            spectral_weights=weights*(width/2)*1e-9*qe
        else:
            lam=np.asarray(sampled['wavelength_nm'])*1e-9
            response=np.asarray(sampled['response'][k])*np.asarray(sampled.get('fixed_emissivity_ratio',np.ones(len(lam))))
            trap=np.r_[np.diff(lam)[0]/2,(lam[2:]-lam[:-2])/2,np.diff(lam)[-1]/2]
            spectral_weights=trap*response
        exponent=.01438776877/(all_T[:,None]*lam[None])
        flux=2*299792458./lam[None]**4/np.expm1(exponent)
        counts.append(flux@spectral_weights)
    counts=np.stack(counts,1)
    scale=cfg.get('absolute_radiometric_scale',cfg['reference_905_e']/counts[-1,-1])
    return counts[:-1]*scale


def _small_reference(cfg):
    small=copy.deepcopy(cfg);small['shape']=[32,32]
    small['carrier_bins_xy']=[[round(x/8),round(y/8)] for x,y in cfg['carrier_bins_xy']]
    p=SpectralPhysics(small,dtype=torch.float32)
    for module in p.modules():
        for name,value in module._buffers.items():
            if value is not None:
                module._buffers[name]=value.to(torch.complex128 if value.is_complex() else torch.float64)
    return p


def _minimum_derivative(secant,left_slope,right_slope):
    a=-6*secant+3*(left_slope+right_slope)
    b=6*secant-4*left_slope-2*right_slope
    safe_a=torch.where(a!=0,a,torch.ones_like(a))
    u=(-b/(2*safe_a)).clamp(0,1)
    vertex=left_slope+b*u+a*u.square()
    minimum=torch.minimum(left_slope,right_slope)
    return torch.where((a>0)&(u>0)&(u<1),torch.minimum(minimum,vertex),minimum)


def run(output=None):
    started=time.perf_counter();torch.set_num_threads(2)
    p=physics('cpu');old=m.physics('cpu');c=contract(p)
    checks=[]
    def check(name,passed,**details):
        checks.append({'name':name,'passed':bool(passed),**s.plain(details)})

    original=dict(old.named_buffers());adapted=dict(p.named_buffers())
    exact=all(tensor_sha(value)==tensor_sha(adapted[name]) for name,value in original.items())
    check('all_original_optical_and_calibration_buffers_and_cfg_are_byte_identical',
        exact and p.cfg==old.cfg and c['original_buffers_byte_identical']
        and c['original_config_unchanged']
        and type(p).color_temperature is SpectralPhysics.color_temperature,
        original_buffers=len(original),common_color_lookup_unchanged=c['common_color_lookup_unchanged'],
        adapter_only_buffer_names=sorted(set(adapted)-set(original)))

    x=p.temperatures.numpy();internal=x[1:-1]
    near=np.r_[-1e-8,1e-8,-1e-6,1e-6,-HALF_WINDOW_K-1e-6,HALF_WINDOW_K+1e-6,
                -HALF_WINDOW_K+1e-6,HALF_WINDOW_K-1e-6]
    # Every half-window has 32 subdivisions, plus explicit points extremely
    # close to both seams and the center.  Wider intervals retain midpoint QA.
    offsets=np.unique(np.r_[np.linspace(-HALF_WINDOW_K,HALF_WINDOW_K,65),near,-2.5,2.5])
    dense=np.unique(np.r_[x,(internal[:,None]+offsets[None]).ravel(),x[0]+2.5,x[-1]-2.5])
    values,derivatives=_matrix(p,dense);legacy,_=_matrix(old,dense)
    knots,_=_matrix(p,x)
    check('all_calibration_knots_are_preserved_bit_for_bit',torch.equal(knots,p.continuum_lut),knots=len(x),bands=values.shape[1])
    indices=np.searchsorted(x,dense,side='right').clip(1,len(x)-1)-1
    lower=p.continuum_lut[indices];upper=p.continuum_lut[indices+1]
    tolerance=1e-10
    check('dense_full_range_is_monotone_and_has_no_overshoot',
        bool((derivatives>=0).all() and (values[1:]>=values[:-1]).all()
            and (values>=lower-tolerance).all() and (values<=upper+tolerance).all()),
        dense_temperature_points=len(dense),bands=values.shape[1],minimum_derivative_e_per_K=float(derivatives.min()),
        half_window_subdivisions=32,temperature_range_K=[float(x[0]),float(x[-1])])

    secants=(p.continuum_lut[1:]-p.continuum_lut[:-1])/(p.temperatures[1:]-p.temperatures[:-1])[:,None]
    shared=p.c1_shared_slopes[1:-1]
    left_min=_minimum_derivative(secants[:-1],secants[:-1],shared)
    right_min=_minimum_derivative(secants[1:],shared,secants[1:])
    band_bounds=HALF_WINDOW_K*(4/27)*torch.maximum((shared-secants[:-1]).abs(),(shared-secants[1:]).abs()).max(0).values
    relay_patterns=torch.fft.ifft2(torch.fft.fft2(p.op.patterns)*p.op.relay_otf).real
    raw_kernel_max=relay_patterns.abs().flatten(1).max(1).values
    raw_upper=float(band_bounds@raw_kernel_max)
    observed=(values-legacy).abs();dense_raw_upper=float((observed@raw_kernel_max).max())
    check('analytic_all_temperature_monotonicity_and_frozen_calibration_raw_error_bound',
        float(torch.minimum(left_min.min(),right_min.min()))>=0
        and bool((observed.max(0).values<=band_bounds+1e-9).all())
        and raw_upper*float(np.exp(.3))<=.17,
        analytic_minimum_derivative_e_per_K=float(torch.minimum(left_min.min(),right_min.min())),
        formula='max |Hermite-line|=(4/27)*0.25K*|shared_slope-secant|; raw infinity norm <= sum band_bound*max(abs(relay(pattern)))',
        per_band_absolute_bound_e=band_bounds.tolist(),analytic_raw_maximum_absolute_bound_e=raw_upper,
        analytic_raw_bound_over_admissible_flat_amplitude_e=raw_upper*float(np.exp(.3)),
        dense_raw_maximum_absolute_upper_bound_e=dense_raw_upper,budget_e=.17,
        scope='all uniform-temperature flat fields over the closed calibration range at unit amplitude; incremental change relative to frozen LUT')

    distance_left=np.where(indices>0,dense-x[indices],np.inf)
    distance_right=np.where(indices<len(x)-2,x[indices+1]-dense,np.inf)
    away=np.minimum(distance_left,distance_right)>=HALF_WINDOW_K
    check('outside_fixed_windows_original_straight_line_is_retained',
        float((values[away]-legacy[away]).abs().max())<1e-10,
        maximum_absolute_difference_e=float((values[away]-legacy[away]).abs().max()),points=int(away.sum()))
    common_T=old.color_temperature(values.T[None,:,None])[0]
    old_T=old.color_temperature(legacy.T[None,:,None])[0]
    color_error=float((common_T-old_T).abs().max())
    check('unchanged_common_color_reporting_stays_within_registered_incremental_budget',
        color_error<=.2 and c['common_color_lookup_unchanged'],maximum_absolute_difference_K=color_error,
        budget_K=.2,denominator=len(dense),includes_all_finite_low_signal_temperatures=True)

    outside_T=torch.tensor([x[0]-100.,x[-1]+100.],dtype=torch.float64)[None,None].requires_grad_(True)
    outside_C,outside_d=p.continuum(outside_T)
    outside_autograd=torch.autograd.grad(outside_C.sum(),outside_T)[0]
    expected=p.continuum_lut[[0,-1]].T[None,:,None]
    nonfinite_rejected=False
    try: p.continuum(torch.tensor([[[float('nan')]]],dtype=torch.float64))
    except ValueError: nonfinite_rejected=True
    check('out_of_range_guard_has_no_extrapolation_and_value_derivative_are_consistent',
        torch.equal(outside_C,expected) and bool((outside_d==0).all())
        and bool((outside_autograd==0).all()) and nonfinite_rejected,
        extension='constant endpoint count; zero analytic and autograd derivative outside range',
        nonfinite_temperatures_rejected=nonfinite_rejected,
        no_C1_claim_across_outer_range_boundaries=True)

    second=physics('cpu');second_contract=contract(second)
    check('calibration_fingerprint_and_CPU_derived_slopes_are_deterministic',
        c==second_contract and c['derived_slope_compute_device']=='cpu float64, then exact device transfer',
        fingerprint=c['fingerprint'],derived_slope_sha256=c['derived_slope_sha256'],
        GPU_executed=False,scope='CPU deterministic construction; transfer preserves payload bytes, no device arithmetic defines slopes')

    # The rejected full-interval PCHIP candidate is kept as a negative result.
    pc=PchipInterpolator(x,p.continuum_lut.numpy(),axis=0)
    full=np.arange(x[0],x[-1]+.01,.3125)
    full_old,_=_matrix(old,full);full_values=pc(full)
    full_delta=full_values-full_old.numpy()
    candidate_bound=np.abs(full_delta)@raw_kernel_max.numpy()
    worst=int(candidate_bound.argmax());raw_delta=torch.einsum('k,khw->hw',torch.tensor(full_delta[worst]),relay_patterns)
    full_negative={'status':'REJECTED_CALIBRATION_CHANGE_EXCEEDS_FROZEN_RAW_BUDGET',
        'dense_points':len(full),'maximum_band_absolute_change_e':float(np.abs(full_delta).max()),
        'selected_temperature_K':float(full[worst]),'selected_raw_maximum_absolute_change_e':float(raw_delta.abs().max()),
        'selected_raw_RMS_change_e':float(raw_delta.square().mean().sqrt()),'budget_e':.17,
        'enabled':False,'reason':'full-interval smoothing changes the frozen calibration beyond the registered incremental budget'}
    check('standard_full_interval_PCHIP_negative_result_is_preserved_and_disabled',
        full_negative['selected_raw_maximum_absolute_change_e']>.17 and not full_negative['enabled'],**full_negative)

    planck64=_integrate_planck(p.cfg,dense,64);planck128=_integrate_planck(p.cfg,dense,128)
    convergence=float(np.max(np.abs(planck128-planck64)))
    absolute_old=np.abs(legacy.numpy()-planck128);absolute_new=np.abs(values.numpy()-planck128)
    planck_color=old.color_temperature(torch.as_tensor(planck128.T)[None,:,None])[0]
    planck={'definition':'independent float64 Planck photon integral with supplied synthetic rectangular band/QE; Gauss-Legendre128',
        'calibration_is_measured':False,'quadrature64_vs128_maximum_absolute_count_difference_e':convergence,
        'old_LUT_per_band_maximum_absolute_bias_e':absolute_old.max(0).tolist(),
        'C1_per_band_maximum_absolute_bias_e':absolute_new.max(0).tolist(),
        'old_LUT_raw_flat_absolute_bias_upper_bound_e':float((absolute_old@raw_kernel_max.numpy()).max()),
        'C1_raw_flat_absolute_bias_upper_bound_e':float((absolute_new@raw_kernel_max.numpy()).max()),
        'old_LUT_common_color_maximum_absolute_bias_K':float((old_T-planck_color).abs().max()),
        'C1_common_color_maximum_absolute_bias_K':float((common_T-planck_color).abs().max()),
        'scope':'absolute synthetic-Planck approximation diagnostic; this adapter retains most original PLUT intervals and does not claim to remove their absolute bias',
        'acceptance_budget_scope':'0.17e and0.2K constrain incremental change relative to the frozen calibration, not absolute recovery of Planck physics'}
    check('independent_true_Planck_integral_is_finite_positive_monotone_and_numerically_converged',
        np.isfinite(planck128).all() and (planck128>0).all()
        and (np.diff(planck128,axis=0)>0).all() and convergence<1e-7,**planck)

    # Test actual analytic/autograd derivatives at every knot and window seam.
    seams=np.unique(np.r_[internal,internal-HALF_WINDOW_K,internal+HALF_WINDOW_K])
    Ts=torch.tensor(seams,dtype=torch.float64)[None,None].requires_grad_(True)
    cv,cd=p.continuum(Ts);autograd_errors=[]
    for band in range(7):
        automatic=torch.autograd.grad(cv[:,band].sum(),Ts,retain_graph=True)[0]
        autograd_errors.append(float((automatic-cd[:,band]).abs().max()))
    e=1e-4;f0,_=_matrix(p,seams);plus,_=_matrix(p,seams+e);minus,_=_matrix(p,seams-e)
    plus2,_=_matrix(p,seams+2*e);minus2,_=_matrix(p,seams-2*e)
    central=(plus-minus)/(2*e);right=(-3*f0+4*plus-plus2)/(2*e);left=(3*f0-4*minus+minus2)/(2*e)
    exact_d=cd[0,:,0].T.detach();denom=exact_d.abs().clamp_min(1.)
    fd_errors={name:float(((derivative-exact_d).abs()/denom).max()) for name,derivative in
        [('central',central),('left_second_order',left),('right_second_order',right)]}
    _,old_left=_matrix(old,internal-1e-7);_,old_right=_matrix(old,internal+1e-7)
    _,new_left=_matrix(p,internal-1e-7);_,new_right=_matrix(p,internal+1e-7)
    old_jump=float((old_right-old_left).abs().max());new_jump=float((new_right-new_left).abs().max())
    check('same_function_analytic_autograd_central_and_one_sided_derivatives_connect_across_all_knots',
        max(autograd_errors)<1e-8 and max(fd_errors.values())<2e-5
        and new_jump<old_jump*2e-5,
        seams=len(seams),autograd_maximum_absolute_error_e_per_K=max(autograd_errors),
        finite_difference_errors=fd_errors,old_maximum_slope_jump_e_per_K=old_jump,
        C1_slope_difference_over_2e_minus7_K_e_per_K=new_jump)

    reference=_small_reference(old.cfg);smooth=SmoothCalibrationPhysics(reference)
    knot=3300.;eps=1e-7
    Tk=reference.M.new_full((1,32,32),knot)
    K,_=reference.continuum(Tk);_,dleft=reference.continuum(Tk-eps);_,dright=reference.continuum(Tk+eps)
    predicted=reference.op(K);L=reference.op(dleft);R=reference.op(dright)
    W=m.fixed_weights(reference,predicted);middle=(L+R)/2;difference=R-L
    inner=lambda a,b: (W*a*b).sum()
    residual=difference-middle*inner(difference,middle)/inner(middle,middle)
    residual*=.002/inner(residual,residual)
    target=predicted-residual
    def scalar(T,model):
        t=torch.tensor(T,dtype=torch.float64,requires_grad=True)
        count,_=model.continuum(t.expand(1,32,32));rr=model.op(count)-target
        objective=.5*(W*rr.square()).sum()
        return float(objective.detach()),float(torch.autograd.grad(objective,t)[0])
    left_gradient=float(inner(residual,L));right_gradient=float(inner(residual,R))
    stationary=brentq(lambda t:scalar(t,smooth)[1],knot-HALF_WINDOW_K,knot+HALF_WINDOW_K,xtol=1e-12)
    stationarity=abs(scalar(stationary,smooth)[1])
    check('controlled_actual_coded_raw_scalar_fixture_reproduces_PLUT_cusp_and_C1_stationary_solution',
        left_gradient<0<right_gradient and float(target.min())>0 and stationarity<1e-6,
        raw_shape=[32,32],fixture_source='constructed finite positive scalar calibration fixture, not a registered raw or scientific scene',
        PLUT_left_derivative=left_gradient,PLUT_right_derivative=right_gradient,
        PLUT_returned_derivative_at_knot=scalar(knot,reference)[1],
        C1_stationary_temperature_K=stationary,C1_actual_gradient_absolute=stationarity,
        scope='demonstrates that a nonsmooth calibration cusp can defeat a smooth gradient criterion; does not identify the cause of v001 endpoints')

    z=reference.M.new_zeros((1,2,4,4));z[:,0]=(knot-reference.T0)/reference.Tscale;z[:,1]=np.log(.53)
    Csmall=m.continuum(smooth,z,grid=4)
    xx=torch.arange(32,dtype=torch.float64)[None,None,:]
    raw_small=reference.op(Csmall)+.2*torch.sin(2*torch.pi*xx/32)
    obj=SharedProfileObjective(smooth,raw_small,z,physical_options={'grid':4},numerical_options={'cost_limit_s':15.})
    flat=z.numpy().ravel();point=obj.evaluate(flat)
    direction=np.r_[np.ones(16)/4,np.zeros(16)];step=1e-6
    f0=point[0];fp=obj.evaluate(flat+step*direction)[0];fm=obj.evaluate(flat-step*direction)[0]
    fpp=obj.evaluate(flat+2*step*direction)[0];fmm=obj.evaluate(flat-2*step*direction)[0]
    analytic=float(point[1]@direction)
    profile_derivatives={'analytic':analytic,'central':(fp-fm)/(2*step),
        'left_second_order':(3*f0-4*fm+fmm)/(2*step),
        'right_second_order':(-3*f0+4*fp-fpp)/(2*step)}
    profile_error=max(abs(value-analytic)/max(abs(analytic),1.) for value in profile_derivatives.values())
    check('actual_CPU32_grid4_profile_gradient_connects_through_calibration_knots',profile_error<2e-5,
        directional_derivatives=profile_derivatives,maximum_normalized_error=profile_error,
        linear_modes=obj.linear.U.dimension,formal_inverse_calls=0,profile_evaluations=5)

    registration=s.read(s.REPORTS/'repair_001/quick_v001/QUICK_REGISTRATION.json')
    registered=registration.get('source_sha256',registration.get('sources',{}))
    if isinstance(registered,list):
        registered={r['path']:r['sha256'] for r in registered}
    frozen={name:s.sha(s.ROOT/name)==digest for name,digest in registered.items()}
    check('all_v001_registered_source_bytes_remain_unchanged',bool(frozen) and all(frozen.values()),
        files=len(frozen),mismatches=[name for name,passed in frozen.items() if not passed])

    result={'time':s.now(),'all_passed':all(row['passed'] for row in checks),'tests':checks,
        'scope':'locally authored CPU calibration and derivative contracts; no GPU, registered raw inversion, scientific candidate selection or final payload access',
        'calibration_contract':c,'enable_for_bounded_quick':all(row['passed'] for row in checks),
        'standard_PCHIP_negative_result':full_negative,'independent_Planck_diagnostic':planck,
        'root_cause_conclusion':'PLUT derivative discontinuities and a controlled cusp mechanism are established; contribution to native v001 failures remains unconfirmed until parameter-cache diagnostics',
        'formal_inverse_calls':0,'native256_inverse_calls':0,'GPU_used':False,
        'synthetic_scalar_fixture_count':1,'small_profile_evaluations':5,
        'scientific_validation':False,'thresholds_relaxed':False,'historical_results_reinterpreted':False,
        'wall_s':time.perf_counter()-started,'source_sha256':{name:s.sha(s.ROOT/name) for name in (
            'research_closeout/smooth_calibration.py','research_closeout/tests/smooth_calibration.py',
            'research_closeout/joint_estimator.py','research_closeout/numerical_strategy.py',
            'research_closeout/measurement.py','physics.py')}}
    s.write(output or s.REPORTS/'repair_001/calibration_v003/CALIBRATION_CONTRACTS.json',result,replace=True)
    print(json.dumps(result,indent=2,ensure_ascii=True),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output')
    raise SystemExit(0 if run(parser.parse_args().output)['all_passed'] else 2)
