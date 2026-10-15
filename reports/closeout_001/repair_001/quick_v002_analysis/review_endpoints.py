"""Read-only four-cell CPU endpoint audit; never calls a complete inverse.

All new persistent files live beside this script. Frozen source, completed
results, raw files and parameter/output caches are only read and hashed.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from research_closeout import storage as s,measurement as m
from research_closeout.smooth_calibration import physics,contract,HALF_WINDOW_K
from research_closeout.numerical_strategy import RepairedObjective
import copy
import hashlib
import json
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
RUN=s.REPORTS/'repair_001/quick_v002'
OUTPUT=HERE/'NUMERICAL_ENDPOINT_REVIEW.json'
LINEAR_TOLERANCE=1e-13
LINEAR_MAXITER=4800
CANONICAL={'T':'temperature_color_K','u':'template_postblur_e_ref',
    'C':'continuum_postblur_bands','predicted_raw_e':'predicted_raw_e',
    'valid':'valid','low_signal':'low_signal'}
COUNTERS={'formal_inverse_calls':0,'outer_optimizer_calls':0,'conditional_linear_solves':0,
    'fixed_parameter_gradient_evaluations':0,'unique_final_nonlinear_endpoints':0}
TOUCHED={}


def array_sha(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str(value.dtype).encode()+json.dumps(list(value.shape)).encode())
    h.update(value.tobytes());return h.hexdigest()


def checked(path,digest=None):
    path=Path(path);actual=s.sha(path)
    if digest is not None and actual!=digest: raise ValueError('File SHA256 mismatch: '+str(path))
    TOUCHED[str(path.resolve())]=actual
    return actual


def checked_ref(ref): return checked(ref['path'],ref['sha256'])


def identity(value,expected):
    differences={key:{'expected':item,'actual':value.get(key)} for key,item in expected.items() if value.get(key)!=item}
    if differences: raise ValueError('Identity mismatch: '+json.dumps(differences))


def quantiles(value):
    q=[0,.01,.05,.25,.5,.75,.95,.99,1]
    return {str(p):float(v) for p,v in zip(q,np.quantile(np.asarray(value),q))}


def grad_record(g,z,objective):
    scale=objective.coordinate_scale;flat=z.detach().cpu().numpy().ravel()
    n=objective.g**2;limits=objective.p.cfg['temperature_range_K']
    lower=np.r_[np.full(n,(limits[0]-objective.p.T0)/objective.p.Tscale),np.full(n,-4.)]/scale
    upper=np.r_[np.full(n,(limits[1]-objective.p.T0)/objective.p.Tscale),np.full(n,.3)]/scale
    v=flat/scale;pg=v-np.clip(v-g*scale,lower,upper)
    return {'finite':bool(np.isfinite(g).all()),'physical_l2':float(np.linalg.norm(g)),
        'physical_linf':float(np.abs(g).max()),'scaled_projected_linf':float(np.abs(pg).max()),
        'active_lower_coordinates':int(np.sum(v<=lower)),
        'active_upper_coordinates':int(np.sum(v>=upper))}


def vector_summary(g,objective):
    g=np.asarray(g).reshape(objective.shape)
    scale=objective.coordinate_scale.reshape(objective.shape)
    return {'physical_l2':float(np.linalg.norm(g)),'physical_linf':float(np.abs(g).max()),
        'C1_scaled_l2':float(np.linalg.norm(g*scale)),
        'C1_scaled_linf':float(np.abs(g*scale).max()),
        'temperature_channel_l2':float(np.linalg.norm(g[:,0])),
        'loga_channel_l2':float(np.linalg.norm(g[:,1]))}


def regularizer(z,regscale,step):
    dx=(z[:,:,:,1:]-z[:,:,:,:-1])/step
    dy=(z[:,:,1:]-z[:,:,:-1])/step
    phi=.5*(regscale*(dx.square().sum((0,2,3))+dy.square().sum((0,2,3)))).sum()
    with torch.no_grad():
        g=torch.zeros_like(z);wx=dx.detach()*regscale[None,:,None,None]/step
        wy=dy.detach()*regscale[None,:,None,None]/step
        g[:,:,:,:-1]-=wx;g[:,:,:,1:]+=wx
        g[:,:,:-1]-=wy;g[:,:,1:]+=wy
    return phi,g


def endpoint(objective,z0,x,regscale=None):
    """Independent objective decomposition, AD and explicit native-grid gradient."""
    COUNTERS['fixed_parameter_gradient_evaluations']+=1
    p=objective.p;z=z0.detach().clone().requires_grad_(True)
    if objective.g!=p.cfg['shape'][0]: raise ValueError('This analytic audit is explicitly native-grid')
    regscale=objective.regscale if regscale is None else regscale
    C=m.continuum(p,z,grid=objective.g);rawC=p.op(C)
    residual=rawC+objective.linear.apply(x)[None]-objective.raw_e
    data=.5*(objective.W*residual.square()).sum()
    linear=.5*(objective.linear.reg*x).square().sum()
    prior,prior_g=regularizer(z,regscale,p.cfg['shape'][0]/objective.g)
    phi=data+linear+prior
    ad=torch.autograd.grad(phi,z)[0].detach()
    with torch.no_grad():
        back=p.op.adjoint(objective.W*residual.detach())
        T=p.T0+p.Tscale*z[:,0].detach();_,dC=p.continuum(T)
        amplitude=z[:,1:2].detach().exp()
        explicit=torch.stack([(back*amplitude*dC*p.Tscale).sum(1),
            (back*C.detach()).sum(1)],1)+prior_g
    gradient=ad.cpu().numpy().ravel()
    return {'objective':float(phi.detach()),'data_objective':float(data.detach()),
        'linear_regularizer_objective':float(linear.detach()),
        'continuum_regularizer_objective':float(prior.detach()),
        'gradient':gradient,'prior_gradient':prior_g.cpu().numpy().ravel(),
        'AD_vs_explicit_maximum_absolute_error':float((ad-explicit).abs().max()),
        'AD_vs_explicit_normalized_l2_error':float((ad-explicit).norm()/ad.norm().clamp_min(1.)),
        'C':C.detach(),'rawC':rawC.detach(),'residual':residual.detach(),
        'record':grad_record(gradient,z0,objective)}


def stripped(point):
    return {key:value for key,value in point.items() if key not in
        ('gradient','prior_gradient','C','rawC','residual')}


def output_at(p,raw_e,objective,point,x):
    return m.output(p,raw_e,point['C'],objective.linear.U.unpack(x[:-3])[None],x[-3:][None],{})


def output_difference(a,b):
    result={}
    for name,key in CANONICAL.items():
        aa=np.asarray(a[key].detach().cpu());bb=np.asarray(b[name])
        if aa.dtype.kind=='b' or bb.dtype.kind=='b':
            result[name]={'different_pixels':int(np.count_nonzero(aa!=bb)),'exact_equal':bool(np.array_equal(aa,bb))}
        else:
            delta=aa-bb
            result[name]={'RMS':float(np.sqrt(np.mean(delta*delta))),'maximum_absolute':float(np.abs(delta).max())}
    return result


def conditional_normal(objective,C,x):
    target=objective.raw_e-objective.p.op(C)
    linear=objective.linear
    r=torch.cat([(linear.sqrtW*(target.reshape(linear.p.cfg['shape'])-linear.apply(x))).flatten(),-linear.reg*x])
    b=torch.cat([(linear.sqrtW*target.reshape(linear.p.cfg['shape'])).flatten(),linear.reg.new_zeros(linear.n)])
    relative=float(linear.aug_adjoint(r).norm()/linear.aug_adjoint(b).norm().clamp_min(1e-30))
    return {'actual_scaled_normal_relative':relative,'scaled_normal_l2':float(linear.aug_adjoint(r).norm())}


def linear_refine(objective,C,x0):
    COUNTERS['conditional_linear_solves']+=1
    return objective.linear.solve(objective.raw_e-objective.p.op(C),tolerance=LINEAR_TOLERANCE,
        maxiter=LINEAR_MAXITER,x0=x0)


def stage_contract(core):
    stages=core['stages'];comparisons=core['output_comparisons'];opt=core['numerical_options']
    records=[];stage_ok={}
    for i,stage in enumerate(stages):
        config=stage['configuration']
        strictly_tighter=i==0 or all(config[k]<stages[i-1]['configuration'][k] for k in ('gtol','ftol'))
        calculated=bool(stage['complete'] and stage['optimizer']['success']
            and stage['gradient']['finite'] and stage['gradient']['scaled_projected_linf']<=config['gtol']
            and stage['inner']['numerical_certified'] and stage['tighter_inner_reference']['numerical_certified']
            and stage['template_inner_difference_RMS_e']<=opt['numeric_template_budget_e']
            and stage['finite_output'] and stage['nonnegative_predicted_measurement'])
        passed=calculated and stage['qualified_endpoint'] and stage['requested_projected_gradient_passed']
        passed=passed and strictly_tighter and config==opt['stages'][i]
        stage_ok[stage['stage']]=passed
        records.append({'stage':stage['stage'],'configuration':config,'optimizer':stage['optimizer'],
            'recorded_gradient':stage['gradient'],'calculated_gtol_passed':stage['gradient']['scaled_projected_linf']<=config['gtol'],
            'inner':stage['inner'],'tighter_inner_reference':stage['tighter_inner_reference'],
            'template_inner_difference_RMS_e':stage['template_inner_difference_RMS_e'],
            'qualification_logic_recomputed':calculated,'record_contract_passed':passed,
            'parameter_cache_available':stage['stage']==core['returned_stage'],
            'evidence_scope':'final stage independently recomputed below; earlier-stage gradients are frozen log evidence only'})
    count=0;plateaus=[]
    for comp in comparisons:
        within=comp['template_RMS_difference_e']<=opt['numeric_template_budget_e'] and comp['color_RMS_difference_K']<=opt['numeric_color_budget_K']
        both=bool(stage_ok.get(comp['from_stage']) and stage_ok.get(comp['to_stage']))
        calculated=within and both and comp['to_stage']==comp['from_stage']+1
        count=count+1 if calculated else 0
        flags=comp['within_output_budgets']==within and comp['both_endpoints_qualified']==both
        flags=flags and comp['passed']==calculated and comp['consecutive_passed_comparisons']==count
        plateaus.append({**comp,'recomputed_contract_passed':bool(flags)})
    passed=bool(len(stages)>=3 and len(comparisons)>=2 and all(stage_ok.values()) and count>=2
        and count==core['observed_consecutive_stable_comparisons'] and all(p['recomputed_contract_passed'] for p in plateaus)
        and not core['errors'] and not core['timed_out'] and core['returned_stage']==stages[-1]['stage'])
    return {'passed':passed,'stages':records,'output_platforms':plateaus,
        'independently_replayed_stage_parameters':1,'earlier_stage_parameters_not_saved':len(stages)-1,
        'platform_output_arrays_not_saved':True,
        'scope':'logical/receipt verification of all recorded stages and two comparisons; endpoint physics independently verified only for saved final parameters'}


def knot_statistics(T,calibration_delta_gradient,objective):
    p=objective.p;first=float(p.temperatures[0]);step=p.cfg['temperature_lut_step_K']
    nearest=np.rint((T-first)/step).astype(int);knot=first+step*nearest
    distance=np.abs(T-knot);internal=(nearest>0)&(nearest<len(p.temperatures)-1)
    window=internal&(distance<=HALF_WINDOW_K)
    bins={str(threshold):int(np.count_nonzero(internal&(distance<=threshold))) for threshold in
        (1e-8,1e-6,1e-4,.001,.01,.05,.1,.25,.5,1.,2.5)}
    total=T.size;g=calibration_delta_gradient.reshape(objective.shape)[0]
    energy=np.sum(g*g,axis=0);all_energy=float(energy.sum())
    largest=np.argsort(np.max(np.abs(g),axis=0).ravel())[-20:][::-1]
    top=[]
    for index in largest:
        y,x=np.unravel_index(index,T.shape)
        top.append({'y':int(y),'x':int(x),'latent_temperature_K':float(T[y,x]),
            'nearest_knot_K':float(knot[y,x]),'distance_K':float(distance[y,x]),
            'inside_C1_window':bool(window[y,x]),'calibration_gradient_difference_temperature':float(g[0,y,x]),
            'calibration_gradient_difference_loga':float(g[1,y,x])})
    unique,counts=np.unique(knot[window],return_counts=True)
    most=sorted(zip(unique.tolist(),counts.tolist()),key=lambda item:item[1],reverse=True)[:12]
    return {'latent_pixels':total,'latent_temperature_quantiles_K':quantiles(T),
        'nearest_5K_knot_distance_quantiles_K':quantiles(distance),
        'internal_knot_denominator':int(internal.sum()),'distance_counts':bins,
        'within_fixed_C1_half_window_pixels':int(window.sum()),'within_fixed_C1_half_window_fraction':float(window.mean()),
        'left_of_knot_inside_window':int(np.count_nonzero(window&(T<knot))),
        'right_of_knot_inside_window':int(np.count_nonzero(window&(T>knot))),
        'exact_float_knot_inside_window':int(np.count_nonzero(window&(T==knot))),
        'most_populated_window_knots':[{'temperature_K':k,'pixels':n} for k,n in most],
        'calibration_only_gradient_difference_L2_energy_fraction_in_windows':None if all_energy==0 else float(energy[window].sum()/all_energy),
        'top20_calibration_gradient_difference_locations':top}


def process_cell(cell,registration,digest,p,old,reference):
    started=time.perf_counter();cell_id={key:cell[key] for key in ('method','case_id','initializer','raw_sha256','cache_key')}
    expected={key:cell[key] for key in ('registration_sha256','method','raw_sha256','cache_key','case_id')}
    if cell['registration_sha256']!=digest: raise ValueError('Cell registration differs')
    raw_path=s.LEGACY/reference['raw_reference']['path'];checked(raw_path,reference['raw_reference']['sha256'])
    with np.load(raw_path,allow_pickle=False) as handle: raw_array=handle['raw_dn'].copy()
    if array_sha(raw_array)!=reference['raw_reference']['array_sha256']: raise ValueError('Raw array SHA differs')
    if str(raw_array.dtype)!=reference['raw_reference']['dtype'] or list(raw_array.shape)!=reference['raw_reference']['shape']:
        raise ValueError('Raw dtype/shape differs')
    if cell['raw_sha256']!=reference['raw_reference']['sha256']: raise ValueError('Cell names another raw')
    raw=m.raw_tensor(raw_array,p);raw_e=p.raw_to_electrons(raw)
    receipts={}
    for name in ('output_cache','output_metadata','inference_receipt','score_receipt'):
        checked_ref(cell[name]);receipts[name]=cell[name]
    inference=s.read(cell['inference_receipt']['path']);score=s.read(cell['score_receipt']['path'])
    output_metadata=s.read(cell['output_metadata']['path'])
    for value in (inference,score,output_metadata): identity(value,expected)
    if score['metrics']!=cell['metrics']: raise ValueError('Scoring metric receipt differs from summary')
    parameter_receipt=cell['parameter_cache_receipt'];identity(parameter_receipt,expected)
    for name in ('parameter_cache','parameter_metadata'): checked_ref(parameter_receipt[name])
    parameter_meta=s.read(parameter_receipt['parameter_metadata']['path']);identity(parameter_meta,expected)
    for key in ('source_sha256','config_fingerprint','objective_fingerprint','calibration_contract'):
        if parameter_meta[key]!=registration[key]: raise ValueError('Parameter registration mismatch: '+key)
    arrays=s.arrays(parameter_receipt['parameter_cache']['path'])
    if set(arrays)!={'z_grid','xi','beta'}: raise ValueError('Parameter array set differs')
    for name,value in arrays.items():
        if value.dtype!=np.float64 or not np.isfinite(value).all(): raise ValueError('Nonfinite/nonfloat64 '+name)
        if array_sha(value)!=parameter_meta['array_sha256'][name]: raise ValueError('Parameter SHA mismatch '+name)
        if list(value.shape)!=parameter_meta['array_shapes'][name]: raise ValueError('Parameter shape mismatch '+name)
    output_arrays=s.arrays(cell['output_cache']['path'])
    for name,value in output_arrays.items():
        actual=array_sha(value)
        if any(mapping[name]!=actual for mapping in
            (output_metadata['array_sha256'],cell['array_sha256'],parameter_meta['canonical_array_sha256'])):
            raise ValueError('Canonical array hashes disagree: '+name)
        if list(value.shape)!=output_metadata['array_shapes'][name]: raise ValueError('Canonical shape differs')
    for name in CANONICAL:
        if name not in output_arrays: raise ValueError('Missing canonical output: '+name)
    if any(output_arrays[name].dtype!=np.float64 for name in ('T','u','C','predicted_raw_e')):
        raise ValueError('Canonical floating precision differs')
    core=cell['info']['numerical_engine'];physical=registration['config']['physical_options']
    if cell['info']['physical_objective_version']!=registration['config']['physical_objective_version']:
        raise ValueError('Effective objective version differs')
    if cell['info']['calibration_contract']!=contract(p): raise ValueError('Current CPU C1 calibration differs')
    row_info=copy.deepcopy(cell['info']);row_info.pop('parameter_cache_receipt',None)
    if parameter_meta['inference_info']!=row_info: raise ValueError('Saved parameter inference info differs')
    z=torch.as_tensor(arrays['z_grid']);x=torch.cat([torch.as_tensor(arrays['xi']),torch.as_tensor(arrays['beta']).ravel()])
    deadline=time.perf_counter()+120.
    objective=RepairedObjective(p,raw_e,z,physical_options=physical,numerical_options=core['numerical_options'],deadline=deadline)
    original=RepairedObjective(old,raw_e,z,physical_options=physical,numerical_options=core['numerical_options'],deadline=deadline)
    COUNTERS['unique_final_nonlinear_endpoints']+=1
    saved=endpoint(objective,z,x);saved_output=output_at(p,raw_e,objective,saved,x)
    reconstruction=output_difference(saved_output,output_arrays)
    reconstruction_passed=all(value.get('maximum_absolute',0)<=1e-7 and value.get('different_pixels',0)==0 for value in reconstruction.values())
    logged_gradient=core['returned_gradient'];reported_phi=core['returned_endpoint_objective']
    gdelta={name:saved['record'][name]-logged_gradient[name] for name in
        ('physical_l2','physical_linf','scaled_projected_linf')}
    final_gtol=core['stages'][-1]['configuration']['gtol']
    final_gradient_passed=saved['record']['scaled_projected_linf']<=final_gtol
    saved_normal=conditional_normal(objective,saved['C'],x)
    warm,warm_info=linear_refine(objective,saved['C'],x)
    cold,cold_info=linear_refine(objective,saved['C'],None)
    warm_point=endpoint(objective,z,warm);cold_point=endpoint(objective,z,cold)
    def inner_comparison(new,point):
        du=float(objective.linear.U.unpack((new-x)[:-3]).square().mean().sqrt())
        pred=point['rawC']+objective.linear.apply(new)[None]
        raw_delta=pred-saved_output['predicted_raw_e']
        return {'template_RMS_difference_e':du,'beta_maximum_absolute_difference':float((new[-3:]-x[-3:]).abs().max()),
            'predicted_raw_RMS_difference_e':float(raw_delta.square().mean().sqrt()),
            'objective_difference':point['objective']-saved['objective'],
            'gradient_difference_physical_l2':float(np.linalg.norm(point['gradient']-saved['gradient'])),
            'actual_gradient':point['record'],'final_stage_gtol_passed':point['record']['scaled_projected_linf']<=final_gtol}
    warm_comparison=inner_comparison(warm,warm_point);cold_comparison=inner_comparison(cold,cold_point)
    tighter_passed=bool(warm_info['numerical_certified'] and cold_info['numerical_certified']
        and warm_comparison['template_RMS_difference_e']<=core['numerical_options']['numeric_template_budget_e']
        and cold_comparison['template_RMS_difference_e']<=core['numerical_options']['numeric_template_budget_e']
        and warm_comparison['final_stage_gtol_passed'] and cold_comparison['final_stage_gtol_passed'])
    old_same_parameters=endpoint(original,z,x)
    old_fixed_C1_prior=endpoint(original,z,x,objective.regscale)
    old_reference,old_reference_info=linear_refine(original,old_same_parameters['C'],x)
    old_profile=endpoint(original,z,old_reference)
    old_profile_fixed_C1_prior=endpoint(original,z,old_reference,objective.regscale)
    calibration_gradient=saved['gradient']-old_fixed_C1_prior['gradient']
    anchor_gradient=old_fixed_C1_prior['gradient']-old_same_parameters['gradient']
    total_gradient=saved['gradient']-old_same_parameters['gradient']
    closure=float(np.linalg.norm(total_gradient-calibration_gradient-anchor_gradient))
    old_output=output_at(old,raw_e,original,old_same_parameters,x)
    old_out_arrays={name:old_output[key].detach().numpy() for name,key in CANONICAL.items()}
    interpolation_delta=output_difference(saved_output,old_out_arrays)
    T=(p.T0+p.Tscale*z[:,0]).numpy()[0]
    knot=knot_statistics(T,calibration_gradient,objective)
    stages=stage_contract(core)
    logged_consistent=abs(saved['objective']-reported_phi)<1e-7 and abs(gdelta['scaled_projected_linf'])<1e-9
    certified=bool(reconstruction_passed and final_gradient_passed and tighter_passed and logged_consistent
        and saved['AD_vs_explicit_normalized_l2_error']<1e-10 and stages['passed']
        and saved_normal['actual_scaled_normal_relative']<=1e-8)
    raw_residual=saved_output['predicted_raw_e']-raw_e
    raw_rmse=float(raw_residual.square().mean().sqrt())
    selected_metrics={key:cell['metrics'][key] for key in (
        'domain_pixels','raw_pixels','valid_fraction','full_FOV_valid_fraction','template_RMSE_e',
        'template_bias_e','template_relative_L2','temperature_reference_color_RMSE_K',
        'temperature_reference_color_bias_K','temperature_reference_color_comparison_pixels',
        'temperature_latent_truth_RMSE_K','temperature_latent_truth_comparison_pixels',
        'raw_RMSE_e','raw_weighted_RMSE','support_pixels','sign_recall')}
    if abs(raw_rmse-cell['metrics']['raw_RMSE_e'])>1e-7: raise ValueError('Independent raw metric differs')
    old_control=next(row for row in registration['v001_summary']['cells'] if row['case_id']==cell['case_id']
        and row['method']=='joint_c1_s1_repaired_v2__'+cell['initializer'])
    old_cached=s.arrays(old_control['output_cache']['path']);checked_ref(old_control['output_cache'])
    v001_differences=output_difference(saved_output,old_cached)
    anchor={'original_PLUT_regscale':original.regscale.tolist(),'C1_regscale':objective.regscale.tolist(),
        'C1_over_original_ratio':(objective.regscale/original.regscale).tolist(),
        'original_coordinate_channel_scale':original.coordinate_contract['channel_scale'],
        'C1_coordinate_channel_scale':objective.coordinate_contract['channel_scale'],
        'calibration_anchor_temperature_K':objective.physical_contract['parameters']['anchor_temperature_K'],
        'interpretation':'knot values are identical at3300K; changed temperature derivative changes the temperature prior scale and coordinate metric. The log-amplitude anchor scale stays unchanged.'}
    return ({**cell_id,'review_status':'PASSED' if certified else 'REVIEW_CHECK_FAILED',
        'reported_numerical_certified':cell['production_numerical_certified'],
        'independent_final_endpoint_passed':certified,'integrity_passed':True,
        'raw_and_parameter_hashes_verified':True,'parameter_dtype':'float64',
        'parameter_shapes':{key:list(value.shape) for key,value in arrays.items()},
        'canonical_array_hashes_verified':True,'canonical_CPU_reconstruction':reconstruction,
        'canonical_CPU_reconstruction_passed':reconstruction_passed,
        'recorded_stage_contract':stages,'independent_final_endpoint':stripped(saved),
        'final_stage_requested_gtol':final_gtol,'final_stage_independent_gtol_passed':final_gradient_passed,
        'logged_vs_recomputed_objective_difference':saved['objective']-reported_phi,
        'logged_vs_recomputed_gradient_differences':gdelta,'saved_conditional_normal':saved_normal,
        'tighter_inner':{'tolerance':LINEAR_TOLERANCE,'maxiter':LINEAR_MAXITER,
            'warm_start':{'solver':warm_info,'comparison':warm_comparison},
            'cold_start':{'solver':cold_info,'comparison':cold_comparison},
            'warm_vs_cold_template_RMS_e':float(objective.linear.U.unpack((warm-cold)[:-3]).square().mean().sqrt()),
            'outer_parameters_changed':False,'passed':tighter_passed},
        'same_final_parameters_original_PLUT':stripped(old_same_parameters),
        'same_final_parameters_original_PLUT_with_fixed_C1_prior':stripped(old_fixed_C1_prior),
        'same_z_original_PLUT_conditional_profile':{'inner':old_reference_info,
            'original_prior':stripped(old_profile),'fixed_C1_prior':stripped(old_profile_fixed_C1_prior),
            'template_RMS_shift_from_C1_e':float(objective.linear.U.unpack((old_reference-x)[:-3]).square().mean().sqrt())},
        'same_parameter_calibration_output_differences':interpolation_delta,
        'anchor_scale_change':anchor,
        'same_parameter_gradient_decomposition':{
            'path':'C1/C1prior minus PLUT/C1prior, then PLUT/C1prior minus PLUT/PLUTprior; x,z fixed throughout',
            'calibration_values_and_derivatives_only':vector_summary(calibration_gradient,objective),
            'anchor_regularizer_only':vector_summary(anchor_gradient,objective),
            'total':vector_summary(total_gradient,objective),'closure_physical_l2':closure,
            'objective_calibration_only':saved['objective']-old_fixed_C1_prior['objective'],
            'objective_anchor_regularizer_only':old_fixed_C1_prior['objective']-old_same_parameters['objective']},
        'knot_and_window_statistics':knot,'v001_to_v002_output_differences':v001_differences,
        'external_scientific_metrics':selected_metrics,
        'scientific_metrics_scope':'copied from SHA-verified external scoring receipt; diagnostic truth not reopened in this audit',
        'independent_raw_RMSE_e':raw_rmse,'scientific_acceptance_assessed':False,
        'wall_s':time.perf_counter()-started},
        {'T':output_arrays['T'],'u':output_arrays['u'],'z':arrays['z_grid'],'xi':arrays['xi'],
            'objective':saved['objective'],'case_id':cell['case_id'],'initializer':cell['initializer']})


def main():
    started=time.perf_counter();torch.set_num_threads(2)
    registration_path=RUN/'QUICK_REGISTRATION.json';result_path=RUN/'QUICK_RESULT.json'
    sidecar=s.read(RUN/'QUICK_REGISTRATION_SHA256.json');checked(RUN/'QUICK_REGISTRATION_SHA256.json')
    digest=checked(registration_path,sidecar['sha256']);checked(result_path)
    registration=s.read(registration_path);result=s.read(result_path)
    if result['registration_sha256']!=digest: raise ValueError('Quick result registration differs')
    source_results={name:checked(s.ROOT/name,digest)==digest for name,digest in registration['source_sha256'].items()}
    for ref in registration['contracts']+registration['v001_files']: checked_ref(ref)
    expected={(ref['case_id'],route['method']) for ref in registration['references'] for route in registration['config']['routes']}
    supplied=result['summary']['cells']
    if len(supplied)!=4 or {(row['case_id'],row['method']) for row in supplied}!=expected:
        raise ValueError('Full four-cell denominator is not present exactly once')
    p=physics('cpu');old=m.physics('cpu')
    if contract(p)!=registration['calibration_contract']: raise ValueError('CPU calibration does not equal registered calibration')
    refs={ref['case_id']:ref for ref in registration['references']}
    rows=[];small=[]
    for index,cell in enumerate(supplied,1):
        print('BEGIN_AUDIT',index,cell['case_id'],cell['initializer'],flush=True)
        try:
            row,aux=process_cell(cell,registration,digest,p,old,refs[cell['case_id']]);rows.append(row);small.append(aux)
        except Exception as error:
            rows.append({key:cell[key] for key in ('method','case_id','initializer')})
            rows[-1].update(review_status='FAILED_RETAINED_IN_DENOMINATOR',
                independent_final_endpoint_passed=False,integrity_passed=False,error_type=type(error).__name__,
                error=str(error),traceback=traceback.format_exc())
        print('END_AUDIT',index,rows[-1]['review_status'],flush=True)
        s.write(HERE/'REVIEW_PROGRESS.json',{'time':s.now(),'completed_cells':index,'planned_cells':4,
            'rows':s.plain(rows),'counters':COUNTERS},replace=True)
    paired=[]
    for case_id in refs:
        same=[row for row in small if row['case_id']==case_id]
        if len(same)!=2:
            paired.append({'case_id':case_id,'status':'NOT_AVAILABLE_RETAINED','available':len(same)});continue
        a,b=same
        paired.append({'case_id':case_id,'initializers':[a['initializer'],b['initializer']],
            'temperature_color_RMS_difference_K':float(np.sqrt(np.mean((a['T']-b['T'])**2))),
            'template_RMS_difference_e':float(np.sqrt(np.mean((a['u']-b['u'])**2))),
            'latent_temperature_RMS_difference_K':float(p.Tscale*np.sqrt(np.mean((a['z'][:,0]-b['z'][:,0])**2))),
            'log_amplitude_RMS_difference':float(np.sqrt(np.mean((a['z'][:,1]-b['z'][:,1])**2))),
            'objective_difference':a['objective']-b['objective'],
            'scope':'two qualified local endpoints agree on these two reused development raws; not a global-optimum or scientific-generalization proof'})
    unchanged={path:s.sha(path)==digest for path,digest in TOUCHED.items()}
    counts={'planned_cells':4,'audited_cells':len(rows),
        'reported_finite_cells':sum(bool(row['finite_output']) for row in supplied),
        'reported_numerically_certified_cells':sum(bool(row['production_numerical_certified']) for row in supplied),
        'independently_recomputed_final_endpoint_passed_cells':sum(bool(row['independent_final_endpoint_passed']) for row in rows),
        'retained_failed_review_cells':sum(not row['independent_final_endpoint_passed'] for row in rows),
        'scientific_acceptance_denominator_not_assessed':4}
    old_cells=registration['v001_summary']['cells']
    review={'time':s.now(),'all_audit_checks_passed':counts['independently_recomputed_final_endpoint_passed_cells']==4 and all(unchanged.values()),
        'schema':'C1v3-four-cell-readonly-CPU-endpoint-review-v001','denominators':counts,
        'source_result':{'path':str(result_path),'sha256':TOUCHED[str(result_path.resolve())]},
        'registration_sha256':digest,'calibration_version':registration['config']['calibration_version'],
        'physical_objective_version':registration['config']['physical_objective_version'],
        'source_hashes_verified':source_results,'all_frozen_files_unchanged_after_review':all(unchanged.values()),
        'frozen_files_rechecked':len(unchanged),'file_sha256':TOUCHED,'cells':rows,'cross_initializer_pairs':paired,
        'v001_control_denominator':{'planned_cells':len(old_cells),
            'finite_cells':sum(bool(row['finite_output']) for row in old_cells),
            'numerically_certified_cells':sum(bool(row['production_numerical_certified']) for row in old_cells),
            'cells':[{key:row[key] for key in ('case_id','method','status','finite_output','production_numerical_certified')} for row in old_cells],
            'historical_failures_relabelled':False,'latent_parameters_saved':False},
        'limits':['Only final-stage parameters were saved. Earlier-stage gradients and both platform distances are SHA-bound production log evidence, not independently replayed parameter states.',
            'C1 changes interpolation and the anchor temperature regularizer/coordinate scale. Fixed-prior and fixed-parameter comparisons quantify both; this is not a precision-only rerun of the old physical estimator.',
            'Original v001 latent endpoints were not saved. Its stalled pixels cannot be linked to5K knots retrospectively from color/output maps.',
            'Four reused development cases (two raws times two initializers) do not establish scientific accuracy, independent test performance, detector calibration, uncertainty coverage or a global optimum.'],
        'mechanism_scope':'calibration and anchor contributions at the saved C1 endpoints are directly measurable; unique causal attribution for the v001 failures remains unproven',
        'scientific_acceptance_assessed':False,'truth_payloads_opened':False,'final_payloads_opened':False,
        'GPU_used':False,'complete_inversions_run':False,'counters':COUNTERS,
        'audit_script_sha256':s.sha(__file__),'wall_s':time.perf_counter()-started}
    s.write(OUTPUT,s.plain(review),replace=True)
    receipt={'time':s.now(),'review':{'path':str(OUTPUT),'sha256':s.sha(OUTPUT)},
        'script':{'path':str(Path(__file__)),'sha256':s.sha(__file__)},'all_audit_checks_passed':review['all_audit_checks_passed'],
        'denominators':counts,'counters':COUNTERS,'GPU_used':False,'frozen_files_changed':not all(unchanged.values())}
    s.write(HERE/'REVIEW_RECEIPT.json',receipt,replace=True)
    print(json.dumps(receipt,indent=2),flush=True)
    return review


if __name__=='__main__':
    raise SystemExit(0 if main()['all_audit_checks_passed'] else 2)
