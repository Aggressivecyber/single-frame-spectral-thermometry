"""Read-only v001 review plus small calibration-anchor CPU linear algebra."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
ROOT=Path('E:/CodexWork/thermometry/closeout_001')
sys.path.insert(0,str(ROOT))
from research_closeout import storage as s
from research_closeout import measurement as m
import hashlib
import json
import time
import numpy as np
import torch


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def arrays(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def ref_check(reference):
    p=Path(reference['path'])
    return {'path':str(p),'sha256':sha(p),'expected_sha256':reference.get('sha256'),
            'matches':sha(p)==reference.get('sha256')}
def relative_ref(reference):
    p=s.LEGACY/reference['path']
    if not p.exists(): p=ROOT/reference['path']
    return {'path':str(p),'sha256':reference['sha256']}


def run():
    started=time.perf_counter();torch.set_num_threads(2)
    folder=ROOT/'reports/closeout_001/repair_001/quick_v001'
    result=read(folder/'QUICK_RESULT.json');reg=read(folder/'QUICK_REGISTRATION.json')
    registration_hash=sha(folder/'QUICK_REGISTRATION.json')
    source_checks={name:{'registered':digest,'current':sha(ROOT/name),
                         'matches':sha(ROOT/name)==digest}
                   for name,digest in reg['source_sha256'].items()}
    supporting_sources=('physics.py','repair_v3/solver.py','research_closeout/reliable_v3.py',
        'research_closeout/numerical_strategy.py','research_closeout/joint_estimator.py',
        'research_closeout/measurement.py','research_closeout/quick_validation.py',
        'research_v5/metrics.py','training_v4/model.py')
    cells=result['summary']['cells'];references={v['case_id']:v for v in reg['references']}
    p=m.physics('cpu');cfg=p.cfg
    raw_cache={};labels={};inputs=[]
    for cid in dict.fromkeys(v['case_id'] for v in cells):
        reference=references[cid]
        raw_ref=relative_ref(reference['raw_reference']);label_ref=relative_ref(reference['label_reference'])
        inputs.extend([ref_check(raw_ref),ref_check(label_ref)])
        raw_cache[cid]=arrays(raw_ref['path'])['raw_dn']
        labels[cid]=arrays(label_ref['path'])

    rows=[];stored={};file_checks=[]
    for cell in cells:
        cid=cell['case_id'];method=cell['method'];info=cell['info'];metric=cell['metrics']
        for name in ('output_cache','output_metadata','inference_receipt','score_receipt'):
            file_checks.append(ref_check(cell[name]))
        a=arrays(cell['output_cache']['path']);stored[(cid,method)]=a
        label=labels[cid];domain=np.asarray(label['evaluation_domain'],dtype=bool).squeeze()
        target_T=np.asarray(label['temperature_color_K'],dtype=float).squeeze()
        target_u=np.asarray(label.get('template_postblur_e',label.get('template_e')),dtype=float).squeeze()
        T=a['T'].squeeze();u=a['u'].squeeze();expected=a['predicted_raw_e'].squeeze()
        raw_e=(raw_cache[cid].astype(float)-cfg['black_level_DN'])*p.adc_step
        error_T=T[domain]-target_T[domain];error_u=u[domain]-target_u[domain]
        measured={'temperature_SSE_K2':float(np.square(error_T).sum()),
            'template_SSE':float(np.square(error_u).sum()),'domain_pixels':int(domain.sum()),
            'temperature_RMSE_K':float(np.sqrt(np.square(error_T).mean())),
            'template_RMSE_e':float(np.sqrt(np.square(error_u).mean())),
            'raw_RMSE_e':float(np.sqrt(np.square(expected-raw_e).mean())),
            'raw_weighted_RMSE':float(np.sqrt(np.mean(np.square(expected-raw_e)/(np.maximum(raw_e,0)+p.extra_variance))))}
        metric_error={key:abs(measured[key]-metric[key]) for key in measured}
        stage_rows=[]
        for stage in info['stages']:
            opt=stage.get('optimizer',{})
            stage_rows.append({'configuration':stage['configuration'],
                'iterations':opt.get('iterations',stage.get('iterations')),
                'success':opt.get('success',stage.get('optimizer_success')),
                'objective':opt.get('objective',stage.get('objective')),
                'projected_gradient_inf':stage.get('gradient',{}).get('scaled_projected_linf',stage.get('scaled_projected_gradient_inf')),
                'message':opt.get('message',stage.get('message')),
                'qualified_endpoint':stage.get('qualified_endpoint'),
                'inner_normal_relative':stage.get('inner',{}).get('actual_scaled_normal_relative'),
                'tighter_inner_normal_relative':stage.get('tighter_inner_reference',{}).get('actual_scaled_normal_relative')})
        rows.append({'case_id':cid,'mother_id':cell['mother_id'],'method':method,
            'finite_output':cell['finite_output'],'production_numerical_certified':cell['production_numerical_certified'],
            'wall_s':cell['wall_s'],'recomputed_metrics':measured,'metric_absolute_differences':metric_error,
            'valid_fraction':metric['valid_fraction'],'returned_gradient':info.get('returned_gradient'),
            'returned_endpoint':info.get('returned_endpoint',info.get('returned_result_role')),
            'objective_history_summary':info.get('objective_history_summary',info.get('outer_history_summary')),
            'stages':stage_rows,'output_comparisons':info['output_comparisons'],
            'errors':info['errors'],'cache_keys':list(a),'terminal_latent_parameters_saved':any(k in a for k in ('state','z','z_grid','beta','xi')),
            'template_inner_difference_RMS_e':info.get('template_inner_difference_RMS_e')})

    aggregates={}
    for method in dict.fromkeys(c['method'] for c in cells):
        group=[r for r in rows if r['method']==method]
        count=sum(r['recomputed_metrics']['domain_pixels'] for r in group)
        T=float(np.sqrt(sum(r['recomputed_metrics']['temperature_SSE_K2'] for r in group)/count))
        u=float(np.sqrt(sum(r['recomputed_metrics']['template_SSE'] for r in group)/count))
        recorded=result['summary']['by_method'][method]
        aggregates[method]={'case_denominator':len(group),'mother_denominator':len({r['mother_id'] for r in group}),
            'scoring_pixel_denominator':count,'temperature_RMSE_K':T,'template_RMSE_e':u,
            'temperature_error_vs_summary':abs(T-recorded['complete_denominator_temperature_RMSE_K']),
            'template_error_vs_summary':abs(u-recorded['complete_denominator_template_RMSE_e'])}
    initializer_differences=[]
    for cid in raw_cache:
        a=stored[(cid,'joint_c1_s1_repaired_v2__v3')];b=stored[(cid,'joint_c1_s1_repaired_v2__smooth_demod')]
        initializer_differences.append({'case_id':cid,'temperature_RMS_K':float(np.sqrt(np.mean((a['T']-b['T'])**2))),
            'template_RMS_e':float(np.sqrt(np.mean((a['u']-b['u'])**2))),
            'predicted_raw_RMS_e':float(np.sqrt(np.mean((a['predicted_raw_e']-b['predicted_raw_e'])**2)))})

    # Three constant field directions are a small calibration-anchor algebra
    # check, not a reconstruction, minimization or terminal-Hessian estimate.
    anchor=p.M.new_zeros((1,5,*cfg['shape']));beta=p.M.new_zeros((1,3))
    anchor[:,0]=(cfg['reference_temperature_K']-p.T0)/p.Tscale;anchor[:,1]=np.log(.5)
    indices=[0,1]+[2+i for i,v in enumerate(p.active) if bool(v)]
    with torch.no_grad():
        jac=p.spectral_jacobian(anchor,beta)[0,indices,:,0,0].numpy()
        directions=[]
        for index in indices:
            direction=torch.zeros_like(anchor);direction[:,index]=1.
            directions.append(p.jvp(anchor,beta,direction,torch.zeros_like(beta)).numpy().ravel())
    J=np.stack(directions)
    object_otf=p.op.object_otf.numpy();n,mshape=cfg['shape']
    fy,fx=np.meshgrid(np.fft.fftfreq(n),np.fft.fftfreq(mshape),indexing='ij')
    lap=4*np.sin(np.pi*fx)**2+4*np.sin(np.pi*fy)**2
    relay=lambda xx,yy:np.exp(-2*np.pi**2*cfg['relay_psf_sigma_px']**2*(xx**2+yy**2))
    base=np.sum(jac*np.asarray(cfg['channel_gain'])[None,:]/2,axis=1)
    symbol=relay(fx,fy)[...,None,None]**2*np.outer(base,base)
    for band,((kx,ky),gain,depth) in enumerate(zip(cfg['carrier_bins_xy'],cfg['channel_gain'],cfg['modulation_depth'])):
        shiftx=kx/mshape;shifty=ky/n
        symbol+=(gain*depth/4)**2*(relay(fx+shiftx,fy+shifty)**2+relay(fx-shiftx,fy-shifty)**2)[...,None,None]*np.outer(jac[:,band],jac[:,band])
    symbol*=object_otf[...,None,None]**2
    strengths=np.asarray(s.read(ROOT/'reports/training_v4/run_001/backbone_selection.json')['solver_config']['regularization'])[indices]
    algebra=[]
    for cid,raw in raw_cache.items():
        raw_e=(raw.astype(float)-cfg['black_level_DN'])*p.adc_step
        W=1./(np.maximum(raw_e,0)+p.extra_variance)
        d,_=p.preconditioner_diagonal(anchor,beta,torch.as_tensor(W,dtype=p.M.dtype)[None])
        diagonal=d[0,indices].mean((-2,-1)).numpy();scale=diagonal**-.5
        exact_gram=(J*W.ravel()[None])@J.T/raw.size
        scaled_gram=exact_gram*scale[:,None]*scale[None,:]
        eigen=np.linalg.eigvalsh(scaled_gram)
        H=W.mean()*symbol+lap[...,None,None]*np.diag(strengths)
        scaled_symbol=H*scale[None,None,:,None]*scale[None,None,None,:]
        spectrum=np.linalg.eigvalsh(scaled_symbol)
        algebra.append({'case_id':cid,'active_parameter_indices':indices,
            'native_normalized_parameter_units':['(latent_T_K-3000)/1000','log_amplitude','signed_species_e/60'],
            'reference_diagonal':diagonal.tolist(),'current_coordinate_scales':scale.tolist(),
            'exact_constant_mode_scaled_Gram':scaled_gram.tolist(),
            'exact_constant_mode_scaled_eigenvalues':eigen.tolist(),
            'exact_constant_mode_scaled_condition_number':float(eigen[-1]/eigen[0]),
            'temperature_amplitude_column_correlation':float(exact_gram[0,1]/np.sqrt(exact_gram[0,0]*exact_gram[1,1])),
            'stationary_anchor_symbol_scaled_eigenvalue_min':float(spectrum.min()),
            'stationary_anchor_symbol_scaled_eigenvalue_max':float(spectrum.max()),
            'stationary_anchor_symbol_scaled_spectral_span':float(spectrum.max()/spectrum.min()),
            'symbol_scope':'constant calibration Jacobian, mean raw weight, frequency-diagonal approximation; not measured terminal Hessian'})
    checksum=all(v['matches'] for v in source_checks.values()) and all(v['matches'] for v in inputs+file_checks)
    numerical_rows=[r for r in rows if r['method'].startswith('joint_')]
    last_deltas=[r['output_comparisons'][-1] for r in numerical_rows]
    payload={'schema':'quick-v001-independent-numerical-scientific-review-v1','time':s.now(),
        'scope':{'read_only_existing_sources_and_v001':True,'registered_sources_modified':False,
            'formal_inverse_calls':0,'optimizer_runs':0,'formal_MC_reruns':0,'GPU_used':False,
            'CPU_calibration_JVP_calls':len(indices),'final_payloads_opened':False,
            'diagnostic_development_labels_read':True},
        'input_integrity':{'all_checks_passed':checksum,'registration_sha256':registration_hash,
            'registration_matches_result':registration_hash==result['registration_sha256'],
            'result_path':str(folder/'QUICK_RESULT.json'),'result_sha256':sha(folder/'QUICK_RESULT.json'),
            'registration_sidecar_sha256':sha(folder/'QUICK_REGISTRATION_SHA256.json'),
            'source_checks':source_checks,'input_and_result_file_checks':inputs+file_checks},
        'denominator_audit':{'planned_cells':result['summary']['planned_cells'],'returned_cells':len(cells),
            'unique_method_case_pairs':len({(r['method'],r['case_id']) for r in rows}),
            'independent_development_mothers':len(raw_cache),'finite_scored_cells':len(rows),
            'numerically_certified_cells':sum(r['production_numerical_certified'] for r in rows),
            'all_cells_full_truth_domain':all(r['recomputed_metrics']['domain_pixels']==65536 for r in rows),
            'all_metric_recomputations_match':all(max(r['metric_absolute_differences'].values())<1e-7 for r in rows),
            'method_aggregates':aggregates,'cells':rows},
        'profile_observations':{'classification':'observed facts','last_tightening_temperature_delta_max_K':max(d['color_RMS_difference_K'] for d in last_deltas),
            'last_tightening_template_delta_max_e':max(d['template_RMS_difference_e'] for d in last_deltas),
            'final_projected_gradient_range':[min(r['returned_gradient']['scaled_projected_linf'] for r in numerical_rows),max(r['returned_gradient']['scaled_projected_linf'] for r in numerical_rows)],
            'final_requested_gradient_tolerance':1e-7,'initializer_output_differences':initializer_differences,
            'all_last_endpoints_fail_registered_stationarity':all(not r['stages'][-1]['qualified_endpoint'] for r in numerical_rows)},
        'units_and_criteria_review':[
            {'classification':'fact','finding':'The color budget0.2 K and template budget0.17 e apply to RMS differences of postblur outputs between tighter solves. They are neither scientific RMSE limits nor optimizer gradient units.'},
            {'classification':'fact','finding':'The primary temperature RMSE is postblur continuum color temperature against full truth-domain labels. It is not latent thermodynamic temperature accuracy; latent comparisons are separately named.'},
            {'classification':'fact','finding':'The reported physical gradient is with respect to native normalized latent coordinates (T/1000, log amplitude, species/60, beta), not a derivative per K or per electron. Scaled projected gradients use their route-specific numerical coordinates; cross-route gradient magnitudes are not directly comparable.'},
            {'classification':'fact','finding':'The raw weighted RMSE uses observed raw-based Poisson/read/ADC variance. Values near1 are residual diagnostics, not numerical convergence or scientific identification certificates, and no fitted-degree-of-freedom correction is applied.'},
            {'classification':'fact','finding':'The four joint profiles pass all endpoint inner references but fail stationarity after stage1. Optimizer ftol success is explicitly insufficient. No threshold relaxation is needed to explain the failed certification.'},
            {'classification':'fact','finding':'Both v3 refinements hit400,600,800 iteration limits, then time out in the next stage. The last cached output is objective.last, which may be an unaccepted line-search evaluation. Recorded interstage deltas describe completed stages, not a terminal convergence test.'}],
        'C1_assessment':{
            'classification':'bounded inference; not a proved root cause',
            'facts':['The frozen continuum LUT is piecewise linear with5 K spacing; its derivative changes at nodes.',
                'v001 caches only T,u,C,predicted_raw_e,valid,low_signal, and omit terminal latent state, beta and profile linear coordinates.',
                'The four profiles have a small output plateau with a persistent projected-gradient floor; v3 has much larger output drift and ongoing objective decreases.'],
            'inference':'A consistent C1 calibration adapter is a plausible test of the profile gradient floor. Its success is not established, and it does not remove the spatial/cross-parameter conditioning evident in the v3 formulation.',
            'not_identifiable_from_current_cache':['latent temperatures distance to5 K knots','identity of the coordinate attaining terminal projected-gradient maximum','one-sided terminal derivative jump','terminal Hessian spectrum'],
            'target_preservation':'A C1 interpolant preserves LUT node values but generally changes between-node forward values. It must be versioned and bounded as a calibration approximation; it is not literally the unchanged piecewise-linear mathematical objective.',
            'adequacy_conclusion':'C1 alone cannot be promised sufficient for the two v3 cases. Require actual tighter-output and unchanged stationarity gates under the new version; retain v001 as an uncertified negative result.'},
        'CPU_conditioning_evidence':{'classification':'calibration-anchor algebra facts; terminal behavior remains an inference',
            'anchor_temperature_K':cfg['reference_temperature_K'],'anchor_amplitude':.5,
            'object_OTF_zero_frequency_fraction':float(np.mean(object_otf==0)),
            'native_parameter_count_v3':len(indices)*65536+3,
            'native_nonlinear_parameter_count_profile':2*65536,
            'original_active_prior_coefficients':strengths.tolist(),'cases':algebra},
        'single_preconditioner_recommendation':{
            'classification':'specific implementation proposal; not executed or validated as a solver',
            'name':'coupled Fourier-block preconditioner for free-variable Gauss-Newton steps',
            'unchanged_physical_objective':'0.5*sum(W*(forward(state,beta)-raw_e)^2)+original periodic nearest-neighbor latent prior; W and prior coefficients remain frozen',
            'unchanged_bounds':'latent T within1200..4500 K, loga within[-4,0.3], active signed species and beta within[-20,20]',
            'step_system':'On the free variables solve (J^T W J + Lambda*L + mu*D) d = -g with actual p.jvp/p.vjp; mu is numerical damping in the step only and is absent from the acceptance objective.',
            'preconditioner':'For the3 active fields use per-frequency3x3 SPD blocks P(k)=G_anchor(k)+diag(lambda)*L(k)+mu*D. G_anchor includes object_OTF(k)^2 and relay-weighted DC and +/-carrier sidebands, retaining T/loga/species cross terms. Invert these tiny blocks by Cholesky/eigendecomposition and apply with FFT. Use an exact3x3 raw-weighted Gram block for beta.',
            'boundary_handling':'Mask bound-active components using the same physical projected-gradient convention; restrict the SPD preconditioner to free variables. Evaluate line-search/trust-region acceptance on the original bounded objective, and retain only accepted states for timeout output.',
            'why_specific':'The existing v3 route only rescales each field by one calibration-anchor diagonal number. It does not invert spectral column correlations or the object-OTF/spatial-prior spectrum. The proxy spectrum recorded above measures the unresolved calibration-anchor spread.',
            'numerical_receipts':'Require recomputed true residual of each step system, accepted objective decrease, unchanged reference-coordinate projected gradient, two successive0.2 K/0.17 e output plateaus, and full parameter/cache receipts; failed linear solves do not become certificates.',
            'no_new_physical_regularization':True,'no_hyperparameter_grid':True},
        'next_bounded_gate':{'one_route_first':'Use the same positive-normal development raw for a paired numerical diagnostic before expanding the panel.',
            'maximum_complete_inversions':2,'maximum_inverse_wall_s_per_call':180,
            'order':'First inspect the C1 adapter CPU derivative/calibration contract; then run one versioned profile and one v3 refinement only if registered by root.',
            'stop':'If v3 still has a large true step residual or multi-K continuation drift at the fixed cap, stop that route and implement/test the numerical preconditioner rather than adding iterations or weakening gtol.',
            'full_parameters_needed':['latent state or z_grid','beta','xi for the profile','accepted-state identifier','per-channel projected-gradient maxima and argmax','latent knot distances and one-sided derivative probes']},
        'interpretation':'These are2 development mothers and6 case-method cells. No independent scientific acceptance, FPR, coverage, global-optimum or hardware-throughput conclusion follows.',
        'source_sha256':{name:sha(ROOT/name) for name in supporting_sources},
        'review_script_sha256':sha(__file__),'CPU_review_wall_s':time.perf_counter()-started}
    target=Path(__file__).with_name('NUMERICAL_SCIENTIFIC_REVIEW.json')
    s.write(target,s.plain(payload))
    print(json.dumps({'path':str(target),'sha256':sha(target),'integrity_passed':checksum,
        'metric_recomputations_match':payload['denominator_audit']['all_metric_recomputations_match'],
        'conditioning':algebra,'wall_s':time.perf_counter()-started},indent=2),flush=True)


if __name__=='__main__': run()
