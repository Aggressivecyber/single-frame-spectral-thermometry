"""Independent, read-only qv003 certificate and two-endpoint CPU review.

This reviewer does not invoke an initializer, outer optimizer, complete inverse,
GPU or truth loader. Only two explicitly chosen final nonlinear states receive
CPU physics evaluation. Conditional linear checks keep those states fixed.
"""
from . import storage as s
from pathlib import Path
import copy
import hashlib
import json
import os
import time
import traceback
import numpy as np

RUN=Path('D:/CodexWork/thermometry/closeout_001/reports/repair_001/quick_v003')
BASE=s.REPORTS/'repair_001/quick_v003_analysis'
HERE=BASE/'attempt_002'
SOURCE='research_closeout/quick_v003_numerical_review.py'
VERSION='qv003-independent-numerical-review-v001'
SELECTED=(('v5_development_selection_mother_0000_blank_noise0','v3'),
          ('v5_development_selection_mother_0005_positive_weak_noise0','v3'))
TOUCHED={}
COUNTERS={'formal_inverse_calls':0,'outer_optimizer_calls':0,'GPU_calls':0,
          'fixed_nonlinear_endpoints_evaluated':0,'conditional_linear_solves':0,
          'fixed_parameter_gradient_evaluations':0}


def checked(path,digest=None):
    path=Path(path).resolve();actual=s.sha(path)
    if digest is not None and actual!=digest: raise ValueError('File SHA mismatch: '+str(path))
    if str(path) in TOUCHED and TOUCHED[str(path)]!=actual: raise ValueError('File changed while reviewing: '+str(path))
    TOUCHED[str(path)]=actual
    return actual


def ref(path):
    return {'path':str(Path(path).resolve()),'sha256':checked(path),'bytes':Path(path).stat().st_size}


def sealed(path):
    path=Path(path).resolve();side=path.with_name(path.name+'.sha256.json')
    checked(side);receipt=s.read(side)
    if Path(receipt['path']).resolve()!=path: raise ValueError('Misbound SHA sidecar')
    checked(path,receipt['sha256'])
    return s.read(path)


def checked_ref(value): return checked(value['path'],value['sha256'])


def array_sha(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str(value.dtype).encode()+json.dumps(list(value.shape)).encode())
    h.update(value.tobytes());return h.hexdigest()


def descriptor(value):
    return {'dtype':str(value.dtype),'shape':list(value.shape),'nbytes':value.nbytes,'sha256':array_sha(value)}


def read_parameters(receipt,prepared):
    checked_ref(receipt['parameter_cache']);checked_ref(receipt['prepared'])
    with np.load(receipt['parameter_cache']['path'],allow_pickle=False) as handle:
        values={name:handle[name].copy() for name in handle.files}
    if set(values)!={'z_grid','xi','beta'}: raise ValueError('Parameter set differs')
    if {name:descriptor(a) for name,a in values.items()}!=prepared['parameter_arrays']:
        raise ValueError('Parameter dtype/shape/byte descriptors differ')
    if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in values.values()):
        raise ValueError('Nonfinite or nonfloat64 parameter')
    if values['z_grid'].shape!=(1,2,256,256) or values['beta'].shape!=(1,3):
        raise ValueError('Non-native parameter geometry')
    return values


def stage_contract(core):
    opt=core['numerical_options'];stages=core['stages'];records=[];qualified={}
    for index,stage in enumerate(stages):
        config=stage['configuration'];gradient=stage['gradient']
        tighter=index==0 or all(config[k]<stages[index-1]['configuration'][k] for k in ('gtol','ftol'))
        logical=bool(stage['complete'] and stage['optimizer']['success'] and gradient['finite']
            and gradient['scaled_projected_linf']<=config['gtol']
            and stage['inner']['numerical_certified'] and stage['tighter_inner_reference']['numerical_certified']
            and stage['inner']['actual_scaled_normal_relative']<=1e-8
            and stage['tighter_inner_reference']['actual_scaled_normal_relative']<=1e-8
            and stage['template_inner_difference_RMS_e']<=opt['numeric_template_budget_e']
            and stage['finite_output'] and stage['nonnegative_predicted_measurement'])
        passed=logical and tighter and stage['qualified_endpoint'] and stage['requested_projected_gradient_passed']
        passed=passed and config==opt['stages'][index]
        qualified[stage['stage']]=passed
        records.append({'stage':stage['stage'],'configuration':config,'optimizer':stage['optimizer'],
            'gradient':gradient,'inner':stage['inner'],'tighter_inner_reference':stage['tighter_inner_reference'],
            'template_inner_difference_RMS_e':stage['template_inner_difference_RMS_e'],
            'wall_s':stage['wall_s'],'passed':bool(passed),
            'saved_parameter_state_available':stage['stage']==core['returned_stage'],
            'evidence':'SHA-bound original log; only final parameters saved'})
    consecutive=0;platforms=[]
    for comp in core['output_comparisons']:
        within=(comp['template_RMS_difference_e']<=opt['numeric_template_budget_e']
                and comp['color_RMS_difference_K']<=opt['numeric_color_budget_K'])
        both=bool(qualified.get(comp['from_stage']) and qualified.get(comp['to_stage']))
        logical=within and both and comp['to_stage']==comp['from_stage']+1
        consecutive=consecutive+1 if logical else 0
        flags=(comp['within_output_budgets']==within and comp['both_endpoints_qualified']==both
            and comp['passed']==logical and comp['consecutive_passed_comparisons']==consecutive)
        platforms.append({**comp,'passed_independent_log_check':bool(flags),
            'output_arrays_saved_for_replay':False})
    passed=bool(len(stages)>=3 and len(platforms)>=2 and all(qualified.values())
        and consecutive>=2 and consecutive==core['observed_consecutive_stable_comparisons']
        and all(v['passed_independent_log_check'] for v in platforms)
        and not core['errors'] and not core['timed_out'] and core['numerical_certified']
        and core['returned_stage']==stages[-1]['stage']
        and core['returned_result_role']=='tightest_certified_endpoint'
        and core['returned_gradient']==stages[-1]['gradient']
        and core['returned_endpoint_objective']==stages[-1]['returned_endpoint_objective']
        and opt['cost_limit_s']==180. and opt['numeric_template_budget_e']==.17
        and opt['numeric_color_budget_K']==.2 and opt['required_stable_comparisons']==2)
    return {'passed':passed,'executed_final_gtol':stages[-1]['configuration']['gtol'],
        'stage4_1e_minus7_executed':any(v['configuration']['gtol']<=1e-7 for v in stages),
        'stages':records,'platforms':platforms,'earlier_stage_parameters_not_saved':len(stages)-1,
        'platform_replay_independently_performed':False}


def raw_array(reference):
    path=s.LEGACY/reference['raw_reference']['path'];checked(path,reference['raw_reference']['sha256'])
    with np.load(path,allow_pickle=False) as handle: raw=handle['raw_dn'].copy()
    expected=reference['raw_reference']
    if descriptor(raw)['sha256']!=expected['array_sha256'] or str(raw.dtype)!=expected['dtype'] or list(raw.shape)!=expected['shape']:
        raise ValueError('Camera raw array identity differs')
    return raw


def metadata_review():
    registration=sealed(RUN/'PANEL_REGISTRATION.json');result=sealed(RUN/'PANEL_RESULT.json')
    digest=checked(RUN/'PANEL_REGISTRATION.json')
    if result['registration_sha256']!=digest: raise ValueError('Result registration mismatch')
    sources=registration['source_sha256']
    for name,value in sources.items(): checked(s.ROOT/name,value)
    for dependency in registration['read_dependencies']: checked_ref(dependency)
    qualification=sealed(RUN/'NATIVE_COMPACT_QUALIFICATION.json')
    if not qualification['all_passed'] or qualification['registration_sha256']!=digest:
        raise ValueError('Separate native qualification not passed')
    if qualification['passed_existing_endpoints']!=4: raise ValueError('Qualification denominator differs')
    rows=result['summary']['cells'];planned=registration['planned_cells']
    if len(rows)!=12 or {(c['case_id'],c['method']) for c in rows}!={(c['case_id'],c['method']) for c in planned}:
        raise ValueError('Full 12-cell denominator differs')
    new=[c for c in rows if c['execution_role']=='NEW_INVERSE'];reused=[c for c in rows if c['execution_role']=='REUSE_V002']
    if len(new)!=8 or len(reused)!=4: raise ValueError('Reuse/new denominator differs')
    old=s.read(s.REPORTS/'repair_001/quick_v002/QUICK_RESULT.json')
    checked(s.REPORTS/'repair_001/quick_v002/QUICK_RESULT.json')
    origin={(c['case_id'],c['method']):c for c in old['summary']['cells']}
    reuse_checks=[{'case_id':c['case_id'],'initializer':c['initializer'],
        'origin_row_preserved':c['origin_cell']==origin[(c['case_id'],c['method'])],
        'zero_new_inverse_calls':c['new_inverse_calls_panel_total']==0} for c in reused]
    references={r['case_id']:r for r in registration['references']};raws={};provenance_checks=[]
    for case,reference in references.items():
        if reference['cohort']!='development' or reference['partition_role']!='development_selection':
            raise ValueError('Reviewer only opens registered development raws')
        raw=raw_array(reference);raws[case]=raw
        provenance=registration['v3_initializer_provenance'][case]
        for name in ('state_cache','receipt_journal'): checked_ref(provenance[name])
        journal=list(s.rows(provenance['receipt_journal']['path']))
        line=journal[provenance['receipt_journal']['line']-1]
        dtype={'torch.float32':np.float32,'torch.float64':np.float64}[provenance['source_input_dtype']]
        passed=bool(array_sha(raw.astype(dtype))==provenance['inference_input_array_sha256']
            and array_sha(raw)==provenance['raw_array_sha256']
            and reference['raw_reference']['sha256']==provenance['raw_file_sha256']==line['raw_sha256']
            and line['case_id']==case and line['source']=='C_v3'
            and line['source_metadata']==provenance['source_metadata']
            and not provenance['truth_or_identity_used'])
        provenance_checks.append({'case_id':case,'passed':passed,'source_input_dtype':provenance['source_input_dtype']})
    cells=[];payloads={}
    for cell in new:
        report=Path(cell['compact_report_root']);records={}
        for path in report.glob('*.json'):
            if not path.name.endswith('.sha256.json'): records[path.name]=sealed(path)
        prepared=records['PREPARED.json'];parameters=records['PARAMETERS.json'];started=records['STARTED.json']
        final=records['RESULT.json'];score=records['SCORE_panel_external_v001.json'];context=prepared['context']
        fingerprint=s.canonical(context);key=s.canonical({'identity':context['identity'],
            'raw_sha256':context['input_reference']['sha256'],'calibration':context['calibration']['fingerprint'],
            'config':context['config']})
        if context!=started['context'] or prepared['context_fingerprint']!=fingerprint:
            raise ValueError('Prepared context differs')
        for record in records.values():
            if record['context_fingerprint']!=fingerprint or record['cache_key']!=key:
                raise ValueError('Cache identity/context mismatch')
        if key!=report.name or key!=Path(cell['compact_data_root']).name: raise ValueError('Cache path key mismatch')
        identity=context['identity']
        for field in ('method','case_id','raw_sha256'):
            if identity[field]!=cell[field]: raise ValueError('Identity mismatch '+field)
        if identity['registration_sha256']!=digest or context['calibration']!=registration['calibration_contract']:
            raise ValueError('Cache calibration or registration differs')
        for name,value in context['source_sha256'].items(): checked(s.ROOT/name,value)
        if any(context['source_sha256'].get(name)!=value for name,value in sources.items()):
            raise ValueError('Cache omits registered execution source')
        values=read_parameters(parameters,prepared)
        if score['score']['metrics']!=cell['metrics'] or score['score']!=cell['external_score']:
            raise ValueError('External score differs from bound receipt')
        if final['verification']['reconstructed_canonical_arrays']!=prepared['canonical_arrays']:
            raise ValueError('Original/reconstructed canonical descriptors disagree')
        if final['verification']['reconstructed_continuum_latent']!=prepared['continuum_latent']:
            raise ValueError('Continuum latent identity differs')
        if not final['verification']['all_byte_identical'] or not all(final['verification']['per_array_byte_identical'].values()):
            raise ValueError('Original GPU byte verification did not pass')
        if final['status']!='LOSSLESS_READY' or not prepared['original_numerical_valid'] or not prepared['original_finite_canonical']:
            raise ValueError('Original endpoint is not finite/certified')
        info=prepared['original_info'];core=info['numerical_engine'];contract=stage_contract(core)
        if info['anchor_regscale_contract']!=core['physical_objective']:
            raise ValueError('Shared anchor contract differs between wrapper/core')
        if info['physical_objective_version']!=registration['calibration_contract']['physical_objective_version']:
            raise ValueError('Effective C1 physical objective version differs')
        if info['objective_fingerprint']!=registration['objective_fingerprint'] or info['config_fingerprint']!=registration['config_fingerprint']:
            raise ValueError('Frozen objective/controller fingerprint differs')
        metrics=cell['metrics'];denominator=bool(metrics['domain_pixels']==metrics['raw_pixels']==65536
            and metrics['accepted_pixels']==65536 and metrics['nonfinite_domain_pixels']==metrics['nonfinite_output_pixels']==0
            and metrics['temperature_reference_color_comparison_pixels']==metrics['temperature_latent_truth_comparison_pixels']==65536)
        row={'case_id':cell['case_id'],'initializer':cell['initializer'],'condition':cell['condition'],
            'passed':bool(contract['passed'] and denominator),'certificate':contract,'parameter_cache':parameters,
            'parameter_descriptors':prepared['parameter_arrays'],'metadata_sha256':{name:checked(report/name) for name in records},
            'core_wall_s':core['wall_s'],'sum_stage_wall_s':sum(v['wall_s'] for v in core['stages']),
            'objective_evaluations':core['objective_evaluations'],'returned_objective':core['returned_endpoint_objective'],
            'final_gradient':core['returned_gradient'],'shared_anchor_contract':core['physical_objective'],
            'canonical_hashes_bound':True,'canonical_forward_CPU_reconstruction_performed':False,
            'full_65536_domain_verified':denominator,'metrics':metrics,
            'reservation_release_warning':cell['compact_result'].get('reservation_release_error'),
            'canonical_arrays_stored':False,'GPU_exact_byte_reconstruction_evidence':'original frozen cache receipt; independent root GPU recheck is separate'}
        cells.append(row);payloads[(cell['case_id'],cell['initializer'])]=(cell,prepared,values,raws[cell['case_id']])
    cfg=s.read(s.ROOT/'reports/training_v4/run_001/backbone_selection.json')['physics_config']
    pairs=[]
    for case in sorted({c['case_id'] for c in new}):
        a,ap,av,_=payloads[(case,'v3')];b,bp,bv,_=payloads[(case,'smooth_demod')]
        delta=av['z_grid']-bv['z_grid'];anchors=ap['original_info']['anchor_regscale_contract']==bp['original_info']['anchor_regscale_contract']
        pairs.append({'case_id':case,'pixel_denominator':65536,'same_physical_anchor_and_objective':anchors,
            'latent_temperature_pair_RMS_K':float(cfg['temperature_scale_K']*np.sqrt(np.mean(delta[:,0]**2))),
            'latent_loga_pair_RMS':float(np.sqrt(np.mean(delta[:,1]**2))),
            'postblur_template_pair_RMS_e_from_orthonormal_coefficients':float(np.linalg.norm(av['xi']-bv['xi'])/256),
            'template_pair_RMS_derivation':'Parseval in frozen real orthonormal Fourier basis; ||xi_a-xi_b||_2/sqrt(65536); no image forward reconstruction',
            'beta_pair_maximum_absolute_difference':float(np.max(np.abs(av['beta']-bv['beta']))),
            'final_objective_difference':ap['original_info']['numerical_engine']['returned_endpoint_objective']-bp['original_info']['numerical_engine']['returned_endpoint_objective'],
            'reported_color_RMSE_metric_absolute_difference_K':abs(a['metrics']['temperature_RMSE_K']-b['metrics']['temperature_RMSE_K']),
            'color_output_pair_RMS_not_reconstructed_here':True})
    ledger=s.REPORTS/'inverse_calls.jsonl';checked(ledger)
    calls=[r for r in s.rows(ledger) if r.get('phase')=='C1v3_small_panel_v003']
    calls_ok=len(calls)==8 and len({(r['method'],r['case_id']) for r in calls})==8
    sseu=sum(c['metrics']['template_SSE'] for c in new);sset=sum(c['metrics']['temperature_SSE_K2'] for c in new)
    passed=bool(all(c['passed'] for c in cells) and all(c['origin_row_preserved'] and c['zero_new_inverse_calls'] for c in reuse_checks)
        and all(c['passed'] for c in provenance_checks) and all(c['same_physical_anchor_and_objective'] for c in pairs) and calls_ok)
    metadata={'schema':VERSION,'time':s.now(),'all_metadata_checks_passed':passed,
        'denominators':{'panel_cells':12,'reused_v002_cells':4,'new_cells':8,'new_raws':4,'panel_raws':6,
            'new_metric_cell_pixel_total':8*65536,'panel_metric_cell_pixel_total':12*65536,
            'per_cell_FOV_pixels':65536,'independent_scientific_sample_size_claimed':False},
        'registered_sources':len(sources),'registered_read_dependencies':len(registration['read_dependencies']),
        'registration':ref(RUN/'PANEL_REGISTRATION.json'),'result':ref(RUN/'PANEL_RESULT.json'),
        'physical_objective_version':registration['calibration_contract']['physical_objective_version'],
        'calibration_version':registration['calibration_contract']['calibration_version'],
        'reused_v002_origin_checks':reuse_checks,'initializer_provenance_checks':provenance_checks,
        'cells':cells,'cross_initializer_pairs':pairs,'actual_new_call_ledger':calls,'call_ledger_check_passed':calls_ok,
        'recorded_compute':{'new_eight_core_wall_s':sum(c['core_wall_s'] for c in cells),
            'new_eight_stage_wall_s':sum(c['sum_stage_wall_s'] for c in cells),
            'new_eight_objective_evaluations':sum(c['objective_evaluations'] for c in cells),
            'maximum_per_cell_s':180,'declared_eight_cell_budget_s_excluding_overhead':1440,
            'existing_v3_initializer_cost_excluded':True,'GPU_forward_cache_scoring_and_controller_overhead_excluded':True},
        'physical_metrics_without_acceptance_claim':{'new_eight_pooled_template_RMSE_e':float(np.sqrt(sseu/(8*65536))),
            'new_eight_pooled_color_temperature_RMSE_K':float(np.sqrt(sset/(8*65536))),
            'detector_threshold_attached':False,'scientific_acceptance_evaluated':False},
        'limits':['Only final nonlinear parameters saved; early stages and two platforms have SHA-bound log evidence, not independent replay.',
            'Exact native-GPU compact-output reconstruction is audited separately by root; CPU endpoints below use numerical comparisons and never rewrite GPU byte contracts.',
            'All8 original cache results retain a reservation-release warning; no cache/source/result is repaired by this audit.'],
        'formal_inverse_calls':0,'GPU_used':False,'truth_payloads_opened':False}
    return metadata,registration,payloads


def gradient_record(gradient,z,objective):
    scale=objective.coordinate_scale;flat=z.detach().numpy().ravel();n=objective.g**2
    limits=objective.p.cfg['temperature_range_K']
    lower=np.r_[np.full(n,(limits[0]-objective.p.T0)/objective.p.Tscale),np.full(n,-4.)]/scale
    upper=np.r_[np.full(n,(limits[1]-objective.p.T0)/objective.p.Tscale),np.full(n,.3)]/scale
    v=flat/scale;projected=v-np.clip(v-gradient*scale,lower,upper)
    return {'finite':bool(np.isfinite(gradient).all()),'physical_l2':float(np.linalg.norm(gradient)),
        'physical_linf':float(np.abs(gradient).max()),'scaled_projected_linf':float(np.abs(projected).max())}


def endpoint(objective,z0,x):
    import torch
    from . import measurement as m
    COUNTERS['fixed_parameter_gradient_evaluations']+=1
    objective.check_deadline();z=z0.detach().clone().requires_grad_(True);p=objective.p
    C=m.continuum(p,z,grid=256);rawC=p.op(C);residual=rawC+objective.linear.apply(x)[None]-objective.raw_e
    data=.5*(objective.W*residual.square()).sum();linear=.5*(objective.linear.reg*x).square().sum()
    dx=z[:,:,:,1:]-z[:,:,:,:-1];dy=z[:,:,1:]-z[:,:,:-1]
    prior=.5*(objective.regscale*(dx.square().sum((0,2,3))+dy.square().sum((0,2,3)))).sum()
    phi=data+linear+prior;ad=torch.autograd.grad(phi,z)[0].detach()
    with torch.no_grad():
        prior_g=torch.zeros_like(z);wx=dx.detach()*objective.regscale[None,:,None,None];wy=dy.detach()*objective.regscale[None,:,None,None]
        prior_g[:,:,:,:-1]-=wx;prior_g[:,:,:,1:]+=wx;prior_g[:,:,:-1]-=wy;prior_g[:,:,1:]+=wy
        back=p.op.adjoint(objective.W*residual.detach());T=p.T0+p.Tscale*z[:,0].detach();_,dC=p.continuum(T)
        explicit=torch.stack([(back*z[:,1:2].detach().exp()*dC*p.Tscale).sum(1),(back*C.detach()).sum(1)],1)+prior_g
    g=ad.numpy().ravel()
    return {'objective':float(phi.detach()),'data_objective':float(data.detach()),
        'linear_regularizer_objective':float(linear.detach()),'continuum_regularizer_objective':float(prior.detach()),
        'gradient':gradient_record(g,z0,objective),'AD_vs_explicit_maximum_absolute_error':float((ad-explicit).abs().max()),
        'AD_vs_explicit_normalized_l2_error':float((ad-explicit).norm()/ad.norm().clamp_min(1.)),
        'C':C.detach(),'rawC':rawC.detach(),'residual':residual.detach()}


def conditional_normal(objective,C,x):
    import torch
    target=objective.raw_e-objective.p.op(C);linear=objective.linear
    b=torch.cat([(linear.sqrtW*target.reshape(256,256)).flatten(),linear.reg.new_zeros(linear.n)])
    r=b-linear.aug(x/linear.column_scale);normal=linear.aug_adjoint(r)
    return float(normal.norm()/linear.aug_adjoint(b).norm().clamp_min(1e-30))


def stripped(point): return {key:value for key,value in point.items() if key not in ('C','rawC','residual')}


def machine_contract_check(actual,expected,path=''):
    """Check cross-device reductions before installing the recorded anchor.

    This bound concerns float64 reduction equality, never gtol or an output
    acceptance threshold. The actual fixed anchor and coordinate scales used
    below are copied exactly from the original per-raw certificate.
    """
    differences=[]
    if isinstance(expected,dict):
        if set(actual)!=set(expected): raise ValueError('Contract structure differs: '+path)
        for key,value in expected.items():
            if key=='regularizer_fingerprint': continue
            differences.extend(machine_contract_check(actual[key],value,path+'/'+key))
    elif isinstance(expected,list):
        if len(actual)!=len(expected): raise ValueError('Contract list length differs: '+path)
        for index,value in enumerate(expected):
            differences.extend(machine_contract_check(actual[index],value,path+'/'+str(index)))
    elif isinstance(expected,float):
        if not np.isclose(actual,expected,rtol=5e-13,atol=1e-15):
            raise ValueError('Contract differs beyond float64 reduction roundoff: '+path+' '+repr((actual,expected)))
        if actual!=expected: differences.append({'field':path,'CPU':actual,'original_GPU':expected,'absolute_difference':actual-expected})
    elif actual!=expected: raise ValueError('Contract semantic value differs: '+path)
    return differences


def cpu_endpoint_review(payload,registration,p):
    import torch
    from . import measurement as m
    from .numerical_strategy import RepairedObjective
    cell,prepared,values,raw_array_value=payload;core=prepared['original_info']['numerical_engine'];started=time.perf_counter()
    z=torch.as_tensor(values['z_grid']);x=torch.cat([torch.as_tensor(values['xi']),torch.as_tensor(values['beta']).ravel()])
    raw=m.raw_tensor(raw_array_value,p);raw_e=p.raw_to_electrons(raw)
    objective=RepairedObjective(p,raw_e,z,physical_options=registration['config']['physical_options'],
        numerical_options=core['numerical_options'],deadline=time.perf_counter()+60.)
    roundoff=machine_contract_check(objective.physical_contract,core['physical_objective'],'physical')
    roundoff.extend(machine_contract_check(objective.coordinate_contract,core['numerical_coordinates'],'coordinates'))
    original_anchor=core['physical_objective']['continuum_regscale']
    objective.regscale=torch.tensor(original_anchor,dtype=torch.float64)
    objective.physical_contract=copy.deepcopy(core['physical_objective'])
    original_scale=torch.tensor(core['numerical_coordinates']['channel_scale'],dtype=torch.float64)
    objective.coordinate_scale=original_scale[None,:,None,None].expand(objective.shape).numpy().ravel()
    objective.coordinate_contract=copy.deepcopy(core['numerical_coordinates'])
    COUNTERS['fixed_nonlinear_endpoints_evaluated']+=1
    saved=endpoint(objective,z,x);gtol=core['stages'][-1]['configuration']['gtol']
    output=m.output(p,raw_e,saved['C'],objective.linear.U.unpack(x[:-3])[None],x[-3:][None],{})
    aggregate={'raw_weighted_SSE':float((objective.W*saved['residual'].square()).sum()),
        'raw_RMSE_e':float(saved['residual'].square().mean().sqrt()),'valid_pixels':int(output['valid'].sum()),
        'low_signal_pixels':int(output['low_signal'].sum()),'domain_pixels':65536}
    logged=core['returned_gradient'];normal=conditional_normal(objective,saved['C'],x)
    references=[]
    for name,start in (('warm_saved_linear_state',x),('cold_zero_linear_state',None)):
        COUNTERS['conditional_linear_solves']+=1
        refined,info=objective.linear.solve(raw_e-p.op(saved['C']),tolerance=1e-13,maxiter=1200,x0=start)
        point=endpoint(objective,z,refined)
        difference={'template_RMS_e':float(objective.linear.U.unpack((refined-x)[:-3]).square().mean().sqrt()),
            'beta_maximum_absolute':float((refined[-3:]-x[-3:]).abs().max()),
            'predicted_raw_RMS_e':float((objective.linear.apply(refined-x)).square().mean().sqrt()),
            'objective':point['objective']-saved['objective']}
        passed=bool(info['numerical_certified'] and point['gradient']['scaled_projected_linf']<=gtol
            and difference['template_RMS_e']<=.17 and point['AD_vs_explicit_normalized_l2_error']<1e-10)
        references.append({'start':name,'passed':passed,'linear_info':info,'fixed_endpoint':stripped(point),
            'difference_from_saved_linear_solution':difference,
            'actual_normal_relative_independently_recomputed':conditional_normal(objective,point['C'],refined)})
    T=(p.T0+p.Tscale*z[:,0]).numpy();distance=np.abs(T-(np.rint((T-float(p.temperatures[0]))/5)*5+float(p.temperatures[0])))
    objective_delta=saved['objective']-core['returned_endpoint_objective']
    gradient_delta={name:saved['gradient'][name]-logged[name] for name in ('physical_l2','physical_linf','scaled_projected_linf')}
    raw_metric_delta={name:aggregate[name]-cell['metrics'][name] for name in ('raw_weighted_SSE','raw_RMSE_e')}
    passed=bool(saved['gradient']['scaled_projected_linf']<=gtol and abs(objective_delta)<1e-7
        and abs(gradient_delta['scaled_projected_linf'])<1e-9 and normal<=1e-8
        and abs(raw_metric_delta['raw_weighted_SSE'])<1e-6 and abs(raw_metric_delta['raw_RMSE_e'])<1e-8
        and aggregate['valid_pixels']==65536 and all(r['passed'] for r in references))
    return {'case_id':cell['case_id'],'initializer':cell['initializer'],'passed':passed,'device':'cpu','native_shape':[256,256],
        'fixed_nonlinear_parameters_unchanged':array_sha(z.numpy())==prepared['parameter_arrays']['z_grid']['sha256'],
        'original_shared_anchor_regscale_used_exactly':objective.regscale.tolist()==original_anchor,
        'original_coordinate_scale_used_exactly':True,'CPU_vs_GPU_reduction_roundoff_before_freezing_recorded_scales':roundoff,
        'executed_final_gtol':gtol,
        'saved_endpoint':stripped(saved),'saved_linear_actual_normal_relative':normal,
        'objective_difference_from_original_GPU_log':objective_delta,'gradient_difference_from_original_GPU_log':gradient_delta,
        'raw_forward_aggregates':aggregate,'raw_forward_metric_difference_from_original_GPU_receipt':raw_metric_delta,
        'tighter_conditional_linear_references':references,'latent_temperature_near_5K_knots':{
            'pixels':65536,'within_0p25K':int(np.count_nonzero(distance<=.25)),
            'within_1e_minus4K':int(np.count_nonzero(distance<=1e-4))},
        'full_canonical_GPU_byte_equality_claimed':False,'wall_s':time.perf_counter()-started,
        'complete_inverse_calls':0,'outer_optimizer_calls':0}


def main():
    from .runtime_entry import install_platform_fallback
    entry=install_platform_fallback();os.environ['CUDA_VISIBLE_DEVICES']='-1'
    HERE.mkdir(parents=True,exist_ok=True);source_sha=checked(s.ROOT/SOURCE)
    started=time.perf_counter();metadata,registration,payloads=metadata_review()
    s.write(HERE/'METADATA_REVIEW.json',metadata)
    print(json.dumps({'stage':'metadata_complete','passed':metadata['all_metadata_checks_passed'],
        'new_cells':len(metadata['cells']),'recorded_core_wall_s':metadata['recorded_compute']['new_eight_core_wall_s']}),flush=True)
    from .smooth_calibration import physics,contract
    import torch
    from . import numerical_strategy,joint_estimator
    def forbidden(*args,**kwargs): raise RuntimeError('A complete inverse or outer optimizer is forbidden in this reviewer')
    for module in (numerical_strategy,joint_estimator):
        for name in ('repaired_infer','recovered_infer','infer','minimize'):
            if hasattr(module,name): setattr(module,name,forbidden)
    torch.set_num_threads(2);p=physics('cpu')
    if contract(p)!=registration['calibration_contract']: raise ValueError('Frozen C1 CPU calibration differs')
    endpoint_rows=[]
    for identity in SELECTED:
        print(json.dumps({'stage':'cpu_fixed_endpoint_begin','case_id':identity[0],'initializer':identity[1]}),flush=True)
        try: row=cpu_endpoint_review(payloads[identity],registration,p)
        except Exception as error:
            row={'case_id':identity[0],'initializer':identity[1],'passed':False,
                'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
                'complete_inverse_calls':0,'nonlinear_optimization_performed':False}
        endpoint_rows.append(row);s.write(HERE/('ENDPOINT_'+str(len(endpoint_rows))+'.json'),row)
        print(json.dumps({'stage':'cpu_fixed_endpoint_end','case_id':identity[0],'passed':row['passed'],
            'wall_s':row.get('wall_s')}),flush=True)
    immutable={path:s.sha(path)==digest for path,digest in TOUCHED.items()}
    # Negative controls reject optimizer success without true gtol, or a
    # missing second platform, using copies of a frozen certificate only.
    core=next(iter(payloads.values()))[1]['original_info']['numerical_engine']
    fake=copy.deepcopy(core);fake['stages'][-1]['gradient']['scaled_projected_linf']=1.
    missing=copy.deepcopy(core);missing['output_comparisons']=missing['output_comparisons'][:1]
    controls={'false_optimizer_success_with_large_gradient_rejected':not stage_contract(fake)['passed'],
        'missing_second_platform_rejected':not stage_contract(missing)['passed']}
    passed=bool(metadata['all_metadata_checks_passed'] and all(r['passed'] for r in endpoint_rows)
        and all(immutable.values()) and all(controls.values()) and not torch.cuda.is_initialized())
    final={'schema':VERSION,'time':s.now(),'all_review_checks_passed':passed,'metadata_review':ref(HERE/'METADATA_REVIEW.json'),
        'denominators':{**metadata['denominators'],'new_certificate_stage_records_checked':sum(len(c['certificate']['stages']) for c in metadata['cells']),
            'new_certificate_platform_records_checked':sum(len(c['certificate']['platforms']) for c in metadata['cells']),
            'independently_CPU_recomputed_final_endpoints':len(endpoint_rows),'CPU_recomputed_passed_endpoints':sum(r['passed'] for r in endpoint_rows)},
        'CPU_endpoint_checks':endpoint_rows,'negative_controls':controls,'file_sha256':dict(TOUCHED),
        'all_frozen_read_files_unchanged':all(immutable.values()),'frozen_read_files_rechecked':len(immutable),
        'counters':COUNTERS,'GPU_initialized':torch.cuda.is_initialized(),'execution_entry':entry,
        'runtime':{'torch':str(torch.__version__),'threads':torch.get_num_threads(),'dtype':'float64','device':'cpu'},
        'source_sha256':{SOURCE:source_sha},'wall_s':time.perf_counter()-started,
        'scientific_acceptance_claimed':False,'detection_calibrated':False,'truth_or_sealed_payloads_opened':False,
        'findings':['All8 new cells and all4 reused v002 cells are retained in the full12 denominator.',
            'The8 new logs execute3 stages ending at gtol=1e-6; the declared stage4 gtol=1e-7 is not executed.',
            'Earlier-stage parameters and platform arrays were not saved; their24 stage checks and16 comparisons are frozen-log checks, not independent physical replays.',
            'Only the preselected blank/v3 and positive_weak/v3 final states receive independent CPU physical-gradient and tighter warm/cold conditional-linear checks.',
            'Initializer pair template RMS uses orthonormal coefficient Parseval and latent temperature RMS uses saved z; a color-output pair RMS is not inferred from scalar metric differences.',
            'Numerical certification and close initializers do not establish scientific accuracy, detector calibration, identifiability or independent-test generalization.',
            'All8 caches retain the original reservation-release warning; no existing record or registered source is edited.']}
    s.write(HERE/'NUMERICAL_REVIEW.json',final)
    receipt={'time':s.now(),'schema':VERSION+'-receipt','all_review_checks_passed':passed,
        'source':ref(s.ROOT/SOURCE),'review':ref(HERE/'NUMERICAL_REVIEW.json'),
        'metadata':ref(HERE/'METADATA_REVIEW.json'),'denominators':final['denominators'],
        'counters':COUNTERS,'formal_inverse_calls':0,'GPU_used':False,'old_sources_or_results_modified':False}
    s.write(HERE/'REVIEW_RECEIPT.json',receipt)
    print(json.dumps({'receipt':ref(HERE/'REVIEW_RECEIPT.json'),**receipt},indent=2),flush=True)
    raise SystemExit(0 if passed else 2)


if __name__=='__main__': main()
