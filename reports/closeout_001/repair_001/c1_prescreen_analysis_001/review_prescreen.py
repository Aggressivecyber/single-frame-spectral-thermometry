"""Independent18-cell prescreen audit; only two fixed CPU endpoints, no inverse.

All writes remain beside this new script. Original sources, results and latent
caches are read-only. The two CPU examples are the opposing declared corners
c=.1,s=10,v3 and c=10,s=.1,smooth_demod, chosen without consulting truth scores.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from research_closeout.runtime_entry import install_platform_fallback
ENTRY=install_platform_fallback()
from research_closeout import storage as s
from research_closeout import quick_v003_numerical_review as R
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=s.REPORTS/'repair_001/c1_development_001'
VERSION='C1-prescreen-independent-numerical-review-v001'
SELECTED=(('joint_c0.1_s10','v3'),('joint_c10_s0.1','smooth_demod'))
CASE='v5_development_selection_mother_0000_positive_normal_noise0'
HELPER=s.ROOT/'research_closeout/quick_v003_numerical_review.py'


def stamp(value): return datetime.fromisoformat(value)
def required(condition,message):
    if not condition: raise ValueError(message)
def identity(value,expected):
    required(all(value.get(k)==v for k,v in expected.items()),'Identity mismatch: '+repr(expected))


def audit_prior(cell,registration):
    """Verify the original c1s1 parameters and complete canonical arrays."""
    item=cell['origin_evidence'];origin=item['row'];key=cell['reuse_key']
    required(item==registration['reuse_catalog'][key],'Reuse catalog changed')
    oldpath=s.REPORTS/'repair_001/quick_v002/QUICK_RESULT.json';R.checked(oldpath)
    old=s.read(oldpath);actual=next(r for r in old['summary']['cells'] if r['method']==origin['method'] and r['case_id']==origin['case_id'])
    required(origin==actual,'Reused original result row changed')
    for alias in item['aliases']: required(alias['row']==origin,'Prior alias differs from original row')
    for field in ('status','finite_output','production_numerical_certified','score_status','metrics'):
        required(cell[field]==origin[field],'Reused score/status changed: '+field)
    expected={k:origin[k] for k in ('registration_sha256','method','raw_sha256','case_id','cache_key')}
    for field in ('inference_receipt','score_receipt','output_cache','output_metadata'):
        R.checked_ref(origin[field])
        if field!='output_cache': identity(s.read(origin[field]['path']),expected)
    required(s.read(origin['score_receipt']['path'])['metrics']==cell['metrics'],'Reused score receipt differs')
    receipt=origin['parameter_cache_receipt']
    for field in ('parameter_cache','parameter_metadata'): R.checked_ref(receipt[field])
    meta=s.read(receipt['parameter_metadata']['path']);identity(meta,expected)
    with np.load(receipt['parameter_cache']['path'],allow_pickle=False) as handle: values={k:handle[k].copy() for k in handle.files}
    required(set(values)=={'z_grid','xi','beta'},'Reused parameter set differs')
    for name,a in values.items():
        required(a.dtype==np.float64 and np.isfinite(a).all(),'Reused parameter precision/finite check failed')
        required(R.array_sha(a)==meta['array_sha256'][name] and list(a.shape)==meta['array_shapes'][name],'Reused latent bytes differ')
    output_meta=s.read(origin['output_metadata']['path'])
    with np.load(origin['output_cache']['path'],allow_pickle=False) as handle:
        for name in handle.files:
            a=handle[name]
            required(R.array_sha(a)==origin['array_sha256'][name]==output_meta['array_sha256'][name]==meta['canonical_array_sha256'][name],
                     'Reused canonical output hash differs: '+name)
    info=copy.deepcopy(origin['info']);info.pop('parameter_cache_receipt',None)
    required(meta['inference_info']==info,'Reused inference metadata differs')
    required(meta['objective_fingerprint']==registration['objective_fingerprints'][cell['candidate_id']],
             'Reused objective fingerprint differs')
    oldregistration=s.read(s.REPORTS/'repair_001/quick_v002/QUICK_REGISTRATION.json')
    required(all(meta[k]==oldregistration[k] for k in ('source_sha256','calibration_contract','config_fingerprint','objective_fingerprint')),
             'Reused metadata no longer binds original registration')
    return origin['info'],values,{'parameter_cache':receipt,'output_cache':origin['output_cache'],
        'metadata':origin['output_metadata'],'original_result_row_unchanged':True,'canonical_bytes_verified':True},None,None


def audit_new(cell,registration,digest):
    report=Path(cell['compact_report_root']);records={}
    for path in report.glob('*.json'):
        if not path.name.endswith('.sha256.json'): records[path.name]=R.sealed(path)
    prepared=records['PREPARED.json'];context=prepared['context'];parameters=records['PARAMETERS.json']
    required(context==records['STARTED.json']['context'],'STARTED/PREPARED context differs')
    fingerprint=s.canonical(context)
    key=s.canonical({'identity':context['identity'],'raw_sha256':context['input_reference']['sha256'],
        'calibration':context['calibration']['fingerprint'],'config':context['config']})
    for value in records.values():
        required(value['context_fingerprint']==fingerprint and value['cache_key']==key,'Compact identity/context differs')
    required(key==report.name==Path(cell['compact_data_root']).name,'Compact path identity differs')
    identity(context['identity'],{'registration_sha256':digest,'method':cell['method'],'case_id':cell['case_id'],
        'raw_sha256':cell['raw_sha256'],'controller_version':registration['config']['version']})
    required(context['calibration']==registration['calibration_contract'],'Compact calibration differs')
    required(context['config']['physical_options']==cell['physical_options'],'Physical options differ')
    required(context['source_sha256']==registration['source_sha256'],'Compact/source registry differs')
    for name,value in context['source_sha256'].items(): R.checked(s.ROOT/name,value)
    values=R.read_parameters(parameters,prepared);final=records['RESULT.json'];score=records['SCORE_panel_external_v001.json']
    required(score['score']==cell['external_score'] and score['score']['metrics']==cell['metrics'],'Bound external score differs')
    required(final['verification']['reconstructed_canonical_arrays']==prepared['canonical_arrays'],
             'Original/readback canonical descriptors differ')
    required(final['verification']['reconstructed_continuum_latent']==prepared['continuum_latent'],'Derived continuum hash differs')
    required(final['verification']['all_byte_identical'] and all(final['verification']['per_array_byte_identical'].values()),
             'Original native byte verification did not pass')
    required(final['status']=='LOSSLESS_READY' and prepared['original_numerical_valid'] and prepared['original_finite_canonical'],
             'Original compact endpoint is not finite/certified')
    info=prepared['original_info']
    required(info['objective_fingerprint']==registration['objective_fingerprints'][cell['candidate_id']],
             'Candidate objective fingerprint differs')
    required(info['config_fingerprint']==registration['config_fingerprint'],'Controller fingerprint differs')
    return info,values,{'parameters':parameters,'prepared':R.ref(report/'PREPARED.json'),
        'all_original_canonical_hashes_bound':True,'file_sha256':{name:R.checked(report/name) for name in records}},records['STARTED.json'],score


def metadata():
    begin=time.perf_counter();R.checked(__file__);R.checked(HELPER)
    registration=R.sealed(RUN/'PRESCREEN_REGISTRATION.json');result=R.sealed(RUN/'PRESCREEN_RESULT.json');digest=R.checked(RUN/'PRESCREEN_REGISTRATION.json')
    required(result['registration_sha256']==digest,'Terminal registration differs')
    for name,value in registration['source_sha256'].items(): R.checked(s.ROOT/name,value)
    for dependency in registration['read_dependencies']: R.checked_ref(dependency)
    config=registration['config'];expected_config=s.canonical({'config':config,'sources':registration['source_sha256'],
        'store':registration['store_contract'],'format':registration['format_contract'],'objectives':registration['objective_fingerprints'],
        'planned_cells':registration['planned_cells'],'prior_call_prefix':registration['prior_call_prefix']})
    required(expected_config==registration['config_fingerprint'],'Registration fingerprint differs')
    expected_candidates=[{'id':f'joint_c{c:g}_s{t:g}','continuum_multiplier':c,'template_multiplier':t}
        for c in (.1,1.,10.) for t in (.1,1.,10.)]
    required(config['candidates']==expected_candidates and config['initializers']==['v3','smooth_demod'],
             'Original Q=9 or two deployable initializers changed')
    oldregistration=s.read(s.REPORTS/'repair_001/quick_v002/QUICK_REGISTRATION.json')
    for candidate in expected_candidates:
        base=registration['base_c1s1_objective_fingerprint']
        expected=base if candidate['id']=='joint_c1_s1' else s.canonical({'base_C1v3_c1s1_objective':base,
            'regularizer_multipliers':candidate,'scientific_sources':oldregistration['source_sha256']})
        required(expected==registration['objective_fingerprints'][candidate['id']],'Objective identity differs')
    qualification=R.sealed(RUN/'NATIVE_E_COMPACT_QUALIFICATION.json')
    required(qualification['all_passed'] and qualification['registration_sha256']==digest and qualification['passed_existing_endpoints']==4,
             'Separate native E-cache qualification failed')
    reference=registration['reference'];required(reference['case_id']==CASE and reference['partition_role']=='development_selection',
        'Only registered development raw may be read')
    raw=R.raw_array(reference);provenance=registration['v3_initializer_provenance'][CASE]
    for name in ('state_cache','receipt_journal'): R.checked_ref(provenance[name])
    journal=list(s.rows(provenance['receipt_journal']['path']));line=journal[provenance['receipt_journal']['line']-1]
    dtype={'torch.float32':np.float32,'torch.float64':np.float64}[provenance['source_input_dtype']]
    required(R.array_sha(raw)==provenance['raw_array_sha256'] and R.array_sha(raw.astype(dtype))==provenance['inference_input_array_sha256']
        and line['raw_sha256']==provenance['raw_file_sha256']==reference['raw_reference']['sha256']
        and line['case_id']==CASE and line['source']=='C_v3' and line['source_metadata']==provenance['source_metadata']
        and not provenance['truth_or_identity_used'],'Cached initializer raw/source identity differs')
    rows=result['summary']['cells'];planned=registration['planned_cells']
    required(len(rows)==len(planned)==18 and {(c['method'],c['initializer']) for c in rows}=={(c['method'],c['initializer']) for c in planned},
             'The complete18-cell denominator changed')
    required(sum(c['execution_role']=='NEW_INVERSE' for c in rows)==16 and sum(c['execution_role']=='REUSE_PRIOR_C1V3' for c in rows)==2,
             'Reuse/new denominator differs')
    cells=[];states={};starts=[];ticket_costs=[]
    for cell in rows:
        required(cell['case_id']==CASE and cell['raw_sha256']==reference['raw_reference']['sha256'],'Cell raw differs')
        planned_cell=next(c for c in planned if c['method']==cell['method'])
        required(all(cell[k]==v for k,v in planned_cell.items()),'Cell planned identity/parameters changed')
        if cell['execution_role']=='NEW_INVERSE': info,values,chain,started,score=audit_new(cell,registration,digest)
        else: info,values,chain,started,score=audit_prior(cell,registration)
        core=info['numerical_engine'];certificate=R.stage_contract(core)
        required(info['anchor_regscale_contract']==core['physical_objective'],'Wrapper/core anchor differs')
        required(info['physical_objective_version']==registration['calibration_contract']['physical_objective_version'],
                 'Effective physical objective version differs')
        physical=core['physical_objective'];required(all(physical['parameters'][k]==v for k,v in cell['physical_options'].items()),
             'Regularizer multipliers differ from planned candidate')
        m=cell['metrics'];domain=bool(m['domain_pixels']==m['raw_pixels']==m['accepted_pixels']==65536
            and m['temperature_reference_color_comparison_pixels']==m['temperature_latent_truth_comparison_pixels']==65536
            and m['nonfinite_domain_pixels']==m['nonfinite_output_pixels']==0)
        evidence=cell['numerical_evidence'];required(evidence['passed'] and evidence['actual_requested_endpoint_gtol']==certificate['executed_final_gtol']
            and evidence['returned_scaled_projected_linf']==core['returned_gradient']['scaled_projected_linf'],'Summary numerical evidence differs')
        cost_ok=core['wall_s']<=180. and core['numerical_options']==registration['numerical_policy']
        if started:
            starts.append(stamp(started['time']));ticket_costs.append((stamp(score['time'])-stamp(started['time'])).total_seconds())
        row={'candidate_id':cell['candidate_id'],'initializer':cell['initializer'],'execution_role':cell['execution_role'],
            'method':cell['method'],'passed':bool(certificate['passed'] and domain and cost_ok and cell['finite_output']
                and cell['production_numerical_certified'] and cell['score_status']=='SCORED'),
            'physical_options':cell['physical_options'],'numerical_certificate':certificate,'potential_parameter_chain':chain,
            'metrics':m,'full_65536_denominator_checked':domain,'core_wall_s':core['wall_s'],
            'stage_wall_s':sum(v['wall_s'] for v in core['stages']),'objective_evaluations':core['objective_evaluations'],
            'final_objective':core['returned_endpoint_objective'],'final_gradient':core['returned_gradient'],
            'physical_anchor':physical,'coordinate_contract':core['numerical_coordinates'],
            'earlier_stage_parameter_replay_performed':False,'canonical_CPU_reconstruction_performed':False}
        cells.append(row);states[(cell['candidate_id'],cell['initializer'])]=(cell,values,physical,core)
    cfg=s.read(s.ROOT/'reports/training_v4/run_001/backbone_selection.json')['physics_config'];pairs=[]
    for candidate in expected_candidates:
        a,av,anchor,ac=states[(candidate['id'],'v3')];b,bv,other,bc=states[(candidate['id'],'smooth_demod')]
        delta=av['z_grid']-bv['z_grid'];same=anchor==other
        pairs.append({'candidate_id':candidate['id'],'same_anchor_and_physical_objective':same,
            'latent_temperature_pair_RMS_K':float(cfg['temperature_scale_K']*np.sqrt(np.mean(delta[:,0]**2))),
            'loga_pair_RMS':float(np.sqrt(np.mean(delta[:,1]**2))),
            'template_pair_RMS_e_from_orthonormal_coefficients':float(np.linalg.norm(av['xi']-bv['xi'])/256),
            'template_pair_RMS_derivation':'Parseval in frozen real orthonormal Fourier basis; no image reconstruction',
            'beta_pair_maximum_absolute':float(np.max(np.abs(av['beta']-bv['beta']))),
            'objective_difference':ac['returned_endpoint_objective']-bc['returned_endpoint_objective'],
            'color_RMSE_metric_absolute_difference_K':abs(a['metrics']['temperature_RMSE_K']-b['metrics']['temperature_RMSE_K']),
            'color_output_pair_RMS_not_inferred_from_metric_difference':True})
    ledger=s.REPORTS/'inverse_calls.jsonl';R.checked(ledger)
    with ledger.open('rb') as handle: prefix=handle.read(registration['prior_call_prefix']['prefix_bytes'])
    required(hashlib.sha256(prefix).hexdigest()==registration['prior_call_prefix']['prefix_sha256'],'Preexisting formal call ledger prefix changed')
    calls=[v for v in s.rows(ledger) if v.get('phase')=='C1v3_development_prescreen_v001']
    new_methods={c['method'] for c in rows if c['execution_role']=='NEW_INVERSE'}
    calls_ok=len(calls)==16 and {v['method'] for v in calls}==new_methods and len({(v['method'],v['case_id']) for v in calls})==16
    for path in (RUN/'cell_receipts').rglob('*.json'):
        if not path.name.endswith('.sha256.json'): R.sealed(path)
    new=[c for c in cells if c['execution_role']=='NEW_INVERSE'];platforms=[p for c in cells for p in c['numerical_certificate']['platforms']]
    immutable={path:s.sha(path)==value for path,value in R.TOUCHED.items()}
    passed=bool(all(c['passed'] for c in cells) and all(c['same_anchor_and_physical_objective'] for c in pairs)
        and calls_ok and all(immutable.values()))
    summary={'schema':VERSION,'time':s.now(),'all_metadata_checks_passed':passed,
        'source_sha256':{str(Path(__file__).resolve()):R.checked(__file__),str(HELPER):R.checked(HELPER)},
        'registration':R.ref(RUN/'PRESCREEN_REGISTRATION.json'),'result':R.ref(RUN/'PRESCREEN_RESULT.json'),
        'registered_sources':len(registration['source_sha256']),'registered_read_dependencies':len(registration['read_dependencies']),
        'denominators':{'planned_cells':18,'audited_cells':len(cells),'finite_certified_cells':sum(c['passed'] for c in cells),
            'reused_cells':2,'new_cells':16,'Q':9,'unique_raws':1,'unique_mothers':1,'pixels_per_cell':65536,
            'cell_pixel_total':18*65536,'independent_scientific_samples_claimed':False,
            'recorded_stage_checks':sum(len(c['numerical_certificate']['stages']) for c in cells),
            'recorded_platform_checks':len(platforms),'independently_replayed_early_stages':0},
        'recorded_cost':{'new16_core_wall_s':sum(c['core_wall_s'] for c in new),
            'all18_original_core_wall_s':sum(c['core_wall_s'] for c in cells),'reused_core_cost_recharged':False,
            'new16_stage_wall_s':sum(c['stage_wall_s'] for c in new),'new16_objective_evaluations':sum(c['objective_evaluations'] for c in new),
            'first_new_ticket_to_terminal_wall_s':(stamp(result['time'])-min(starts)).total_seconds(),
            'sum_new16_ticket_to_score_receipt_wall_s':sum(ticket_costs),'new16_registered_budget_s_excluding_overhead':2880,
            'time_scope':'Original wall-time receipts; core excludes cached initializer history/cache/scoring/controller overhead; not GPU kernel time'},
        'gradient_range':{'minimum_final_pg_inf':min(c['final_gradient']['scaled_projected_linf'] for c in cells),
            'maximum_final_pg_inf':max(c['final_gradient']['scaled_projected_linf'] for c in cells),
            'actual_final_gtols':sorted({c['numerical_certificate']['executed_final_gtol'] for c in cells}),
            'stage4_1e_minus7_executed_cells':sum(c['numerical_certificate']['stage4_1e_minus7_executed'] for c in cells)},
        'maximum_logged_platform_changes':{'template_RMS_e':max(p['template_RMS_difference_e'] for p in platforms),
            'color_RMS_K':max(p['color_RMS_difference_K'] for p in platforms)},
        'cells':cells,'initializer_pairs':pairs,'actual_new_call_records':calls,'call_ledger_passed':calls_ok,
        'reference':reference,'CPU_selected_extreme_endpoints':[list(v) for v in SELECTED],
        'CPU_selection_rule':'Opposing declared continuum/template multiplier-ratio extremes (.1/10 and10/.1), with one of each initializer; no truth-score selection',
        'all_frozen_files_unchanged':all(immutable.values()),'protected_file_sha256':dict(R.TOUCHED),
        'frozen_files_rechecked':len(immutable),'formal_inverse_calls':0,'GPU_used':False,
        'truth_payloads_opened':False,'scientific_acceptance_or_method_selection_claimed':False,'wall_s':time.perf_counter()-begin}
    s.write(HERE/'METADATA_REVIEW.json',summary)
    receipt={'time':s.now(),'all_metadata_checks_passed':passed,'metadata':R.ref(HERE/'METADATA_REVIEW.json'),
        'source_sha256':summary['source_sha256'],'denominators':summary['denominators'],
        'formal_inverse_calls':0,'GPU_used':False}
    s.write(HERE/'METADATA_RECEIPT.json',receipt)
    print(json.dumps({'receipt':R.ref(HERE/'METADATA_RECEIPT.json'),'denominators':summary['denominators'],
        'cost':summary['recorded_cost'],'gradient_range':summary['gradient_range'],
        'platforms':summary['maximum_logged_platform_changes'],'all_metadata_checks_passed':passed},indent=2),flush=True)
    return summary


def endpoints():
    begin=time.perf_counter();receipt=s.read(HERE/'METADATA_RECEIPT.json');R.checked_ref(receipt['metadata'])
    for path,value in receipt['source_sha256'].items(): R.checked(path,value)
    meta=s.read(HERE/'METADATA_REVIEW.json');required(meta['all_metadata_checks_passed'],'Metadata audit not passed')
    R.TOUCHED.update(meta['protected_file_sha256'])
    registration=R.sealed(RUN/'PRESCREEN_REGISTRATION.json');result=R.sealed(RUN/'PRESCREEN_RESULT.json')
    raw=R.raw_array(registration['reference']);os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    from research_closeout.smooth_calibration import physics,contract
    from research_closeout import numerical_strategy,joint_estimator
    def forbidden(*args,**kwargs): raise RuntimeError('No outer optimizer or complete inverse is permitted')
    for module in (numerical_strategy,joint_estimator):
        for name in ('minimize','repaired_infer','recovered_infer','infer'):
            if hasattr(module,name): setattr(module,name,forbidden)
    torch.set_num_threads(2);p=physics('cpu');required(contract(p)==registration['calibration_contract'],'C1 calibration differs')
    rows=[]
    for candidate,initializer in SELECTED:
        cell=next(c for c in result['summary']['cells'] if c['candidate_id']==candidate and c['initializer']==initializer)
        print(json.dumps({'stage':'fixed_CPU_endpoint_begin','candidate':candidate,'initializer':initializer}),flush=True)
        try:
            report=Path(cell['compact_report_root']);prepared=R.sealed(report/'PREPARED.json')
            parameter_receipt=R.sealed(report/'PARAMETERS.json');values=R.read_parameters(parameter_receipt,prepared)
            effective={'config':{'physical_options':cell['physical_options']}}
            row=R.cpu_endpoint_review((cell,prepared,values,raw),effective,p)
            row['candidate_id']=candidate
        except Exception as error:
            row={'candidate_id':candidate,'initializer':initializer,'passed':False,'error_type':type(error).__name__,
                'error':str(error),'traceback':traceback.format_exc(),'complete_inverse_calls':0}
        rows.append(row);s.write(HERE/('CPU_ENDPOINT_'+str(len(rows))+'.json'),row)
        print(json.dumps({'stage':'fixed_CPU_endpoint_end','candidate':candidate,'passed':row['passed']}),flush=True)
    immutable={path:s.sha(path)==value for path,value in R.TOUCHED.items()}
    passed=bool(all(r['passed'] for r in rows) and all(immutable.values()) and not torch.cuda.is_initialized())
    findings={'schema':VERSION,'time':s.now(),'all_review_checks_passed':passed,'metadata':R.ref(HERE/'METADATA_REVIEW.json'),
        'denominators':{**meta['denominators'],'independently_CPU_recomputed_final_endpoints':len(rows),
            'independent_CPU_endpoint_passes':sum(r['passed'] for r in rows)},
        'CPU_endpoint_reviews':rows,'recorded_cost':meta['recorded_cost'],'gradient_range':meta['gradient_range'],
        'initializer_pairs':meta['initializer_pairs'],'maximum_logged_platform_changes':meta['maximum_logged_platform_changes'],
        'source_sha256':receipt['source_sha256'],'all_frozen_files_unchanged':all(immutable.values()),
        'frozen_files_rechecked':len(immutable),'protected_file_sha256':dict(R.TOUCHED),'counters':R.COUNTERS,
        'GPU_initialized':torch.cuda.is_initialized(),'formal_inverse_calls':0,'outer_optimizer_calls':0,
        'raw_truth_or_sealed_payloads_opened':False,'scientific_acceptance_or_method_selection_claimed':False,
        'runtime':{'torch':str(torch.__version__),'threads':torch.get_num_threads(),'device':'cpu','dtype':'float64'},
        'execution_entry':ENTRY,'wall_s':time.perf_counter()-begin,
        'limits':['All18/54 stages/36 platforms retain their complete denominator; earlier-stage parameter/output arrays are absent, so only their frozen-log logic is checked.',
            'Actual stage3 gtol=1e-6 is reported; unexecuted stage4 gtol=1e-7 is not claimed.',
            'The frozen helper verifies CPU/GPU float64 reduction differences with rtol5e-13/atol1e-15, then uses exact original GPU anchor regscale and coordinate scales; stage/output budgets remain unchanged.',
            'Two saved nonlinear endpoints only:4 tighter warm/cold linear solves and6 fixed-parameter gradients; no outer optimization, initializer, complete inverse, GPU or truth loader.',
            'Initializer template RMS is computed by Parseval on saved orthonormal coefficients; latent-temperature RMS is not a common-color output RMS.',
            'Prescreen numerical eligibility on one development raw does not select a method or certify scientific accuracy or generalization.']}
    s.write(HERE/'FINAL_FINDINGS.json',findings)
    final={'time':s.now(),'schema':VERSION+'-receipt','all_review_checks_passed':passed,
        'findings':R.ref(HERE/'FINAL_FINDINGS.json'),'metadata':R.ref(HERE/'METADATA_REVIEW.json'),
        'source_sha256':receipt['source_sha256'],'denominators':findings['denominators'],'counters':R.COUNTERS,
        'all_frozen_files_unchanged':all(immutable.values()),'frozen_files_rechecked':len(immutable),
        'formal_inverse_calls':0,'GPU_used':False}
    s.write(HERE/'FINAL_REVIEW_RECEIPT.json',final)
    print(json.dumps({'receipt':R.ref(HERE/'FINAL_REVIEW_RECEIPT.json'),**final},indent=2),flush=True)
    return findings


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('metadata','endpoints'))
    args=parser.parse_args();value=metadata() if args.phase=='metadata' else endpoints()
    raise SystemExit(0 if value.get('all_metadata_checks_passed',value.get('all_review_checks_passed',False)) else 2)
