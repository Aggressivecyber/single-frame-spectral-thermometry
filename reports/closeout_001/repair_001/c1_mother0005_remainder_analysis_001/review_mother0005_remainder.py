"""Read-only review of54 completed mother0005 remaining-condition cells.

The same two predeclared regularizer corners as the prescreen review receive
fixed-state CPU gradient and tighter conditional-linear checks. This program
never calls an initializer, outer optimizer, complete inverse or truth loader.
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
from datetime import datetime
import hashlib
import json
import os
import time
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=s.REPORTS/'repair_001/c1_selection_001'
SHARD=RUN/'shards/569f8ea57f09595ba1bc9a8dd6143013664f3d0034715181dde2ab626d9450fc/RUN_a51ab38052644471b5d6b26501cd8f2a.json'
VERSION='C1-mother0005-remainder-independent-numerical-review-v001'
CASES=('v5_development_selection_mother_0005_negative_weak_noise0',
       'v5_development_selection_mother_0005_positive_normal_noise0',
       'v5_development_selection_mother_0005_negative_normal_noise0')
SELECTED=((CASES[0],'joint_c0.1_s10','v3'),(CASES[2],'joint_c10_s0.1','smooth_demod'))
HELPER=s.ROOT/'research_closeout/quick_v003_numerical_review.py'
PARENT=s.REPORTS/'repair_001/c1_negative_pair_analysis_001/review_negative_pair.py'
RULES=s.REPORTS/'repair_001/LATEST_USER_EXECUTION_RULES_20261016.txt'


def required(condition,message):
    if not condition: raise ValueError(message)
def stamp(value): return datetime.fromisoformat(value)
def identity(value,expected): required(all(value.get(k)==v for k,v in expected.items()),'Identity differs: '+repr(expected))


def load_states(registration,shard,digest):
    rows=[];references={}
    for receipt in shard['case_state_receipts']:
        R.checked_ref(receipt);state=R.sealed(receipt['path']);row=state['row']
        required(state['registration_sha256']==digest,'Case-state registration differs')
        expected=s.canonical(row)
        required(expected==state['content_sha256'] and Path(receipt['path']).name==f"STATE_{state['sequence']:06d}_{expected}.json",
                 'Case-state content/sequence filename differs')
        key=s.canonical([row[k] for k in ('case_id','method','initializer','raw_sha256')])
        required(Path(receipt['path']).parent.name==key==row['reuse_key'],'Semantic case-state key differs')
        planned=next(c for c in registration['planned_cells'] if c['case_id']==row['case_id'] and c['method']==row['method'])
        required(all(row[k]==v for k,v in planned.items()),'Immutable case state changes the registered plan')
        required(row['case_state_durable'] and row['case_id'] in CASES,'State is not a durable registered mother0005 remainder cell')
        for evidence in row['evidence_files']: R.checked_ref(evidence)
        rows.append(row);references[(row['case_id'],row['candidate_id'],row['initializer'])]=receipt
    expected={(c['case_id'],c['candidate_id'],c['initializer']) for c in registration['planned_cells'] if c['case_id'] in CASES}
    required(len(rows)==54 and {(r['case_id'],r['candidate_id'],r['initializer']) for r in rows}==expected,'Full54-cell shard denominator differs')
    return rows,references


def audit_cache(cell,registration,digest):
    required(cell['execution_role']=='NEW_INVERSE' and cell['new_inverse_calls_selection_total']==1
        and not cell['prior_result_reused'],'This shard requires54 distinct new method/raw attempts')
    report=Path(cell['compact_report_root']);source_registration=registration;source_digest=digest
    successes=[r for r in cell['evidence_files'] if Path(r['path']).name=='SCORE_SUCCESS.json']
    required(len(successes)==1,'New cell lacks one durable score-success chain')
    full=R.sealed(successes[0]['path'])['row']
    for field in ('status','finite_output','production_numerical_certified','score_status','metrics'):
        required(full[field]==cell[field],'Case state differs from original full record: '+field)
    records={}
    for path in report.glob('*.json'):
        if not path.name.endswith('.sha256.json'): records[path.name]=R.sealed(path)
    prepared=records['PREPARED.json'];context=prepared['context'];parameters=records['PARAMETERS.json']
    fingerprint=s.canonical(context);key=s.canonical({'identity':context['identity'],
        'raw_sha256':context['input_reference']['sha256'],'calibration':context['calibration']['fingerprint'],'config':context['config']})
    required(context==records['STARTED.json']['context'],'STARTED/PREPARED context differs')
    for record in records.values():
        required(record['cache_key']==key and record['context_fingerprint']==fingerprint,'Compact context/key differs')
    required(report.name==key,'Compact path key differs')
    identity(context['identity'],{'registration_sha256':source_digest,'case_id':cell['case_id'],
        'method':cell['method'],'raw_sha256':cell['raw_sha256']})
    required(context['source_sha256']==source_registration['source_sha256'],'Source registration differs')
    for name,value in context['source_sha256'].items(): R.checked(s.ROOT/name,value)
    required(context['calibration']==registration['calibration_contract'],'Calibration differs')
    required(context['config']['physical_options']==cell['physical_options'],'Physical options differ')
    values=R.read_parameters(parameters,prepared)
    final=records['RESULT.json'];score=records['SCORE_panel_external_v001.json']
    required(final['verification']['reconstructed_canonical_arrays']==prepared['canonical_arrays'],
             'Original/reconstructed canonical descriptors differ')
    required(final['verification']['reconstructed_continuum_latent']==prepared['continuum_latent'],'Latent continuum descriptor differs')
    required(final['verification']['all_byte_identical'] and all(final['verification']['per_array_byte_identical'].values()),
             'Frozen native GPU byte verification did not pass')
    required(final['status']=='LOSSLESS_READY' and prepared['original_numerical_valid'] and prepared['original_finite_canonical'],
             'Original endpoint is not finite/certified')
    required(score['score']==full['external_score'] and score['score']['metrics']==cell['metrics'],'External score chain differs')
    info=prepared['original_info']
    required(info['config_fingerprint']==source_registration['config_fingerprint'],'Source controller fingerprint differs')
    required(info['objective_fingerprint']==registration['objective_fingerprints'][cell['candidate_id']],
             'Candidate frozen physical objective differs')
    required(info['calibration_contract']==registration['calibration_contract'],'Wrapper calibration differs')
    return info,values,{'original_parameter_receipt':parameters,'prepared':R.ref(report/'PREPARED.json'),
        'canonical_hashes_bound':True,'source_registration_sha256':source_digest,'original_result_reused':False,
        'metadata_sha256':{name:R.checked(report/name) for name in records}},records['STARTED.json'],score


def metadata():
    begin=time.perf_counter();R.checked(__file__);R.checked(HELPER);R.checked(PARENT);R.checked(RULES)
    registration=R.sealed(RUN/'SELECTION_REGISTRATION.json');digest=R.checked(RUN/'SELECTION_REGISTRATION.json');shard=R.sealed(SHARD)
    required(shard['registration_sha256']==digest and shard['raw_ids']==list(CASES) and shard['planned_shard_cells']==54,
             'Completed shard identity differs')
    required(shard['status']=='FIRST_STAGE_SHARD_COMPLETE' and shard['new_inverse_calls_this_shard']==54,
             'Completed shard state/count differs')
    config=registration['config'];fingerprint=s.canonical({key:registration[key] for key in ('config','source_sha256','store_contract',
        'format_contract','objective_fingerprints','planned_cells','prior_call_prefix')})
    required(fingerprint==registration['config_fingerprint'],'Selection registration fingerprint differs')
    required(config['planned_cells']==1080 and len(registration['planned_cells'])==1080,'Full first-stage plan denominator changed')
    for name,value in registration['source_sha256'].items(): R.checked(s.ROOT/name,value)
    for dependency in registration['read_dependencies']: R.checked_ref(dependency)
    qualification=R.sealed(RUN/'NAMESPACE_FORWARD_QUALIFICATION.json')
    required(qualification['all_passed'] and qualification['registration_sha256']==digest and qualification['passed_existing_endpoints']==4,
             'Separate namespace qualification failed')
    rows,state_refs=load_states(registration,shard,digest)
    required(sum(c['execution_role']=='NEW_INVERSE' for c in rows)==54 and sum(c['execution_role']=='REUSE_PRIOR_C1V3' for c in rows)==0,
             'New/reused shard denominator differs')
    references=[next(r for r in registration['references'] if r['case_id']==case) for case in CASES]
    for reference in references:
        case=reference['case_id']
        required(reference['partition_role']=='development_selection' and reference['cohort']=='development','Only development raw allowed')
        raw=R.raw_array(reference);provenance=registration['v3_initializer_provenance'][case]
        for name in ('state_cache','receipt_journal'): R.checked_ref(provenance[name])
        journal=list(s.rows(provenance['receipt_journal']['path']));line=journal[provenance['receipt_journal']['line']-1]
        dtype={'torch.float32':np.float32,'torch.float64':np.float64}[provenance['source_input_dtype']]
        required(R.array_sha(raw)==provenance['raw_array_sha256'] and R.array_sha(raw.astype(dtype))==provenance['inference_input_array_sha256']
            and line['raw_sha256']==provenance['raw_file_sha256']==reference['raw_reference']['sha256']
            and line['case_id']==case and line['source']=='C_v3' and line['source_metadata']==provenance['source_metadata']
            and not provenance['truth_or_identity_used'],'Raw camera/source dtype or deployable initializer identity differs')
    cells=[];states={};starts=[];ticket_costs=[]
    for cell in rows:
        info,values,chain,started,score=audit_cache(cell,registration,digest)
        core=info['numerical_engine'];certificate=R.stage_contract(core);physical=core['physical_objective'];m=cell['metrics']
        required(info['anchor_regscale_contract']==physical,'Wrapper/core shared anchor differs')
        required(all(physical['parameters'][k]==v for k,v in cell['physical_options'].items()),'Registered regularizer multipliers differ')
        required(core['numerical_options']==registration['numerical_policy'],'Unchanged numerical policy differs')
        required(cell['numerical_evidence']['actual_requested_endpoint_gtol']==certificate['executed_final_gtol']
            and cell['numerical_evidence']['returned_scaled_projected_linf']==core['returned_gradient']['scaled_projected_linf'],
            'Case-state gradient/tolerance differs from original engine')
        domain=bool(m['domain_pixels']==m['accepted_pixels']==m['raw_pixels']==65536
            and m['temperature_reference_color_comparison_pixels']==m['temperature_latent_truth_comparison_pixels']==65536
            and m['nonfinite_domain_pixels']==m['nonfinite_output_pixels']==0)
        if started:
            starts.append(stamp(started['time']));ticket_costs.append((stamp(score['time'])-stamp(started['time'])).total_seconds())
        row={'candidate_id':cell['candidate_id'],'initializer':cell['initializer'],'case_id':cell['case_id'],'execution_role':cell['execution_role'],
            'passed':bool(certificate['passed'] and domain and cell['finite_output'] and cell['production_numerical_certified']
                and cell['score_status']=='SCORED' and core['wall_s']<=180.),'physical_options':cell['physical_options'],
            'case_state_receipt':state_refs[(cell['case_id'],cell['candidate_id'],cell['initializer'])],
            'numerical_certificate':certificate,'original_float64_parameter_chain':chain,'metrics':m,
            'full_65536_domain_verified':domain,'core_wall_s':core['wall_s'],'stage_wall_s':sum(v['wall_s'] for v in core['stages']),
            'objective_evaluations':core['objective_evaluations'],'returned_objective':core['returned_endpoint_objective'],
            'returned_gradient':core['returned_gradient'],'shared_anchor':physical,'coordinate_contract':core['numerical_coordinates'],
            'independent_early_stage_replay_performed':False,'independent_CPU_canonical_byte_reconstruction_performed':False}
        cells.append(row);states[(cell['case_id'],cell['candidate_id'],cell['initializer'])]=(cell,values,physical,core)
    cfg=s.read(s.ROOT/'reports/training_v4/run_001/backbone_selection.json')['physics_config'];pairs=[]
    for case,candidate in sorted({(c['case_id'],c['candidate_id']) for c in rows}):
        a,av,anchor,ac=states[(case,candidate,'v3')];b,bv,other,bc=states[(case,candidate,'smooth_demod')];delta=av['z_grid']-bv['z_grid']
        pairs.append({'case_id':case,'candidate_id':candidate,'same_physical_anchor':anchor==other,
            'latent_temperature_pair_RMS_K':float(cfg['temperature_scale_K']*np.sqrt(np.mean(delta[:,0]**2))),
            'loga_pair_RMS':float(np.sqrt(np.mean(delta[:,1]**2))),
            'template_pair_RMS_e_from_orthonormal_coefficients':float(np.linalg.norm(av['xi']-bv['xi'])/256),
            'pair_template_RMS_derivation':'Parseval in frozen real orthonormal Fourier basis; no full endpoint image reconstruction',
            'beta_pair_maximum_absolute':float(np.max(np.abs(av['beta']-bv['beta']))),
            'objective_difference':ac['returned_endpoint_objective']-bc['returned_endpoint_objective'],
            'color_RMSE_metric_absolute_difference_K':abs(a['metrics']['temperature_RMSE_K']-b['metrics']['temperature_RMSE_K']),
            'common_color_output_pair_RMS_not_inferred_from_metric_difference':True})
    ledger=s.REPORTS/'inverse_calls.jsonl';R.checked(ledger)
    with ledger.open('rb') as handle: prefix=handle.read(registration['prior_call_prefix']['prefix_bytes'])
    required(hashlib.sha256(prefix).hexdigest()==registration['prior_call_prefix']['prefix_sha256'],'Old true-call ledger prefix changed')
    all_calls=list(s.rows(ledger));calls=[c for c in all_calls if c.get('phase')=='C1v3_selection_first_stage_v001' and c.get('case_id') in CASES]
    method_raws={(c['method'],c['case_id']) for c in rows}
    calls_ok=len(calls)==54 and {(c['method'],c['case_id']) for c in calls}==method_raws
    new=[c for c in cells if c['execution_role']=='NEW_INVERSE'];platforms=[p for c in cells for p in c['numerical_certificate']['platforms']]
    immutable={path:s.sha(path)==value for path,value in R.TOUCHED.items()}
    passed=bool(all(c['passed'] for c in cells) and all(c['same_physical_anchor'] for c in pairs) and calls_ok and all(immutable.values()))
    summary={'schema':VERSION,'time':s.now(),'all_metadata_checks_passed':passed,'shard_receipt':R.ref(SHARD),
        'selection_registration':R.ref(RUN/'SELECTION_REGISTRATION.json'),
        'source_sha256':{str(Path(__file__).resolve()):R.checked(__file__),str(HELPER):R.checked(HELPER),str(PARENT):R.checked(PARENT)},
        'registered_sources':len(registration['source_sha256']),'registered_read_dependencies':len(registration['read_dependencies']),
        'denominators':{'mother0005_remainder_cells':54,'audited_cells':len(cells),'finite_certified_cells':sum(c['passed'] for c in cells),
            'old_reused_cells':0,'new_cells':54,'Q':9,'unique_raws':3,'unique_mothers':1,'pixels_per_cell':65536,
            'repeated_cell_pixel_total':54*65536,'independent_scientific_sample_size_claimed':False,
            'stage_records_checked':sum(len(c['numerical_certificate']['stages']) for c in cells),'platform_records_checked':len(platforms),
            'independently_replayed_early_stages':0,'whole_first_stage_planned_cells':1080,
            'whole_first_stage_completed_finite_cells_at_shard_end':shard['global_summary_counts']['finite_output_cells'],
            'whole_first_stage_not_executed_at_shard_end':shard['global_summary_counts']['not_executed_cells']},
        'recorded_cost':{'new54_core_wall_s':sum(c['core_wall_s'] for c in new),'new54_stage_wall_s':sum(c['stage_wall_s'] for c in new),
            'all54_historical_core_wall_s':sum(c['core_wall_s'] for c in cells),'reused_cells':0,
            'new54_objective_evaluations':sum(c['objective_evaluations'] for c in new),
            'first_new_ticket_to_shard_terminal_wall_s':(stamp(shard['time'])-min(starts)).total_seconds(),
            'sum_new54_ticket_to_score_receipt_wall_s':sum(ticket_costs),'registered_new54_budget_s_excluding_overhead':54*180,
            'time_scope':'Wall clock from original immutable records; not GPU kernel time; core excludes cached initializer history/cache/scoring/controller overhead'},
        'actual_gradients':{'final_gtols':sorted({c['numerical_certificate']['executed_final_gtol'] for c in cells}),
            'minimum_final_scaled_projected_linf':min(c['returned_gradient']['scaled_projected_linf'] for c in cells),
            'maximum_final_scaled_projected_linf':max(c['returned_gradient']['scaled_projected_linf'] for c in cells),
            'stage4_1e_minus7_executed_cells':sum(c['numerical_certificate']['stage4_1e_minus7_executed'] for c in cells)},
        'maximum_logged_platform_changes':{'template_RMS_e':max(p['template_RMS_difference_e'] for p in platforms),
            'color_RMS_K':max(p['color_RMS_difference_K'] for p in platforms)},
        'cells':cells,'initializer_pairs':pairs,'new_shard_call_records':calls,'call_ledger_passed':calls_ok,
        'global_ledger_rows':len(all_calls),'global_charged_calls':sum(not c.get('reused') for c in all_calls),
        'references':references,'CPU_selected_extreme_endpoints':[list(v) for v in SELECTED],
        'latest_direct_user_execution_rules':R.ref(RULES),
        'all_frozen_objects_byte_unchanged':all(immutable.values()),'protected_file_sha256':dict(R.TOUCHED),
        'frozen_files_rechecked':len(immutable),'formal_inverse_calls':0,'GPU_used':False,'truth_payloads_opened':False,
        'scientific_acceptance_or_method_selection_claimed':False,
        'historical_temporary_fixture_policy':{'fixture_root_strings_are_historical_context_not_registered_file_references':True,
            'deleted_tiny_fixture_directories_are_not_required_to_exist':True,
            'retained_registered_CPU_reports_and_manifests_are_hashed_as_file_dependencies':True,
            'fixture_roots_are_not_recursively_inspected_or_recreated':True},'wall_s':time.perf_counter()-begin}
    s.write(HERE/'METADATA_REVIEW.json',summary)
    receipt={'time':s.now(),'all_metadata_checks_passed':passed,'metadata':R.ref(HERE/'METADATA_REVIEW.json'),
        'source_sha256':summary['source_sha256'],'denominators':summary['denominators'],'formal_inverse_calls':0,'GPU_used':False}
    s.write(HERE/'METADATA_RECEIPT.json',receipt)
    print(json.dumps({'receipt':R.ref(HERE/'METADATA_RECEIPT.json'),'denominators':summary['denominators'],'cost':summary['recorded_cost'],
        'actual_gradients':summary['actual_gradients'],'platforms':summary['maximum_logged_platform_changes'],
        'ledger_rows':len(all_calls),'charged_calls':summary['global_charged_calls'],'all_metadata_checks_passed':passed},indent=2),flush=True)
    return summary


def endpoints():
    begin=time.perf_counter();receipt=s.read(HERE/'METADATA_RECEIPT.json');R.checked_ref(receipt['metadata'])
    for path,value in receipt['source_sha256'].items(): R.checked(path,value)
    meta=s.read(HERE/'METADATA_REVIEW.json');required(meta['all_metadata_checks_passed'],'Metadata audit failed')
    R.TOUCHED.update(meta['protected_file_sha256']);registration=R.sealed(RUN/'SELECTION_REGISTRATION.json');shard=R.sealed(SHARD)
    rows,_=load_states(registration,shard,R.checked(RUN/'SELECTION_REGISTRATION.json'))
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    from research_closeout.smooth_calibration import physics,contract
    torch.set_num_threads(2);p=physics('cpu');required(contract(p)==registration['calibration_contract'],'Frozen C1 physics differs')
    reviews=[]
    for case,candidate,initializer in SELECTED:
        cell=next(c for c in rows if c['case_id']==case and c['candidate_id']==candidate and c['initializer']==initializer)
        raw=R.raw_array(next(ref for ref in meta['references'] if ref['case_id']==case))
        print(json.dumps({'stage':'fixed_CPU_endpoint_begin','case_id':case,'candidate':candidate,'initializer':initializer}),flush=True)
        report=Path(cell['compact_report_root']);prepared=R.sealed(report/'PREPARED.json');parameters=R.sealed(report/'PARAMETERS.json')
        values=R.read_parameters(parameters,prepared)
        result=R.cpu_endpoint_review((cell,prepared,values,raw),{'config':{'physical_options':cell['physical_options']}},p)
        result['candidate_id']=candidate
        reviews.append(result);s.write(HERE/('CPU_ENDPOINT_'+str(len(reviews))+'.json'),result)
        print(json.dumps({'stage':'fixed_CPU_endpoint_end','case_id':case,'candidate':candidate,'passed':result['passed']}),flush=True)
    immutable={path:s.sha(path)==value for path,value in R.TOUCHED.items()}
    passed=bool(all(r['passed'] for r in reviews) and all(immutable.values()) and not torch.cuda.is_initialized())
    final={'schema':VERSION,'time':s.now(),'all_review_checks_passed':passed,'metadata':R.ref(HERE/'METADATA_REVIEW.json'),
        'denominators':{**meta['denominators'],'independently_CPU_recomputed_final_endpoints':len(reviews),
            'independent_CPU_endpoint_passes':sum(r['passed'] for r in reviews)},
        'CPU_endpoint_reviews':reviews,'recorded_cost':meta['recorded_cost'],'actual_gradients':meta['actual_gradients'],
        'initializer_pairs':meta['initializer_pairs'],'maximum_logged_platform_changes':meta['maximum_logged_platform_changes'],
        'historical_temporary_fixture_policy':meta['historical_temporary_fixture_policy'],
        'latest_direct_user_execution_rules':meta['latest_direct_user_execution_rules'],
        'source_sha256':receipt['source_sha256'],'all_frozen_objects_byte_unchanged':all(immutable.values()),
        'frozen_files_rechecked':len(immutable),'protected_file_sha256':dict(R.TOUCHED),'counters':R.COUNTERS,
        'GPU_initialized':torch.cuda.is_initialized(),'formal_inverse_calls':0,'outer_optimizer_calls':0,
        'raw_truth_or_sealed_payloads_opened':False,'scientific_acceptance_or_method_selection_claimed':False,
        'runtime':{'torch':str(torch.__version__),'threads':torch.get_num_threads(),'device':'cpu','dtype':'float64'},
        'execution_entry':ENTRY,'wall_s':time.perf_counter()-begin,
        'limits':['The54 mother0005 remainder cells are three registered raws from one mother repeated over Q9 and two initializers. The full1080-cell first stage remains incomplete.',
            '162 stage records and108 platform comparisons are frozen-log checks. Early-stage parameters/output arrays were not saved, so they cannot be physically replayed.',
            'Actual stage3 gtol=1e-6 is reported; unexecuted stage4 gtol=1e-7 is not claimed.',
            'Existing float64 CPU/GPU reduction rule rtol5e-13/atol1e-15 only checks machine roundoff; the original GPU shared-anchor regscale and coordinate scales are then copied exactly. No stage/output tolerance is relaxed.',
            'Two fixed nonlinear CPU endpoints only;4 tighter warm/cold conditional solves and6 fixed-parameter gradients, no outer optimizer/initializer/full inverse/GPU/truth loader.',
            'Template initializer pair RMS follows Parseval on saved orthonormal coefficients. Latent-temperature pair RMS is not a common-color output RMS.',
            'Numerical stationarity does not establish scientific superiority, detector calibration, identifiability or method selection.']}
    s.write(HERE/'FINAL_FINDINGS.json',final)
    output={'time':s.now(),'schema':VERSION+'-receipt','all_review_checks_passed':passed,
        'findings':R.ref(HERE/'FINAL_FINDINGS.json'),'metadata':R.ref(HERE/'METADATA_REVIEW.json'),
        'source_sha256':receipt['source_sha256'],'denominators':final['denominators'],'counters':R.COUNTERS,
        'all_frozen_objects_byte_unchanged':all(immutable.values()),'frozen_files_rechecked':len(immutable),
        'formal_inverse_calls':0,'GPU_used':False}
    s.write(HERE/'FINAL_REVIEW_RECEIPT.json',output)
    print(json.dumps({'receipt':R.ref(HERE/'FINAL_REVIEW_RECEIPT.json'),**output},indent=2),flush=True)
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('metadata','endpoints'))
    args=parser.parse_args();result=metadata() if args.phase=='metadata' else endpoints()
    raise SystemExit(0 if result.get('all_metadata_checks_passed',result.get('all_review_checks_passed',False)) else 2)
