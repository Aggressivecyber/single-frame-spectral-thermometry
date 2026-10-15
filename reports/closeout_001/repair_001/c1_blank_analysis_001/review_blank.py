"""Explicit read-only review of the completed18-cell blank selection shard.

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
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=s.REPORTS/'repair_001/c1_selection_001'
SHARD=RUN/'shards/465abeba9ad35f2075ef7ba2dc0a6f6dadb3de5addcdccb89bbfcfb2d628141d/RUN_3f3182eb954a430690f3ea0bb94e104f.json'
VERSION='C1-blank-shard-independent-numerical-review-v001'
CASE='v5_development_selection_mother_0000_blank_noise0'
SELECTED=(('joint_c0.1_s10','v3'),('joint_c10_s0.1','smooth_demod'))
HELPER=s.ROOT/'research_closeout/quick_v003_numerical_review.py'
PRIOR=Path('D:/CodexWork/thermometry/closeout_001/reports/repair_001/quick_v003')


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
        required(row['case_state_durable'] and row['case_id']==CASE,'State is not a durable blank-shard cell')
        for evidence in row['evidence_files']: R.checked_ref(evidence)
        rows.append(row);references[(row['candidate_id'],row['initializer'])]=receipt
    expected={(c['candidate_id'],c['initializer']) for c in registration['planned_cells'] if c['case_id']==CASE}
    required(len(rows)==18 and {(r['candidate_id'],r['initializer']) for r in rows}==expected,'Full18-cell shard denominator differs')
    return rows,references


def audit_cache(cell,registration,digest,prior_registration,prior_result):
    new=cell['execution_role']=='NEW_INVERSE'
    if new:
        report=Path(cell['compact_report_root']);source_registration=registration;source_digest=digest
        successes=[r for r in cell['evidence_files'] if Path(r['path']).name=='SCORE_SUCCESS.json']
        required(len(successes)==1,'New cell lacks one durable score-success chain')
        full=R.sealed(successes[0]['path'])['row']
    else:
        item=registration['reuse_catalog'][cell['reuse_key']];locator=item['locator']
        required(locator['kind']=='compact' and item['origin_kind']=='v003' and item['physical_identity_verified'],
                 'Blank reuse no longer points to verified original qv003 compact state')
        report=Path(locator['report']);source_registration=prior_registration;source_digest=R.checked(PRIOR/'PANEL_REGISTRATION.json')
        full=next(c for c in prior_result['summary']['cells'] if c['case_id']==CASE and c['method']==item['original_method_label'])
        required(full==item['original_row'],'Original qv003 reused row changed')
        required(cell['new_inverse_calls_selection_total']==0 and cell['prior_result_reused'],'Reused cell counted another inverse')
        required(item['row']['physical_options']==cell['physical_options'],'Reused physical multipliers differ')
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
    identity(context['identity'],{'registration_sha256':source_digest,'case_id':CASE,
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
        'canonical_hashes_bound':True,'source_registration_sha256':source_digest,'original_result_reused':not new,
        'metadata_sha256':{name:R.checked(report/name) for name in records}},records['STARTED.json'] if new else None,score if new else None


def metadata():
    begin=time.perf_counter();R.checked(__file__);R.checked(HELPER)
    registration=R.sealed(RUN/'SELECTION_REGISTRATION.json');digest=R.checked(RUN/'SELECTION_REGISTRATION.json');shard=R.sealed(SHARD)
    required(shard['registration_sha256']==digest and shard['raw_ids']==[CASE] and shard['planned_shard_cells']==18,
             'Completed shard identity differs')
    required(shard['status']=='FIRST_STAGE_SHARD_COMPLETE' and shard['new_inverse_calls_this_shard']==16,
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
    prior_registration=R.sealed(PRIOR/'PANEL_REGISTRATION.json');prior_result=R.sealed(PRIOR/'PANEL_RESULT.json')
    required(prior_result['registration_sha256']==R.checked(PRIOR/'PANEL_REGISTRATION.json'),'Original qv003 registration differs')
    rows,state_refs=load_states(registration,shard,digest)
    required(sum(c['execution_role']=='NEW_INVERSE' for c in rows)==16 and sum(c['execution_role']=='REUSE_PRIOR_C1V3' for c in rows)==2,
             'New/reused shard denominator differs')
    reference=next(r for r in registration['references'] if r['case_id']==CASE)
    required(reference['partition_role']=='development_selection' and reference['cohort']=='development','Only development raw allowed')
    raw=R.raw_array(reference);provenance=registration['v3_initializer_provenance'][CASE]
    for name in ('state_cache','receipt_journal'): R.checked_ref(provenance[name])
    journal=list(s.rows(provenance['receipt_journal']['path']));line=journal[provenance['receipt_journal']['line']-1]
    dtype={'torch.float32':np.float32,'torch.float64':np.float64}[provenance['source_input_dtype']]
    required(R.array_sha(raw)==provenance['raw_array_sha256'] and R.array_sha(raw.astype(dtype))==provenance['inference_input_array_sha256']
        and line['raw_sha256']==provenance['raw_file_sha256']==reference['raw_reference']['sha256']
        and line['case_id']==CASE and line['source']=='C_v3' and line['source_metadata']==provenance['source_metadata']
        and not provenance['truth_or_identity_used'],'Raw camera/source dtype or deployable initializer identity differs')
    cells=[];states={};starts=[];ticket_costs=[]
    for cell in rows:
        info,values,chain,started,score=audit_cache(cell,registration,digest,prior_registration,prior_result)
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
        row={'candidate_id':cell['candidate_id'],'initializer':cell['initializer'],'case_id':CASE,'execution_role':cell['execution_role'],
            'passed':bool(certificate['passed'] and domain and cell['finite_output'] and cell['production_numerical_certified']
                and cell['score_status']=='SCORED' and core['wall_s']<=180.),'physical_options':cell['physical_options'],
            'case_state_receipt':state_refs[(cell['candidate_id'],cell['initializer'])],
            'numerical_certificate':certificate,'original_float64_parameter_chain':chain,'metrics':m,
            'full_65536_domain_verified':domain,'core_wall_s':core['wall_s'],'stage_wall_s':sum(v['wall_s'] for v in core['stages']),
            'objective_evaluations':core['objective_evaluations'],'returned_objective':core['returned_endpoint_objective'],
            'returned_gradient':core['returned_gradient'],'shared_anchor':physical,'coordinate_contract':core['numerical_coordinates'],
            'independent_early_stage_replay_performed':False,'independent_CPU_canonical_byte_reconstruction_performed':False}
        cells.append(row);states[(cell['candidate_id'],cell['initializer'])]=(cell,values,physical,core)
    cfg=s.read(s.ROOT/'reports/training_v4/run_001/backbone_selection.json')['physics_config'];pairs=[]
    for candidate in sorted({c['candidate_id'] for c in rows}):
        a,av,anchor,ac=states[(candidate,'v3')];b,bv,other,bc=states[(candidate,'smooth_demod')];delta=av['z_grid']-bv['z_grid']
        pairs.append({'candidate_id':candidate,'same_physical_anchor':anchor==other,
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
    all_calls=list(s.rows(ledger));calls=[c for c in all_calls if c.get('phase')=='C1v3_selection_first_stage_v001' and c.get('case_id')==CASE]
    methods={c['method'] for c in rows if c['execution_role']=='NEW_INVERSE'}
    calls_ok=len(calls)==16 and {c['method'] for c in calls}==methods and len({(c['method'],c['case_id']) for c in calls})==16
    new=[c for c in cells if c['execution_role']=='NEW_INVERSE'];platforms=[p for c in cells for p in c['numerical_certificate']['platforms']]
    immutable={path:s.sha(path)==value for path,value in R.TOUCHED.items()}
    passed=bool(all(c['passed'] for c in cells) and all(c['same_physical_anchor'] for c in pairs) and calls_ok and all(immutable.values()))
    summary={'schema':VERSION,'time':s.now(),'all_metadata_checks_passed':passed,'shard_receipt':R.ref(SHARD),
        'selection_registration':R.ref(RUN/'SELECTION_REGISTRATION.json'),
        'source_sha256':{str(Path(__file__).resolve()):R.checked(__file__),str(HELPER):R.checked(HELPER)},
        'registered_sources':len(registration['source_sha256']),'registered_read_dependencies':len(registration['read_dependencies']),
        'denominators':{'blank_shard_cells':18,'audited_cells':len(cells),'finite_certified_cells':sum(c['passed'] for c in cells),
            'old_reused_cells':2,'new_cells':16,'Q':9,'unique_raws':1,'unique_mothers':1,'pixels_per_cell':65536,
            'repeated_cell_pixel_total':18*65536,'independent_scientific_sample_size_claimed':False,
            'stage_records_checked':sum(len(c['numerical_certificate']['stages']) for c in cells),'platform_records_checked':len(platforms),
            'independently_replayed_early_stages':0,'whole_first_stage_planned_cells':1080,
            'whole_first_stage_completed_finite_cells_at_shard_end':shard['global_summary_counts']['finite_output_cells'],
            'whole_first_stage_not_executed_at_shard_end':shard['global_summary_counts']['not_executed_cells']},
        'recorded_cost':{'new16_core_wall_s':sum(c['core_wall_s'] for c in new),'new16_stage_wall_s':sum(c['stage_wall_s'] for c in new),
            'all18_historical_core_wall_s':sum(c['core_wall_s'] for c in cells),'old2_core_wall_recharged':False,
            'new16_objective_evaluations':sum(c['objective_evaluations'] for c in new),
            'first_new_ticket_to_shard_terminal_wall_s':(stamp(shard['time'])-min(starts)).total_seconds(),
            'sum_new16_ticket_to_score_receipt_wall_s':sum(ticket_costs),'registered_new16_budget_s_excluding_overhead':16*180,
            'time_scope':'Wall clock from original immutable records; not GPU kernel time; core excludes cached initializer history/cache/scoring/controller overhead'},
        'actual_gradients':{'final_gtols':sorted({c['numerical_certificate']['executed_final_gtol'] for c in cells}),
            'minimum_final_scaled_projected_linf':min(c['returned_gradient']['scaled_projected_linf'] for c in cells),
            'maximum_final_scaled_projected_linf':max(c['returned_gradient']['scaled_projected_linf'] for c in cells),
            'stage4_1e_minus7_executed_cells':sum(c['numerical_certificate']['stage4_1e_minus7_executed'] for c in cells)},
        'maximum_logged_platform_changes':{'template_RMS_e':max(p['template_RMS_difference_e'] for p in platforms),
            'color_RMS_K':max(p['color_RMS_difference_K'] for p in platforms)},
        'cells':cells,'initializer_pairs':pairs,'new_shard_call_records':calls,'call_ledger_passed':calls_ok,
        'global_ledger_rows':len(all_calls),'global_charged_calls':sum(not c.get('reused') for c in all_calls),
        'reference':reference,'CPU_selected_extreme_endpoints':[list(v) for v in SELECTED],
        'all_frozen_objects_byte_unchanged':all(immutable.values()),'protected_file_sha256':dict(R.TOUCHED),
        'frozen_files_rechecked':len(immutable),'formal_inverse_calls':0,'GPU_used':False,'truth_payloads_opened':False,
        'scientific_acceptance_or_method_selection_claimed':False,'wall_s':time.perf_counter()-begin}
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
    raw=R.raw_array(meta['reference']);os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    from research_closeout.smooth_calibration import physics,contract
    from research_closeout import numerical_strategy,joint_estimator
    def forbidden(*args,**kwargs): raise RuntimeError('Outer optimizer/complete inverse forbidden in independent reviewer')
    for module in (numerical_strategy,joint_estimator):
        for name in ('minimize','repaired_infer','recovered_infer','infer'):
            if hasattr(module,name): setattr(module,name,forbidden)
    torch.set_num_threads(2);p=physics('cpu');required(contract(p)==registration['calibration_contract'],'Frozen C1 physics differs')
    reviews=[]
    for candidate,initializer in SELECTED:
        cell=next(c for c in rows if c['candidate_id']==candidate and c['initializer']==initializer)
        print(json.dumps({'stage':'fixed_CPU_endpoint_begin','candidate':candidate,'initializer':initializer}),flush=True)
        try:
            report=Path(cell['compact_report_root']);prepared=R.sealed(report/'PREPARED.json');parameters=R.sealed(report/'PARAMETERS.json')
            values=R.read_parameters(parameters,prepared)
            result=R.cpu_endpoint_review((cell,prepared,values,raw),{'config':{'physical_options':cell['physical_options']}},p)
            result['candidate_id']=candidate
        except Exception as error:
            result={'candidate_id':candidate,'initializer':initializer,'passed':False,'error_type':type(error).__name__,
                'error':str(error),'traceback':traceback.format_exc(),'complete_inverse_calls':0}
        reviews.append(result);s.write(HERE/('CPU_ENDPOINT_'+str(len(reviews))+'.json'),result)
        print(json.dumps({'stage':'fixed_CPU_endpoint_end','candidate':candidate,'passed':result['passed']}),flush=True)
    immutable={path:s.sha(path)==value for path,value in R.TOUCHED.items()}
    passed=bool(all(r['passed'] for r in reviews) and all(immutable.values()) and not torch.cuda.is_initialized())
    final={'schema':VERSION,'time':s.now(),'all_review_checks_passed':passed,'metadata':R.ref(HERE/'METADATA_REVIEW.json'),
        'denominators':{**meta['denominators'],'independently_CPU_recomputed_final_endpoints':len(reviews),
            'independent_CPU_endpoint_passes':sum(r['passed'] for r in reviews)},
        'CPU_endpoint_reviews':reviews,'recorded_cost':meta['recorded_cost'],'actual_gradients':meta['actual_gradients'],
        'initializer_pairs':meta['initializer_pairs'],'maximum_logged_platform_changes':meta['maximum_logged_platform_changes'],
        'source_sha256':receipt['source_sha256'],'all_frozen_objects_byte_unchanged':all(immutable.values()),
        'frozen_files_rechecked':len(immutable),'protected_file_sha256':dict(R.TOUCHED),'counters':R.COUNTERS,
        'GPU_initialized':torch.cuda.is_initialized(),'formal_inverse_calls':0,'outer_optimizer_calls':0,
        'raw_truth_or_sealed_payloads_opened':False,'scientific_acceptance_or_method_selection_claimed':False,
        'runtime':{'torch':str(torch.__version__),'threads':torch.get_num_threads(),'device':'cpu','dtype':'float64'},
        'execution_entry':ENTRY,'wall_s':time.perf_counter()-begin,
        'limits':['The18 blank cells are one raw/mother repeated over Q9 and two initializers. The full1080-cell first stage remains incomplete.',
            '54 stage records and36 platform comparisons are frozen-log checks. Early-stage parameters/output arrays were not saved, so they cannot be physically replayed.',
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
