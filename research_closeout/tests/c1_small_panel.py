"""CPU fake-inference controller contracts; no GPU, native inverse or D write."""
from .. import storage as s
from .. import c1_small_panel as panel
from .. import compact_joint_cache as compact
import argparse
import ast
import contextlib
import copy
import inspect
import io
import json
from pathlib import Path
import uuid
import numpy as np


class FakeBudget:
    def __init__(self): self.requests=[];self.failure=None
    def check(self,requested_bytes=0):
        self.requests.append(requested_bytes)
        if self.failure: raise RuntimeError(self.failure)
        return self.ledger()
    def ledger(self): return {'fixture_only':True,'all_file_kinds_counted':True,'bytes':0}


class FakeStore:
    """E-only isolated controller fixture; never accepted by the production CLI."""
    def __init__(self,root):
        self.root=Path(root).resolve();self.report_root=self.root/'reports';self.data_root=self.root/'data'
        self.guard=FakeBudget()
    def write_path(self,path):
        p=Path(path).resolve()
        if p!=self.root and self.root not in p.parents: raise panel.PanelIntegrityError('Fixture path escapes')
        return p
    def write(self,path,value,replace=False):
        p=self.write_path(path);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w' if replace else 'x',encoding='utf8',newline='\n') as handle:
            json.dump(value,handle,allow_nan=False);handle.write('\n')
        return p
    def read(self,path): return json.loads(self.write_path(path).read_text(encoding='utf8'))
    def append(self,path,value):
        p=self.write_path(path);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('a',encoding='utf8') as handle: handle.write(json.dumps(value)+'\n')
    def budget(self): return self.guard
    def contract(self):
        return {'fixture_only':True,'task_name':'quick_v003_cpu_fixture','owner_token':'fixed-cpu-fixture-owner',
            'report_root':str(self.report_root),'data_root':str(self.data_root),
            'global_budget_roots':[str(self.report_root),str(self.data_root)],
            'E_free_floor_root':str(s.ROOT),'E_free_floor_bytes':2*2**30,
            'D_free_floor_bytes':128*2**20,'D_project_limit_bytes':256*2**20,'global_limit_bytes':4*2**30}


class FakeCache:
    def __init__(self,store,name,events,*,persist_failure=None):
        self.report=store.report_root/'compact'/name;self.data=store.data_root/'compact'/name
        self.events=events;self.started=False;self.out=None;self.result={'status':'NEW'}
        self.persist_failure=persist_failure;self.break_readback=False
    def begin(self,raw):
        self.events.append('begin')
        if self.started: return {**self.result,'run_new_inverse':False}
        self.started=True;return {'run_new_inverse':True,'owner':{'token':'fixture'}}
    def persist(self,ticket,out,raw):
        self.events.extend(['parameters_persisted','canonical_byte_verified'])
        self.out=copy.deepcopy(out)
        if self.persist_failure:
            self.result=copy.deepcopy(self.persist_failure)
        else:
            self.result={'status':'LOSSLESS_READY','score_allowed':True,
                'lossless_representation_verified':True,'numerical_certification':out['numerical_valid']}
        return self.result
    def resume(self,raw):
        self.events.append('resume')
        if self.out is None:
            return {'status':'FAIL_CLOSED_INTERRUPTED','score_allowed':False,'original_fallback_available':False}
        return self.result
    def load_for_score(self,raw):
        self.events.append('load_for_score')
        if self.break_readback or self.out is None or not self.result.get('score_allowed'):
            raise RuntimeError('Injected compact readback mismatch')
        return copy.deepcopy(self.out)
    def record_score(self,name,value,raw):
        self.events.append('score_receipt');return {'status':'EXTERNAL_SCORE_SAVED','score_id':name}
    def inspect(self): return self.result


def fixture_refs():
    refs=[]
    for index,(mother,condition) in enumerate(panel.q.PANEL):
        case=f'v5_development_selection_mother_{mother:04d}_{condition}_noise0'
        refs.append({'case_id':case,'mother_id':f'v5_development_selection_mother_{mother:04d}',
            'condition':condition,'input_kind':'noise','cohort':'development','fixture_only':False,
            'partition_role':'development_selection','quick_geometry':{'fixture_metadata_only':True},
            'raw_reference':{'shape':[256,256],'dtype':'uint16','sha256':f'{index+1:064x}',
                'path':'metadata_only_not_opened.npz','array_sha256':f'{index+11:064x}'}})
    return refs


def fixture_origin(cell):
    return {**cell,'registration_sha256':'origin-v002','status':'FINITE_NUMERICALLY_CERTIFIED',
        'score_status':'SCORED','finite_output':True,'production_numerical_certified':True,
        'metrics':{'domain_pixels':65536,'template_RMSE_e':2.},'output_cache':{'path':'origin-never-opened'},
        'parameter_cache_receipt':{'fixture_only':True}}


def run(output=None):
    # The explicit entry was installed by the root-approved launcher before
    # any optional numerical library can ask the platform WMI endpoint.
    entry=panel.execution_entry();token=uuid.uuid4().hex
    old_registration=s.read(panel.c2.REGISTRATION)
    protected_paths=[s.ROOT/name for name in old_registration['source_sha256']]
    protected_paths.extend((panel.c2.REGISTRATION,panel.c2.REPORT_DIR/'QUICK_RESULT.json',
        panel.q.REGISTRATION,panel.c2.V001_RESULT,s.REPORTS/'inverse_calls.jsonl'))
    def protected_snapshot():
        return {str(path):s.sha(path) if path.is_file() else None for path in protected_paths}
    protected_before=protected_snapshot()
    root=panel.CPU_REPORT_DIR/'fixtures'/token;root.mkdir(parents=True,exist_ok=False)
    checks=[]
    def check(name,passed,**details): checks.append({'name':name,'passed':bool(passed),**s.plain(details)})
    def rejects(action,exceptions=(panel.PanelIntegrityError,FileExistsError,ValueError,RuntimeError)):
        try: action()
        except exceptions: return True
        return False
    refs=fixture_refs();planned=panel.planned_cells(refs)
    originals=[fixture_origin(cell) for cell in planned[:4]]
    check('fixed_six_raw_two_initializer_matrix_reuses_four_and_runs_only_eight',
        len(planned)==12 and sum(c['execution_role']=='REUSE_V002' for c in planned)==4
        and sum(c['execution_role']=='NEW_INVERSE' for c in planned)==8
        and panel.CONFIG['maximum_new_inverse_calls']==8
        and panel.CONFIG['new_inverse_wall_budget_s_excluding_overhead']==1440
        and panel.CONFIG['numerical_options']=={'cost_limit_s':180.}
        and panel.CONFIG['physical_options']=={'grid':256,'continuum_multiplier':1.,'template_multiplier':1.}
        and [r['condition'] for r in refs[2:]]==['blank','positive_weak','negative_weak','positive_normal'])
    check('raw_order_extra_cases_changed_initializer_and_grid_are_rejected_or_absent',
        rejects(lambda:panel.planned_cells(refs[::-1])) and rejects(lambda:panel.planned_cells(refs[:5]))
        and rejects(lambda:panel.planned_cells(refs+[refs[0]]))
        and {r['initializer'] for r in planned}=={'v3','smooth_demod'}
        and panel.CONFIG['grid']==256 and not panel.CONFIG['scientific_nine_regularizer_grid_modified'])
    reused=[panel.reuse_row(cell,origin,'panel-registration') for cell,origin in zip(planned[:4],originals)]
    mixed=panel.summary(planned,reused+[{**planned[4],'status':'INFERENCE_FAILED_NO_RERUN',
        'inference_attempted':True,'finite_output':False,'production_numerical_certified':False,
        'new_inverse_calls_this_run':1,'new_inverse_calls_panel_total':1,'score_status':'NOT_SCOREABLE'}])
    check('full_twelve_denominator_preserves_reused_failed_and_unexecuted_cells',
        mixed['planned_cells']==12 and len(mixed['cells'])==12 and mixed['reused_v002_cells']==4
        and mixed['not_executed_cells']==7 and mixed['finite_output_cells']==4
        and mixed['cells'][4]['status']=='INFERENCE_FAILED_NO_RERUN'
        and all(r['origin_cell']==o for r,o in zip(reused,originals)))
    changed={**originals[0],'initializer':'smooth_demod'}
    check('reused_v002_origin_method_raw_initializer_and_scores_cannot_be_rewritten',
        rejects(lambda:panel.reuse_row(planned[0],changed,'digest'))
        and all(r['new_inverse_calls_this_run']==0 and r['score_reused'] and r['inference_reused'] for r in reused))

    calls=[]
    qualification=panel.qualification_matrix(originals,lambda origin:{'passed':origin is not originals[1],
        'status':'fixture comparison','complete_inverse_calls':0})
    check('native_qualification_failure_retains_all_four_and_blocks_new_inference',
        qualification['audited_existing_endpoints']==4 and qualification['passed_existing_endpoints']==3
        and not qualification['all_passed'] and qualification['complete_inverse_calls']==0
        and panel.summary(planned,reused)['not_executed_cells']==8)
    def broken(origin):
        calls.append(origin['case_id'])
        if len(calls)==2: raise ArithmeticError('Injected forward-only qualification failure')
        return {'passed':True}
    failed=panel.qualification_matrix(originals,broken)
    check('qualification_exceptions_cannot_drop_later_endpoints_or_invoke_inverse',
        len(calls)==4 and len(failed['cells'])==4 and not failed['all_passed']
        and failed['optimizer_calls']==failed['inner_solver_calls']==failed['complete_inverse_calls']==0)
    reg={'format_contract':{'schema':'fixture-v002'},'source_sha256':{'fixture':'hash'},'store_contract':{'owner':'fixture'}}
    runtime={'device':'cuda:fixture','dtype':'float64'}
    good={'all_passed':True,'passed_existing_endpoints':4,'registration_sha256':'digest','native_GPU':True,
        'runtime':runtime,**reg}
    panel.require_qualification(good,reg,'digest',runtime)
    mismatch=[]
    for altered in ({'all_passed':False},{'passed_existing_endpoints':3},{'native_GPU':False},
        {'registration_sha256':'another'},{'runtime':{'device':'different'}},
        {'format_contract':{'schema':'different'}},{'source_sha256':{}},{'store_contract':{'owner':'foreign'}}):
        mismatch.append(rejects(lambda altered=altered:panel.require_qualification({**good,**altered},reg,'digest',runtime)))
    check('formal_run_gate_binds_four_passes_native_runtime_format_registration_source_and_owner',all(mismatch),rejected=len(mismatch))

    values=np.array([[[0.,-0.],[.12345678912345678,1.]]],dtype=np.float64)
    old={'T':values,'u':values.copy(),'C':values[:,None].repeat(3,axis=1),
        'predicted_raw_e':values+10.,'valid':np.ones_like(values,dtype=bool),'low_signal':np.zeros_like(values,dtype=bool)}
    mapping={'T':'temperature_color_K','u':'template_postblur_e_ref','C':'continuum_postblur_bands',
        'predicted_raw_e':'predicted_raw_e','valid':'valid','low_signal':'low_signal'}
    out={mapping[k]:v.copy() for k,v in old.items()}
    identical=panel.byte_matches(out,old)
    out['predicted_raw_e'][0,0,0]=np.nextafter(out['predicted_raw_e'][0,0,0],np.inf)
    changed=panel.byte_matches(out,old)
    check('qualification_compares_original_six_float64_and_mask_array_bytes_including_one_bit_change',
        all(identical.values()) and not changed['predicted_raw_e'] and sum(changed.values())==5)

    store=FakeStore(root/'flow');events=[];cache=FakeCache(store,'success',events);counters=[];scores=[]
    cell=planned[4]
    def action(): events.append('inverse');return {'numerical_valid':False,'info':{'finite_output':True}}
    def count(*args): counters.append(args);events.append('count')
    def score(out):
        events.append('external_truth_score');scores.append(True)
        if len(scores)==1: raise RuntimeError('Injected external score-only failure')
        return {'status':'FINITE_NUMERICALLY_UNCERTIFIED','metrics':{'domain_pixels':4,'template_RMSE_e':1.},
            'production_numerical_certified':False}
    arguments=dict(cache=cache,raw='raw-fixture',action=action,score_action=score,count_call=count,
        store=store,digest='fixture-registration')
    first=panel.evaluate_new_cell(cell,**arguments)
    second=panel.evaluate_new_cell(cell,**arguments)
    third=panel.evaluate_new_cell(cell,**arguments)
    check('count_and_inverse_once_parameters_and_bytes_before_truth_score_then_score_only_retry',
        len(counters)==1 and events.count('inverse')==1 and len(scores)==2
        and first['score_status']=='SCORING_FAILED' and second['score_status']=='SCORED'
        and not second['production_numerical_certified'] and second['finite_output']
        and events.index('parameters_persisted')<events.index('canonical_byte_verified')<events.index('external_truth_score')
        and events.index('begin')<events.index('count')<events.index('inverse'),events=events)
    check('cached_score_reload_does_not_recount_or_rescore_and_still_verifies_readback',
        third['new_inverse_calls_this_run']==0 and third['new_inverse_calls_panel_total']==1
        and third['inference_reused'] and third['score_reused'] and len(scores)==2
        and events[-1]=='load_for_score')
    cache.break_readback=True
    damaged=panel.evaluate_new_cell(cell,**arguments)
    check('later_compact_readback_failure_blocks_cached_scores_without_refit',
        damaged['controller_stop'] and damaged['score_status']=='NOT_SCOREABLE'
        and damaged['new_inverse_calls_this_run']==0 and events.count('inverse')==1)

    stopped_store=FakeStore(root/'budget');stopped_store.guard.failure='E below independent floor'
    stopped_events=[];stopped_cache=FakeCache(stopped_store,'blocked',stopped_events)
    stopped=panel.evaluate_new_cell(planned[5],cache=stopped_cache,raw='raw',action=lambda:1,
        score_action=lambda out:1,count_call=lambda *args:stopped_events.append('count'),
        store=stopped_store,digest='floor-fixture')
    check('live_budget_failure_precedes_ticket_count_and_inference',
        stopped['controller_stop'] and stopped['status']=='CACHE_PRE_INFERENCE_FAILED'
        and not stopped_events and stopped['new_inverse_calls_this_run']==0)

    count_store=FakeStore(root/'count_error');count_events=[];count_cache=FakeCache(count_store,'blocked',count_events)
    def refused_count(*args): raise RuntimeError('Global formal-call cap already reached')
    refused=panel.evaluate_new_cell(planned[5],cache=count_cache,raw='raw',
        action=lambda:count_events.append('forbidden-inverse'),score_action=lambda out:None,
        count_call=refused_count,store=count_store,digest='count-fixture')
    check('global_call_counter_failure_stops_panel_before_the_callback_without_consuming_another_cell',
        refused['controller_stop'] and not refused['inference_attempted']
        and not refused['inference_callback_started'] and refused['new_inverse_calls_this_run']==0
        and 'forbidden-inverse' not in count_events)

    returned_store=FakeStore(root/'score_returned_failure');returned_events=[]
    returned_cache=FakeCache(returned_store,'retryable',returned_events);returned_scores=[]
    def returned_score(out):
        returned_scores.append(True)
        if len(returned_scores)==1: return {'status':'SCORING_FAILED','metrics':None}
        return {'status':'OUTSIDE_SCIENTIFIC_ACCEPTANCE','metrics':{'template_RMSE_e':1000.}}
    returned_args=dict(cache=returned_cache,raw='raw',action=lambda:{'numerical_valid':True},
        score_action=returned_score,count_call=lambda *args:returned_events.append('count'),
        store=returned_store,digest='returned-score-fixture')
    returned_first=panel.evaluate_new_cell(planned[5],**returned_args)
    returned_second=panel.evaluate_new_cell(planned[5],**returned_args)
    check('explicit_score_execution_failure_retries_only_score_and_scientific_failure_stays_in_the_denominator',
        returned_first['score_status']=='SCORING_FAILED' and returned_second['score_status']=='SCORED'
        and returned_second['external_score']['status']=='OUTSIDE_SCIENTIFIC_ACCEPTANCE'
        and returned_second['metrics']['template_RMSE_e']==1000.
        and returned_events.count('count')==returned_events.count('score_receipt')==1
        and len(returned_scores)==2)

    failed_store=FakeStore(root/'inverse_error');failed_events=[];failed_cache=FakeCache(failed_store,'failure',failed_events)
    counted=[]
    def raising(): failed_events.append('inverse');raise ArithmeticError('No finite profile iterate')
    args=dict(cache=failed_cache,raw='raw',action=raising,score_action=lambda out:1,
        count_call=lambda *a:counted.append(a),store=failed_store,digest='failed-inverse')
    error1=panel.evaluate_new_cell(planned[6],**args);error2=panel.evaluate_new_cell(planned[6],**args)
    check('failed_inference_is_durable_in_denominator_and_never_gets_a_second_inverse',
        error1['status']=='INFERENCE_FAILED_NO_RERUN' and error2['new_inverse_calls_this_run']==0
        and error2['new_inverse_calls_panel_total']==1 and len(counted)==1 and failed_events.count('inverse')==1)

    fallback_store=FakeStore(root/'fallback');fallback_events=[]
    failure={'status':'FAIL_CLOSED_INITIAL_BYTE_MISMATCH','score_allowed':False,
        'original_fallback_available':True,'lossless_representation_verified':False}
    fallback=FakeCache(fallback_store,'original_saved',fallback_events,persist_failure=failure)
    received=panel.evaluate_new_cell(planned[7],cache=fallback,raw='raw',action=lambda:{'numerical_valid':True},
        score_action=lambda out:fallback_events.append('forbidden-score'),count_call=lambda *a:None,
        store=fallback_store,digest='fallback-fixture')
    check('original_full_fallback_is_preserved_but_representation_failure_forbids_scoring_and_stops',
        received['finite_output'] and received['original_fallback_available'] and received['controller_stop']
        and not received['production_numerical_certified'] and received['inference_numerical_certified']
        and 'forbidden-score' not in fallback_events)
    unavailable_store=FakeStore(root/'unavailable');unavailable_events=[]
    unavailable=FakeCache(unavailable_store,'no_persistence',unavailable_events,persist_failure={
        'status':'FAIL_CLOSED_ORIGINAL_PERSISTENCE_UNAVAILABLE','score_allowed':False,
        'original_fallback_available':False,'durable_failure_receipt_written':False})
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):
        unavailable_row=panel.evaluate_new_cell(planned[8],cache=unavailable,raw='raw',
            action=lambda:{'numerical_valid':False},score_action=lambda out:None,
            count_call=lambda *a:None,store=unavailable_store,digest='unavailable-fixture')
    check('unexpected_floor_drop_prints_memory_receipt_does_not_claim_durable_finite_output_and_stops',
        unavailable_row['controller_stop'] and not unavailable_row['finite_output']
        and unavailable_row['original_finite_endpoint_not_durably_preserved']
        and 'FAIL_CLOSED_ORIGINAL_PERSISTENCE_UNAVAILABLE' in capture.getvalue())

    class UnwritableStore(FakeStore):
        def write(self,*args,**kwargs): raise RuntimeError('Strict E floor blocks all durable writes')
    unwritable=UnwritableStore(root/'terminal_write_error');captured=io.StringIO()
    with contextlib.redirect_stdout(captured):
        unwritten=panel.run_receipt(unwritable,unwritable.report_root/'PANEL_RESULT.json',
            {'status':'C1V3_SMALL_PANEL_COMPLETE','summary':panel.summary(planned,reused)},
            failure_status='PANEL_TERMINAL_RECEIPT_FAILED')
    check('receipt_write_failure_prints_full_twelve_denominator_without_claiming_durability_or_allowing_refit',
        unwritten['status']=='PANEL_TERMINAL_RECEIPT_FAILED'
        and not unwritten['durable_failure_receipt_written'] and not unwritten['inference_rerun_permitted']
        and unwritten['summary']['planned_cells']==12
        and len(unwritten['summary']['cells'])==12 and 'PANEL_TERMINAL_RECEIPT_FAILED' in captured.getvalue()
        and not (unwritable.report_root/'PANEL_RESULT.json').exists())

    seals=FakeStore(root/'sealed');path=seals.report_root/'record.json'
    panel.seal(seals,path,{'fixed':True});equal=panel.read_sealed(seals,path)=={'fixed':True}
    path.write_text('{"fixed":false}',encoding='utf8')
    check('sealed_registration_and_receipt_body_tampering_fails_closed',
        equal and rejects(lambda:panel.read_sealed(seals,path)))
    locks=FakeStore(root/'locks');a=locks.report_root/'lock_a.json';b=locks.report_root/'lock_b.json'
    with panel.phase_locks(locks,'fixture','CPU_lock_fixture',paths=(a,b)):
        exclusive=rejects(lambda:_enter_locks(locks,a))
    released=not a.exists() and not b.exists()
    with panel.phase_locks(locks,'fixture','CPU_lock_fixture',paths=(a,)):
        foreign={'pid':-1,'owner_token':'foreign'};a.write_text(json.dumps(foreign),encoding='utf8')
    check('phase_locks_are_exclusive_owner_bound_release_only_own_and_preserve_replacement',
        exclusive and released and json.loads(a.read_text())==foreign)

    all_hashes={name:s.sha(s.ROOT/name) for name in panel.ADDITIONAL_SOURCES}
    binding=FakeStore(root/'contract_bindings')
    contract_path=binding.report_root/'CPU_CONTRACT.json'
    binding.write(contract_path,{'all_passed':True,'D_persistence_contracts_executed':True,
        'cache_schema':compact.SCHEMA,'source_sha256':all_hashes})
    receipts,bound=panel._contract_sources([contract_path])
    binding.write(binding.report_root/'NO_WRITE.json',{'all_passed':True,'D_persistence_contracts_executed':False,
        'cache_schema':compact.SCHEMA,'source_sha256':all_hashes})
    bad=dict(all_hashes);bad['research_closeout/c1_small_panel.py']='0'*64
    binding.write(binding.report_root/'BAD_SOURCE.json',{'all_passed':True,'D_persistence_contracts_executed':True,
        'cache_schema':compact.SCHEMA,'source_sha256':bad})
    check('register_freezes_current_controller_store_format_and_entry_CPU_sources_and_rejects_no_write_only_or_stale',
        set(bound)==set(panel.ADDITIONAL_SOURCES) and len(receipts)==1
        and rejects(lambda:panel._contract_sources([binding.report_root/'NO_WRITE.json']))
        and rejects(lambda:panel._contract_sources([binding.report_root/'BAD_SOURCE.json'])))

    tree=ast.parse(inspect.getsource(panel.qualify))
    called_names=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            if isinstance(node.func,ast.Name): called_names.append(node.func.id)
            elif isinstance(node.func,ast.Attribute): called_names.append(node.func.attr)
    prohibited={'repaired_infer','minimize','solve','count_call','initialize','cached_v3'}
    check('native_qualification_code_has_only_saved_parameters_and_forward_cache_operations',
        not prohibited.intersection(called_names) and 'forward_reconstruct' in called_names
        and 'load_parameters' in called_names and 'persist' in called_names,
        prohibited_calls_found=sorted(prohibited.intersection(called_names)))
    check('D_roots_budgets_entry_and_no_historical_detection_transfer_are_explicit',
        str(panel.REPORT_DIR).replace('\\','/')=='D:/CodexWork/thermometry/closeout_001/reports/repair_001/quick_v003'
        and str(panel.DATA_DIR).replace('\\','/')=='D:/CodexWork/thermometry/closeout_001/outputs/repair_001/quick_v003'
        and panel.CONFIG['D_project_limit_bytes']==256*2**20 and panel.CONFIG['D_free_reserve_bytes']==128*2**20
        and panel.CONFIG['E_free_floor_bytes']==2*2**30 and panel.CONFIG['global_E_D_new_study_limit_bytes']==4*2**30
        and panel.CONFIG['detector'] is None and not panel.CONFIG['historical_v3_threshold_transferred']
        and 'research_closeout/runtime_entry.py' in panel.ADDITIONAL_SOURCES)

    protected_after=protected_snapshot()
    unchanged=protected_before==protected_after
    check('live_frozen_v002_sources_old_results_and_actual_formal_call_ledger_are_byte_unchanged',
        unchanged,protected_files=len(protected_before))
    result={'time':s.now(),'all_passed':all(row['passed'] for row in checks),'tests':checks,
        'scope':'isolated E CPU fake-cache/inference/score and metadata contracts for preparation; no production D write, native GPU qualification or formal inverse',
        'fixture_root':str(root),'source_sha256':{name:s.sha(s.ROOT/name) for name in
            ('research_closeout/c1_small_panel.py','research_closeout/tests/c1_small_panel.py')},
        'execution_entry':entry,'formal_inverse_calls':0,'GPU_used':False,'D_writes':0,
        'native_GPU_qualification_performed':False,'D_persistence_contracts_executed':False,
        'register_or_run_executed':False,'actual_new_inverse_call_ledger_unchanged':unchanged,
        'protected_file_sha256_before':protected_before,'protected_file_sha256_after':protected_after}
    destination=Path(output) if output else panel.CPU_REPORT_DIR/('PANEL_CONTROLLER_CONTRACTS_'+token+'.json')
    s.write(destination,result)
    print(json.dumps({'receipt':str(destination),'sha256':s.sha(destination),**result},ensure_ascii=True,indent=2),flush=True)
    return result


def _enter_locks(store,path):
    with panel.phase_locks(store,'other','other',paths=(path,)): pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    raise SystemExit(0 if run(parser.parse_args().output)['all_passed'] else 2)
