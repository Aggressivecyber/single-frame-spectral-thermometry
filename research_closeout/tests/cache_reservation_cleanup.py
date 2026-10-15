"""Tiny billed D metadata fixtures; never apply cleanup to production caches."""
from .. import storage as s
from .. import explicit_store as e
from .. import cache_reservation_cleanup as cleanup
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import time
import uuid
from unittest.mock import patch


def run():
    started=time.perf_counter();checks=[];fixtures=[]
    registration_paths=(s.REPORTS/'repair_001/v3_tangent_probe_002/REGISTRATION.json',
        s.REPORTS/'repair_001/quick_v003/PANEL_REGISTRATION.json')
    frozen={}
    for path in registration_paths: frozen.update(s.read(path)['source_sha256'])
    before={name:s.sha(s.ROOT/name) for name in frozen}
    prior_paths=registration_paths+(s.REPORTS/'repair_001/v3_tangent_probe_002/RESULT.json',
        s.REPORTS/'repair_001/v3_preconditioner_003/CONTRACTS.json')
    prior={str(path):s.sha(path) for path in prior_paths}
    def check(name,passed,**evidence): checks.append({'name':name,'passed':bool(passed),**evidence})
    def rejects(fn):
        try: fn()
        except (cleanup.CleanupIntegrityError,e.StoreIntegrityError,OSError,ValueError,KeyError): return True
        return False
    def write(store,path,value=None,*,payload=None,sealed=False):
        path=e._checked_path(path,store.roots);path.parent.mkdir(parents=True,exist_ok=True)
        data=payload if payload is not None else (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
        with path.open('wb') as handle: handle.write(data);handle.flush();os.fsync(handle.fileno())
        if sealed:
            side=e._checked_path(path.with_name(path.name+'.sha256.json'),store.roots)
            side.write_text(json.dumps(cleanup._ref(path)),encoding='utf8')
        return cleanup._ref(path)
    def fixture(namespace='qualification',*,explicit_result_owner=False):
        token=uuid.uuid4().hex;store=e.ExplicitDStore('reserve_cleanup_cpu_'+token,owner_token=token)
        store.budget().check(64*2**10)
        store.write_path(store.data_root/'fixture_enrollment_guard')
        owner={'pid':os.getpid(),'token':uuid.uuid4().hex}
        identity={'registration_sha256':s.canonical(token),'method':'metadata_fixture','case_id':token}
        context={'schema':cleanup.CACHE_SCHEMA,'identity':identity,'input_reference':{'sha256':'a'*64},
            'calibration':{'fingerprint':'b'*64},'config':{'fixture_only':True},
            'frame_plan':{'fallback_zip_bound_bytes':256},'storage_contract':store.contract()}
        fingerprint=s.canonical(context)
        key=s.canonical({'identity':identity,'raw_sha256':'a'*64,'calibration':'b'*64,'config':context['config']})
        data=store.data_root/namespace/'cells'/key;report=store.report_root/namespace/'cells'/key
        reserve=data/('FALLBACK_RESERVED_'+owner['token']+'.bin')
        reserve_ref=write(store,reserve,payload=bytes(256))
        base={'schema':cleanup.CACHE_SCHEMA,'cache_key':key,'context_fingerprint':fingerprint}
        descriptor={'dtype':'float64','shape':[1,1,1],'nbytes':8,'sha256':'d'*64}
        canonical={name:{**descriptor,'dtype':'bool' if name in ('valid','low_signal') else 'float64'} for name in cleanup.CANONICAL}
        prepared={**base,'context':context,'canonical_arrays':canonical,'continuum_latent':descriptor}
        prepared_ref=write(store,report/'PREPARED.json',prepared,sealed=True)
        param_ref=write(store,data/'PARAMETERS.npz',payload=b'opaque-parameter-fixture')
        parameters={**base,'prepared':prepared_ref,'parameter_cache':param_ref}
        parameter_receipt=write(store,report/'PARAMETERS.json',parameters,sealed=True)
        verification={'all_byte_identical':True,'per_array_byte_identical':{name:True for name in cleanup.CANONICAL+('continuum_latent',)},
            'forward_only':True,'complete_inverse_calls':0,'optimizer_calls':0,'linear_solver_calls':0,
            'reconstructed_canonical_arrays':canonical,'reconstructed_continuum_latent':descriptor}
        result={**base,'status':'LOSSLESS_READY','lossless_representation_verified':True,
            'original_fallback_available':False,'fallback_retained':False,'verification':verification,
            'parameter_receipt':parameter_receipt}
        if explicit_result_owner: result['owner']=owner
        records={'STARTED.json':{**base,'status':'STARTED','owner':owner,'context':context,
                'reserve_path':str(reserve),'reserve_bytes':256},
            'RESERVATION.json':{**base,'owner':owner,'reservation':reserve_ref},'RESULT.json':result}
        for name,value in records.items(): write(store,report/name,value,sealed=True)
        write(store,report/'RESERVATION_RELEASE_FAILURE.json',{**base,'status':'OWN_RESERVATION_NOT_RELEASED',
            'error':'Only a compact cache instance reservation can be released'},sealed=True)
        if namespace=='qualification':
            cell={'case_id':identity['case_id'],'method':identity['method'],'passed':True,
                'compact_report_root':str(report),'compact_data_root':str(data),
                'original_canonical_byte_matches':{name:True for name in cleanup.ORIGINAL_SIX},
                'compact_readback_original_byte_matches':{name:True for name in cleanup.ORIGINAL_SIX}}
            cells=[cell]+[{**cell,'case_id':'other_'+str(index),'compact_report_root':'not-this-instance'} for index in range(3)]
            terminal=store.report_root/'NATIVE_COMPACT_QUALIFICATION.json'
            phase={'registration_sha256':identity['registration_sha256'],'planned_existing_endpoints':4,
                'audited_existing_endpoints':4,'passed_existing_endpoints':4,'all_passed':True,'cells':cells}
        else:
            terminal=store.report_root/'PANEL_RESULT.json'
            cells=[{'case_id':identity['case_id'],'method':identity['method']}]+[{'case_id':'other_'+str(index),'method':'metadata_fixture'} for index in range(11)]
            phase={'registration_sha256':identity['registration_sha256'],'status':'C1V3_SMALL_PANEL_COMPLETE',
                'summary':{'planned_cells':12,'returned_cell_rows':12,'cells':cells}}
        write(store,terminal,phase,sealed=True)
        value={'store':store,'data':data,'report':report,'reserve':reserve,'owner':owner,'context':fingerprint,
            'terminal':terminal,'records':records,'phase':phase,'key':key}
        fixtures.append(value);return value
    def prepare(f,**kwargs):
        return cleanup.prepare(f['store'],kwargs.pop('reservation',f['reserve']),owner_token=kwargs.pop('owner_token',f['owner']['token']),
            context_fingerprint=kwargs.pop('context',f['context']),phase_terminal=kwargs.pop('terminal',f['terminal']),**kwargs)
    def release(f,prepared,**kwargs):
        return cleanup.release(f['store'],prepared['plan_ref']['path'],owner_token=kwargs.pop('owner_token',f['owner']['token']),
            context_fingerprint=kwargs.pop('context',f['context']),**kwargs)

    for namespace in ('qualification','inference'):
        f=fixture(namespace,explicit_result_owner=namespace=='inference');store=f['store']
        old_files={str(path):s.sha(path) for root in store.roots for path in root.rglob('*') if path.is_file() and path!=f['reserve']}
        check(namespace+'_frozen_registered_release_still_rejects_namespace',
            rejects(lambda:store.release_reservation(f['reserve'],owner=f['owner'])) and f['reserve'].exists())
        plan=prepare(f)
        check(namespace+'_plan_is_durable_and_zero_reserve_still_exists_before_release',
            f['reserve'].exists() and Path(plan['plan_ref']['path']).is_file()
            and Path(plan['plan_ref']['path']+'.sha256.json').is_file())
        result=release(f,plan)
        remaining={str(path):s.sha(path) for root in store.roots for path in root.rglob('*') if path.is_file()}
        check(namespace+'_single_zero_reserve_removed_and_every_other_D_byte_unchanged',
            result['status']=='RELEASED_ZERO_RESERVATION' and result['unlink_count']==1
            and not f['reserve'].exists() and old_files==remaining)
        before_repeat={str(path):s.sha(path) for path in Path(plan['plan_ref']['path']).parent.rglob('*') if path.is_file()}
        with patch.object(Path,'unlink',side_effect=AssertionError('Repeated success cannot unlink anything')):
            repeated=release(f,plan)
        after_repeat={str(path):s.sha(path) for path in Path(plan['plan_ref']['path']).parent.rglob('*') if path.is_file()}
        check(namespace+'_repeat_reuses_same_success_without_new_writes_or_any_unlink',repeated==result
            and before_repeat==after_repeat and rejects(lambda:release(f,plan,owner_token='f'*32))
            and rejects(lambda:release(f,plan,context='f'*64)))

    f=fixture();store=f['store']
    wrong_paths=(store.data_root/'other'/'cells'/f['key']/f['reserve'].name,
        store.data_root/'qualification'/'cells'/'not64hex'/f['reserve'].name,
        f['data']/'FALLBACK.npz',f['data']/('FALLBACK_RESERVED_'+'F'*32+'.bin'),
        e.DATA/'other_task'/'qualification'/'cells'/f['key']/f['reserve'].name)
    check('wrong_namespace_key_owner_filename_fallback_and_other_task_are_rejected',
        all(rejects(lambda path=path:prepare(f,reservation=path)) for path in wrong_paths)
        and rejects(lambda:prepare(f,owner_token='f'*32)) and rejects(lambda:prepare(f,context='f'*64))
        and f['reserve'].exists())
    guard_fixture=f
    def bad_record(name,mutate,*,reseal=True,label):
        value=copy.deepcopy(guard_fixture['records'][name]);mutate(value)
        write(guard_fixture['store'],guard_fixture['report']/name,value,sealed=reseal)
        check(label,rejects(lambda:prepare(guard_fixture)) and guard_fixture['reserve'].exists())
        write(guard_fixture['store'],guard_fixture['report']/name,guard_fixture['records'][name],sealed=True)
    bad_record('STARTED.json',lambda v:v['owner'].update(token='f'*32),label='different_started_reserved_owner_is_rejected')
    bad_record('RESULT.json',lambda v:v.update(owner={'pid':os.getpid(),'token':'f'*32}),label='explicit_different_RESULT_owner_is_rejected')
    bad_record('RESULT.json',lambda v:v.update(cache_key='f'*64),label='mismatched_sealed_key_is_rejected')
    bad_record('RESULT.json',lambda v:v.update(context_fingerprint='f'*64),label='mismatched_sealed_context_is_rejected')
    bad_record('RESULT.json',lambda v:v.update(status='FAIL_CLOSED'),label='non_ready_result_is_rejected')
    bad_record('RESULT.json',lambda v:v.update(lossless_representation_verified=False),label='unverified_result_is_rejected')
    bad_record('RESULT.json',lambda v:v['verification']['per_array_byte_identical'].update(valid=False),label='one_failed_original_array_byte_match_is_rejected')
    bad_record('RESULT.json',lambda v:v.update(parameter_receipt={'path':'D:/outside','sha256':'f'*64,'bytes':1}),label='different_parameter_receipt_is_rejected')
    bad_record('RESULT.json',lambda v:v.update(status='LOSSLESS_READY_CHANGED'),reseal=False,label='changed_body_with_original_seal_is_rejected')
    f=guard_fixture;write(f['store'],f['reserve'],payload=bytes(255)+b'x')
    check('nonzero_reservation_is_rejected_without_deletion',rejects(lambda:prepare(f)) and f['reserve'].exists())
    write(f['store'],f['reserve'],payload=bytes(256))
    value=copy.deepcopy(f['records']['RESERVATION.json']);value['reservation']['sha256']='f'*64
    write(f['store'],f['report']/'RESERVATION.json',value,sealed=True)
    check('sealed_but_wrong_reservation_hash_is_rejected',rejects(lambda:prepare(f)) and f['reserve'].exists())
    write(f['store'],f['report']/'RESERVATION.json',f['records']['RESERVATION.json'],sealed=True)
    f=fixture();write(f['store'],f['data']/'FALLBACK.npz',payload=b'real-fallback-protected')
    fallback_sha=s.sha(f['data']/'FALLBACK.npz')
    check('actual_fallback_sibling_keeps_entire_instance_untouched',rejects(lambda:prepare(f))
        and f['reserve'].exists() and s.sha(f['data']/'FALLBACK.npz')==fallback_sha)
    f=fixture();write(f['store'],f['terminal'],{**f['phase'],'all_passed':False},sealed=True)
    check('nonterminal_or_failed_namespace_phase_is_rejected',rejects(lambda:prepare(f)) and f['reserve'].exists())
    f=fixture();write(f['store'],f['store'].report_root/'PHASE_ACTIVE.json',{'fixture_active':True})
    check('active_owned_task_phase_is_rejected',rejects(lambda:prepare(f)) and f['reserve'].exists())
    f=fixture();plan=prepare(f);write(f['store'],f['reserve'],payload=bytes(255)+b'Y')
    check('reserve_change_after_plan_is_rejected_before_unlink',rejects(lambda:release(f,plan)) and f['reserve'].exists())
    f=fixture();plan=prepare(f)
    attempt=Path(plan['plan_ref']['path']).parent/'ATTEMPT.json'
    cleanup._seal(attempt,{'fixture_interrupted':True})
    check('interrupted_attempt_is_not_retried_or_inferred_successful',rejects(lambda:release(f,plan)) and f['reserve'].exists())
    f=fixture();plan=prepare(f)
    with patch.object(Path,'unlink',side_effect=OSError('fixture unlink denied')):
        check('unlink_failure_keeps_placeholder_and_does_not_publish_success',rejects(lambda:release(f,plan)))
    check('unlink_failure_has_durable_plan_attempt_but_no_success_receipt',f['reserve'].exists()
        and (Path(plan['plan_ref']['path']).parent/'ATTEMPT.json').exists()
        and not (Path(plan['plan_ref']['path']).parent/'RECEIPT.json').exists())

    # The destructive source path is structurally one explicit unlink; no
    #glob/removal traversal, raw loader, physics, inverse or solver call exists.
    tree=ast.parse((s.ROOT/cleanup.SOURCE).read_text(encoding='utf8'))
    calls=[node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id
        for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,(ast.Attribute,ast.Name))]
    check('helper_has_only_one_explicit_unlink_and_no_recursive_glob_solver_or_array_loader',
        calls.count('unlink')==1 and not set(calls)&{'glob','rglob','rmtree','remove','arrays','load','load_raw',
            'forward_reconstruct','repaired_infer','refined_infer','minimize','lsmr','count_call'})
    after={name:s.sha(s.ROOT/name) for name in frozen}
    check('all46_frozen_sources_and_prior_receipt_bytes_remain_unchanged',len(frozen)==46 and before==after==frozen
        and prior=={str(path):s.sha(path) for path in prior_paths})
    fixture_bytes=sum(path.stat().st_size for f in fixtures for root in f['store'].roots
        for path in root.rglob('*') if path.is_file())
    check('fixtures_are_small_billed_D_files_and_never_production_tasks',fixture_bytes<2**20
        and all(f['store'].task_name.startswith('reserve_cleanup_cpu_') for f in fixtures),fixture_bytes=fixture_bytes)
    value={'schema':cleanup.VERSION+'-CPU-contracts','time':s.now(),'all_passed':all(v['passed'] for v in checks),
        'checks':checks,'source_sha256':{name:s.sha(s.ROOT/name) for name in (cleanup.SOURCE,cleanup.TEST_SOURCE)},
        'frozen_source_sha256':after,'prior_receipt_sha256':prior,'fixture_bytes':fixture_bytes,
        'fixture_stores':[f['store'].contract() for f in fixtures],
        'formal_cache_cleanup_performed':False,'GPU_used':False,'forward_calls':0,'whole_inverse_calls':0,
        'truth_used':False,'fixture_parameter_payload_is_opaque_metadata_standin':True,
        'CPU_wall_s':time.perf_counter()-started}
    cleanup.REPORT.mkdir(parents=True,exist_ok=True)
    number=len(list(cleanup.REPORT.glob('CONTRACTS_ATTEMPT_*.json')))+1
    s.write(cleanup.REPORT/f'CONTRACTS_ATTEMPT_{number:03d}.json',value)
    if value['all_passed']: s.write(cleanup.REPORT/'CONTRACTS.json',value,replace=(cleanup.REPORT/'CONTRACTS.json').exists())
    print(json.dumps({'all_passed':value['all_passed'],'checks':len(checks),'failures':[v for v in checks if not v['passed']],
        'source_sha256':value['source_sha256'],'fixture_bytes':fixture_bytes,'CPU_wall_s':value['CPU_wall_s']},indent=2),flush=True)
    return value


if __name__=='__main__': raise SystemExit(0 if run()['all_passed'] else 2)
