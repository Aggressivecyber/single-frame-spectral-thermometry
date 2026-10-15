"""Prepared six-raw C1v3 regression panel: four reused cells, eight new calls.

Register first.  Qualify the new compact representation on the four already
saved v002 endpoints in a separate native-GPU, forward-only operation.  Only
then may root explicitly run the eight new inversions.  No method selection,
new regularizer grid, historical detector threshold or final data is involved.
"""
from . import storage as s
from . import quick_validation as q
from . import calibration_quick as c2
import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import traceback
import uuid
import numpy as np

VERSION='C1v3-small-regression-panel-v003'
TASK='quick_v003'
STORE_OWNER_TOKEN='c10c003f02b54855b62e9058eab57230'
D_ROOT=Path('D:/CodexWork/thermometry/closeout_001')
REPORT_DIR=D_ROOT/'reports/repair_001'/TASK
DATA_DIR=D_ROOT/'outputs/repair_001'/TASK
CPU_REPORT_DIR=s.REPORTS/'repair_001/quick_v003_cpu_contracts'
RUNTIME_CONTRACT=s.REPORTS/'repair_001/runtime_entry_001/CONTRACTS.json'
REGISTRATION_NAME='PANEL_REGISTRATION.json'
QUALIFICATION_NAME='NATIVE_COMPACT_QUALIFICATION.json'
RESULT_NAME='PANEL_RESULT.json'
ROUTES=tuple(copy.deepcopy(c2.ROUTES))
CONFIG={'version':VERSION,'physical_options':copy.deepcopy(q.PHYSICAL_OPTIONS),
    'numerical_options':{'cost_limit_s':180.},'routes':list(ROUTES),
    'native_shape':[256,256],'grid':256,'photon_ratio':1.,'c1_multiplier':1.,'template_multiplier':1.,
    'planned_raws':6,'planned_cells':12,'reused_v002_cells':4,'maximum_new_inverse_calls':8,
    'new_inverse_wall_budget_s_excluding_overhead':1440,
    'qualification_existing_endpoints':4,'qualification_complete_inverse_calls':0,
    'qualification_requires_same_native_GPU_runtime':True,
    'detector':None,'historical_v3_threshold_transferred':False,'method_selection':False,
    'scientific_nine_regularizer_grid_modified':False,'final_payloads_opened':False,
    'D_project_limit_bytes':256*2**20,'D_free_reserve_bytes':128*2**20,
    'E_free_floor_bytes':2*2**30,'global_E_D_new_study_limit_bytes':4*2**30,
    'global_new_inverse_call_cap':12000,
    'purpose':'debug/regression on reused development raws; no scientific acceptance or independent-test claim'}
ADDITIONAL_SOURCES=(
    'research_closeout/c1_small_panel.py','research_closeout/tests/c1_small_panel.py',
    'research_closeout/compact_joint_cache.py','research_closeout/tests/compact_joint_cache.py',
    'research_closeout/explicit_store.py','research_closeout/tests/explicit_store.py',
    'research_closeout/runtime_entry.py','research_closeout/tests/runtime_entry.py')


class PanelIntegrityError(RuntimeError): pass
class PanelStorageStop(RuntimeError): pass


def execution_entry():
    from .runtime_entry import install_platform_fallback
    return install_platform_fallback()


def production_store():
    from .explicit_store import ExplicitDStore
    store=ExplicitDStore(task_name=TASK,owner_token=STORE_OWNER_TOKEN)
    if Path(store.report_root).resolve()!=REPORT_DIR.resolve() or Path(store.data_root).resolve()!=DATA_DIR.resolve():
        raise PanelIntegrityError('The explicit D store returned another task root')
    return store


def file_ref(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':s.sha(path),'bytes':path.stat().st_size}


def verify_ref(ref):
    path=Path(ref['path'])
    if not path.is_file() or s.sha(path)!=ref['sha256']:
        raise PanelIntegrityError('Frozen file changed or disappeared: '+str(path))
    return path


def seal(store,path,value):
    """Exclusive body plus exact byte sidecar, entirely inside the explicit store."""
    path=store.write_path(path)
    store.write(path,s.plain(value))
    receipt=file_ref(path)
    store.write(path.with_name(path.name+'.sha256.json'),receipt)
    return receipt


def run_receipt(store,path,value,*,failure_status):
    """Retain the full in-memory denominator if a strict floor blocks a receipt."""
    try:
        seal(store,path,value)
    except Exception as error:
        value.update(original_controller_status=value.get('status'),status=failure_status,
            durable_failure_receipt_written=False,receipt_error_type=type(error).__name__,
            receipt_error=str(error),inference_rerun_permitted=False)
        print(json.dumps(s.plain(value),ensure_ascii=True),flush=True)
    return value


def read_sealed(store,path):
    path=store.write_path(path)
    receipt=store.read(path.with_name(path.name+'.sha256.json'))
    if Path(receipt['path']).resolve()!=path.resolve(): raise PanelIntegrityError('SHA sidecar points at another file')
    verify_ref(receipt)
    return store.read(path)


def planned_cells(refs):
    if len(refs)!=6: raise PanelIntegrityError('Exactly six frozen old raw references are required')
    cells=[]
    for i,(ref,(mother,condition)) in enumerate(zip(refs,q.PANEL)):
        if (ref.get('mother_id')!=f'v5_development_selection_mother_{mother:04d}' or ref.get('condition')!=condition
            or ref.get('cohort')!='development' or ref.get('input_kind')!='noise'
            or ref.get('partition_role')!='development_selection' or ref.get('fixture_only')
            or ref['raw_reference'].get('shape')!=[256,256]):
            raise PanelIntegrityError('Panel order, role or unchanged native raw differs')
        for route in ROUTES:
            cells.append({**{key:ref[key] for key in ('case_id','mother_id','condition','input_kind','quick_geometry')},
                'raw_sha256':ref['raw_reference']['sha256'],**route,
                'execution_role':'REUSE_V002' if i<2 else 'NEW_INVERSE'})
    if len({(r['case_id'],r['method']) for r in cells})!=12: raise PanelIntegrityError('Duplicate panel cell')
    return cells


def cell_key(cell,digest):
    return s.canonical({'registration_sha256':digest,'method':cell['method'],'raw_sha256':cell['raw_sha256']})


def identity_for(cell,digest):
    return {'registration_sha256':digest,'case_id':cell['case_id'],'method':cell['method'],
        'raw_sha256':cell['raw_sha256'],'panel_version':VERSION}


def summary(planned,rows):
    supplied={}
    for row in rows:
        key=(row['case_id'],row['method'])
        if key in supplied: raise PanelIntegrityError('Duplicate result cell')
        supplied[key]=row
    expected={(r['case_id'],r['method']) for r in planned}
    if set(supplied)-expected: raise PanelIntegrityError('Unplanned result cell')
    complete=[]
    for cell in planned:
        complete.append(supplied.get((cell['case_id'],cell['method']),{**cell,'status':'NOT_EXECUTED',
            'inference_attempted':False,'finite_output':False,'production_numerical_certified':False,
            'new_inverse_calls_this_run':0,'score_status':'NOT_EXECUTED','metrics':None}))
    return {'planned_raws':6,'planned_cells':len(planned),'returned_cell_rows':len(rows),
        'reused_v002_cells':sum(r.get('execution_role')=='REUSE_V002' and r['status']!='NOT_EXECUTED' for r in complete),
        'new_cells_attempted':sum(bool(r.get('inference_attempted')) and r.get('execution_role')=='NEW_INVERSE' for r in complete),
        'new_inverse_calls_this_run':sum(int(r.get('new_inverse_calls_this_run',0)) for r in complete),
        'new_inverse_calls_panel_total':sum(int(r.get('new_inverse_calls_panel_total',0)) for r in complete),
        'finite_output_cells':sum(bool(r.get('finite_output')) for r in complete),
        'numerically_certified_cells':sum(bool(r.get('production_numerical_certified')) for r in complete),
        'scoring_failed_cells':sum(r.get('score_status')=='SCORING_FAILED' for r in complete),
        'not_executed_cells':sum(r['status']=='NOT_EXECUTED' for r in complete),
        'cells':complete,'all_denominators_preserved':True,'method_selection_performed':False,
        'unique_scientific_sample_size_claimed':False,'final_payloads_opened':False}


def reuse_row(cell,origin,digest):
    for key in ('case_id','method','initializer','raw_sha256'):
        if cell[key]!=origin[key]: raise PanelIntegrityError('Cannot change an existing v002 method/raw/initializer')
    return {**cell,'execution_registration_sha256':digest,'status':origin['status'],
        'finite_output':origin['finite_output'],'production_numerical_certified':origin['production_numerical_certified'],
        'metrics':copy.deepcopy(origin['metrics']),'score_status':origin['score_status'],
        'inference_attempted':False,'inference_reused':True,'score_reused':True,
        'new_inverse_calls_this_run':0,'origin_registration_sha256':origin['registration_sha256'],
        'new_inverse_calls_panel_total':0,
        'origin_cell':copy.deepcopy(origin),'historical_result_relabelled':False}


@contextmanager
def phase_locks(store,digest,phase,*,paths=None):
    owner={'pid':os.getpid(),'owner_token':uuid.uuid4().hex,'phase':phase,
        'registration_sha256':digest,'time':s.now()}
    targets=paths or (s.REPORTS/'PHASE_ACTIVE.json',Path(store.report_root)/'PHASE_ACTIVE.json')
    acquired=[]
    try:
        store.budget().check(0)
        for target in targets:
            path=Path(target).resolve()
            if path== (s.REPORTS/'PHASE_ACTIVE.json').resolve(): s.write_path(path)
            else: store.write_path(path)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x',encoding='utf8',newline='\n') as handle:
                json.dump(owner,handle,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
            acquired.append(path)
        yield owner
    finally:
        for path in reversed(acquired):
            try:
                current=json.loads(path.read_text(encoding='utf8'))
                if current.get('pid')==owner['pid'] and current.get('owner_token')==owner['owner_token']: path.unlink()
            except (FileNotFoundError,json.JSONDecodeError,UnicodeError): pass


def _contract_sources(contract_paths):
    refs=[];sources={};explicit_persistence_passed=False
    for path in contract_paths:
        path=Path(path).resolve();value=s.read(path)
        metadata_passed=(value.get('schema')=='runtime-entry-metadata-contracts-v001'
            and value.get('total',0)>0 and value.get('passed')==value['total']
            and all(row.get('passed') for row in value.get('checks',[])))
        if value.get('all_passed') is not True and not metadata_passed:
            raise PanelIntegrityError('CPU contract did not pass: '+str(path))
        if (value.get('all_passed') is True and value.get('D_persistence_contracts_executed') is True
            and value.get('cache_schema','').endswith('v002-explicit-store')):
            explicit_persistence_passed=True
        refs.append(file_ref(path))
        for name,digest in value.get('source_sha256',{}).items():
            if s.sha(s.ROOT/name)!=digest: raise PanelIntegrityError('CPU contract source changed: '+name)
            if name in sources and sources[name]!=digest: raise PanelIntegrityError('Conflicting CPU source contracts')
            sources[name]=digest
    if not set(ADDITIONAL_SOURCES).issubset(sources):
        raise PanelIntegrityError('Register current panel/controller, explicit-store and compact-format CPU contracts')
    if not explicit_persistence_passed:
        raise PanelIntegrityError('A passed current explicit-D compact persistence CPU contract is required; no-write arithmetic alone does not qualify persistence')
    return refs,sources


def _verified_origins(v002_registration,v002_result):
    expected=c2.planned_cells(v002_registration['references'])
    cells=v002_result['summary']['cells']
    if len(cells)!=4 or {(r['case_id'],r['method']) for r in cells}!={(r['case_id'],r['method']) for r in expected}:
        raise PanelIntegrityError('Retain the complete four-cell v002 terminal result')
    files=[]
    for row in cells:
        if not row.get('status') or row.get('score_status') not in ('SCORED','SCORING_FAILED'):
            raise PanelIntegrityError('v002 origin lacks a terminal inference/scoring state')
        for key in ('inference_receipt','score_receipt','output_cache','output_metadata'):
            if row.get(key): verify_ref(row[key]);files.append(row[key])
        receipt=row['parameter_cache_receipt']
        for key in ('parameter_cache','parameter_metadata'): verify_ref(receipt[key]);files.append(receipt[key])
        identity=c2.identity_for(row,row['registration_sha256'])
        c2.load_parameters(receipt,identity,v002_registration)
        q._load_output(row,identity=identity)
    return cells,files


def register(contract_paths,*,store=None):
    entry_metadata=execution_entry()
    from . import compact_joint_cache as compact
    from .smooth_calibration import physics,contract
    store=production_store() if store is None else store
    destination=Path(store.report_root)/REGISTRATION_NAME
    if destination.exists(): raise FileExistsError('Never replace a panel registration')
    paths=list(contract_paths)
    if RUNTIME_CONTRACT.resolve() not in [Path(p).resolve() for p in paths]: paths.append(RUNTIME_CONTRACT)
    cpu_contracts,contract_hashes=_contract_sources(paths)
    v002,d2=c2.verify_registration();r2=s.read(c2.REPORT_DIR/'QUICK_RESULT.json')
    if r2['registration_sha256']!=d2: raise PanelIntegrityError('v002 terminal registration mismatch')
    v001,d1=q.verify_registration();refs=copy.deepcopy(v001['references']);planned=planned_cells(refs)
    origins,origin_files=_verified_origins(v002,r2)
    if refs[:2]!=v002['references']: raise PanelIntegrityError('v002 origins are not the first two unchanged panel raws')
    if v002['config']['physical_options']!=CONFIG['physical_options'] or v002['config']['numerical_options']!=CONFIG['numerical_options']:
        raise PanelIntegrityError('The C1v3 physical/numerical estimator may not change')
    p=physics('cpu');adapter=s.plain(contract(p))
    if adapter!=v002['calibration_contract']: raise PanelIntegrityError('C1 calibration changed since v002')
    sources={**v002['source_sha256'],**{name:s.sha(s.ROOT/name) for name in ADDITIONAL_SOURCES}}
    for name,digest in contract_hashes.items():
        if name in sources and sources[name]!=digest: raise PanelIntegrityError('Source contract conflict: '+name)
        sources[name]=digest
    for name,digest in v002['source_sha256'].items():
        if sources[name]!=digest or s.sha(s.ROOT/name)!=digest: raise PanelIntegrityError('A v002 source changed: '+name)
    format_value=compact.format_contract(storage=store)
    if not format_value.get('new_D_root_writes_supported') or 'v002' not in format_value['schema']:
        raise PanelIntegrityError('The explicit-store compact v002 format is required')
    provenance=q._v3_provenances(refs)
    dependencies=cpu_contracts+origin_files+[file_ref(q.REGISTRATION),file_ref(q.REGISTRATION_SHA),
        file_ref(c2.REGISTRATION),file_ref(c2.REGISTRATION_SHA),file_ref(c2.REPORT_DIR/'QUICK_RESULT.json'),
        file_ref(q.INVENTORY),file_ref(s.ROOT/'reports/training_v4/run_001/backbone_selection.json'),
        file_ref(s.ROOT/'research_closeout/execution_spec.yaml')]
    s.frozen_bridge()
    from research_v6 import inputs
    for ref in refs:
        raw=inputs.load_raw(ref,p)
        provenance[ref['case_id']].update(q.verify_initializer_raw(ref,raw,provenance[ref['case_id']]))
        dependencies.extend([file_ref(s.LEGACY/ref['raw_reference']['path']),
            provenance[ref['case_id']]['state_cache'],provenance[ref['case_id']]['receipt_journal']])
        values=s.arrays(provenance[ref['case_id']]['state_cache']['path'])
        if set(('state','beta'))-set(values) or not all(np.isfinite(values[k]).all() for k in ('state','beta')):
            raise PanelIntegrityError('Nonfinite or missing frozen initializer parameters')
    deduplicated={ref['path']:ref for ref in dependencies}
    for ref in deduplicated.values(): verify_ref(ref)
    store_contract=store.contract()
    value={'schema':VERSION+'-registration','time':s.now(),'config':copy.deepcopy(CONFIG),
        'source_sha256':sources,'contracts':cpu_contracts,'read_dependencies':list(deduplicated.values()),
        'store_contract':store_contract,'format_contract':format_value,'calibration_contract':adapter,
        'execution_entry_metadata':entry_metadata,'execution_entry_contract':file_ref(RUNTIME_CONTRACT),
        'references':refs,'planned_cells':planned,'v3_initializer_provenance':provenance,
        'v002_registration_sha256':d2,'v001_registration_sha256':d1,'v002_origins':copy.deepcopy(origins),
        'v002_source_sha256':v002['source_sha256'],'objective_fingerprint':v002['objective_fingerprint'],
        'objective_identity_scope':'same frozen C1v3 physical/numerical estimator as v002; controller and compact representation are independently registered',
        'new_complete_inverse_calls':0,'truth_loaded':False,'final_payloads_opened':False}
    value['config_fingerprint']=s.canonical({'config':CONFIG,'sources':sources,'format':format_value,
        'store':store_contract,'v002_registration_sha256':d2})
    store.budget().check(len(json.dumps(value).encode('utf8'))+64*2**10)
    seal(store,destination,value)
    return value,file_ref(destination)['sha256']


def verify_registration(store):
    from . import compact_joint_cache as compact
    path=Path(store.report_root)/REGISTRATION_NAME;value=read_sealed(store,path)
    if value['schema']!=VERSION+'-registration' or value['config']!=CONFIG:
        raise PanelIntegrityError('Panel config differs from registration')
    if value['store_contract']!=store.contract() or value['format_contract']!=compact.format_contract(storage=store):
        raise PanelIntegrityError('Explicit store or compact format changed')
    for name,digest in value['source_sha256'].items():
        if s.sha(s.ROOT/name)!=digest: raise PanelIntegrityError('Registered source changed: '+name)
    for ref in value['read_dependencies']: verify_ref(ref)
    if value['planned_cells']!=planned_cells(value['references']): raise PanelIntegrityError('Panel plan changed')
    fingerprint=s.canonical({'config':CONFIG,'sources':value['source_sha256'],'format':value['format_contract'],
        'store':value['store_contract'],'v002_registration_sha256':value['v002_registration_sha256']})
    if fingerprint!=value['config_fingerprint']: raise PanelIntegrityError('Panel fingerprint differs')
    return value,s.sha(path)


def cache_for(store,p,cell,registration,digest,*,qualification=False):
    from .compact_joint_cache import CompactJointCache
    reference=next(r for r in registration['references'] if r['case_id']==cell['case_id'])
    sub='qualification' if qualification else 'inference'
    return CompactJointCache(Path(store.report_root)/sub,Path(store.data_root)/sub,
        storage=store,p=p,identity=identity_for(cell,digest),input_reference=reference['raw_reference'],
        calibration=registration['calibration_contract'],
        config={'grid':256,'physical_options':CONFIG['physical_options'],'numerical_options':CONFIG['numerical_options'],
            'panel_version':VERSION,'operation':'existing_endpoint_forward_only_qualification' if qualification else 'new_or_resumed_joint_inference'},
        source_sha256=registration['source_sha256'])


def byte_matches(out,old_arrays):
    from .compact_joint_cache import array_sha
    mapping={'T':'temperature_color_K','u':'template_postblur_e_ref','C':'continuum_postblur_bands',
        'predicted_raw_e':'predicted_raw_e','valid':'valid','low_signal':'low_signal'}
    return {name:array_sha(out[key])==array_sha(old_arrays[name]) for name,key in mapping.items()}


def qualification_matrix(origins,action):
    rows=[]
    for origin in origins:
        try:
            row=action(origin)
            rows.append({'case_id':origin['case_id'],'method':origin['method'],**row})
        except Exception as error:
            rows.append({'case_id':origin['case_id'],'method':origin['method'],'passed':False,
                'status':'QUALIFICATION_FAILED_ORIGIN_PRESERVED','error_type':type(error).__name__,
                'error':str(error),'origin_parameter_cache':origin.get('parameter_cache_receipt'),
                'origin_output_cache':origin.get('output_cache')})
    return {'planned_existing_endpoints':4,'audited_existing_endpoints':len(rows),
        'passed_existing_endpoints':sum(bool(row.get('passed')) for row in rows),
        'all_passed':len(rows)==4 and all(row.get('passed') for row in rows),
        'cells':rows,'complete_inverse_calls':0,'optimizer_calls':0,'inner_solver_calls':0,'truth_loaded':False}


def require_qualification(value,registration,digest,runtime):
    if not (value.get('all_passed') and value.get('passed_existing_endpoints')==4
        and value.get('registration_sha256')==digest and value.get('native_GPU') is True
        and value.get('runtime')==runtime and value.get('format_contract')==registration['format_contract']
        and value.get('source_sha256')==registration['source_sha256']
        and value.get('store_contract')==registration['store_contract']):
        raise PanelIntegrityError('An exact-registration, same-runtime4/4 native compact qualification is required before any new inverse')


def qualify(device='cuda',*,store=None):
    if device!='cuda': raise PanelIntegrityError('Native qualification is a separate root-owned GPU operation')
    execution_entry()
    from .smooth_calibration import physics,contract
    from .compact_joint_cache import forward_reconstruct,array,runtime_identity
    import torch
    store=production_store() if store is None else store
    registration,digest=verify_registration(store);path=Path(store.report_root)/QUALIFICATION_NAME
    if path.exists(): return read_sealed(store,path)
    with phase_locks(store,digest,VERSION+'-native-qualification'):
        torch.set_num_threads(2);p=physics(device)
        if p.M.device.type!='cuda' or contract(p)!=registration['calibration_contract']:
            raise PanelIntegrityError('Native GPU C1 calibration differs')
        runtime=runtime_identity(p);s.frozen_bridge()
        from research_v6 import inputs
        origin_registration=s.read(c2.REGISTRATION)
        def action(origin):
            cell=next(cell for cell in registration['planned_cells'] if cell['case_id']==origin['case_id'] and cell['method']==origin['method'])
            reference=next(r for r in registration['references'] if r['case_id']==cell['case_id'])
            raw=inputs.load_raw(reference,p)
            parameters,metadata=c2.load_parameters(origin['parameter_cache_receipt'],
                c2.identity_for(origin,origin['registration_sha256']),origin_registration)
            original=q._load_output(origin,identity=c2.identity_for(origin,origin['registration_sha256']))
            original_arrays=s.arrays(origin['output_cache']['path'])
            out=forward_reconstruct(p,raw,parameters,metadata['inference_info'],grid=256)
            original_match=byte_matches(out,original_arrays)
            # The six original arrays define the comparison target.  Additional
            # cache arrays are deterministic forward derivations, never fits.
            for name,key in (('T','temperature_color_K'),('u','template_postblur_e_ref'),('C','continuum_postblur_bands'),
                             ('predicted_raw_e','predicted_raw_e'),('valid','valid'),('low_signal','low_signal')):
                out[key]=torch.as_tensor(original_arrays[name],device=p.M.device)
            out['numerical_valid']=bool(origin['production_numerical_certified'])
            out['predicted_raw_DN']=out['predicted_raw_e']/p.adc_step+p.cfg['black_level_DN']
            out['standardized_residual']=(out['predicted_raw_e']-p.raw_to_electrons(raw))*m_fixed_weights(p,p.raw_to_electrons(raw)).sqrt()
            cache=cache_for(store,p,cell,registration,digest,qualification=True)
            ticket=cache.begin(raw)
            result=cache.persist(ticket,out,raw) if ticket.get('run_new_inverse') else cache.resume(raw)
            passed=all(original_match.values()) and result.get('lossless_representation_verified') and result.get('score_allowed')
            readback_match={}
            if result.get('score_allowed'):
                readback=cache.load_for_score(raw);readback_match=byte_matches(readback,original_arrays)
                passed=passed and all(readback_match.values())
            return {'passed':bool(passed),'status':'QUALIFIED' if passed else 'QUALIFICATION_FAILED_ORIGIN_PRESERVED',
                'original_canonical_byte_matches':original_match,'compact_readback_original_byte_matches':readback_match,
                'compact_result':result,'compact_report_root':str(cache.report),'compact_data_root':str(cache.data),
                'origin_parameter_cache':origin['parameter_cache_receipt'],'origin_output_cache':origin['output_cache'],
                'cache_owner_ticket_did_not_call_inverse':True,'complete_inverse_calls':0}
        value=qualification_matrix(registration['v002_origins'],action)
        value.update(time=s.now(),registration_sha256=digest,native_GPU=True,runtime=runtime,
            format_contract=registration['format_contract'],source_sha256=registration['source_sha256'],
            store_contract=registration['store_contract'],storage_ledger=store.budget().ledger())
        reused=[reuse_row(cell,next(r for r in registration['v002_origins'] if r['case_id']==cell['case_id'] and r['method']==cell['method']),digest)
            for cell in registration['planned_cells'] if cell['execution_role']=='REUSE_V002']
        value['panel_denominators_before_new_inference']=summary(registration['planned_cells'],reused)
        verify_registration(store);seal(store,path,value)
        return value


def m_fixed_weights(p,raw_e):
    # Separate tiny wrapper keeps the qualification operation visibly forward-only.
    from .measurement import fixed_weights
    return fixed_weights(p,raw_e)


def evaluate_new_cell(cell,*,cache,raw,action,score_action,count_call,store,digest,verify_sources=lambda:None):
    """One-shot inference ticket; all score retries reuse durable parameters."""
    row={**cell,'execution_registration_sha256':digest,'inference_attempted':False,
        'new_inverse_calls_this_run':0,'metrics':None,'finite_output':False,
        'new_inverse_calls_panel_total':0,
        'production_numerical_certified':False,'score_status':'NOT_EXECUTED'}
    report=Path(store.report_root)/'cell_receipts'/cell_key(cell,digest)
    success=report/'SCORE_SUCCESS.json';failure=report/'INFERENCE_FAILURE.json';called=report/'CALL_RECORDED.json'
    for existing in (success,failure):
        if existing.exists():
            previous=copy.deepcopy(read_sealed(store,existing)['row'])
            verify_sources()
            if existing==success:
                try: cache.load_for_score(raw)
                except Exception as error:
                    previous.update(status='CACHED_SCORE_REPRESENTATION_VERIFICATION_FAILED',
                        score_status='NOT_SCOREABLE',production_numerical_certified=False,controller_stop=True,
                        new_inverse_calls_this_run=0,inference_attempted=False,inference_reused=True,
                        error_type=type(error).__name__,error=str(error))
                    return previous
            previous.update(new_inverse_calls_this_run=0,inference_attempted=False,inference_reused=True,
                score_reused=existing==success,previous_attempt_receipt=file_ref(existing))
            return previous
    try:
        verify_sources();store.budget().check(0)
        ticket=cache.begin(raw)
    except Exception as error:
        row.update(status='CACHE_PRE_INFERENCE_FAILED',score_status='NOT_SCOREABLE',controller_stop=True,
            error_type=type(error).__name__,error=str(error))
        return row
    original_out=None
    if ticket.get('run_new_inverse'):
        callback_started=False
        try:
            count_call('C1v3_small_panel_v003',cell['method'],cell['case_id'])
            row.update(new_inverse_calls_this_run=1,new_inverse_calls_panel_total=1,
                external_call_count_recorded=True)
            seal(store,called,{'time':s.now(),'identity':identity_for(cell,digest),
                'external_global_count_recorded':True,'maximum_one_new_call_for_this_cache':True})
            callback_started=True
            row.update(inference_attempted=True,inference_callback_started=True)
            original_out=action()
        except Exception as error:
            row.update(status='INFERENCE_FAILED_NO_RERUN',score_status='NOT_SCOREABLE',
                error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                compact_status=cache.inspect(),finite_output=False,inference_callback_started=callback_started,
                controller_stop=not callback_started)
            try:
                seal(store,failure,{'time':s.now(),'row':row,'one_shot_cache_ticket':ticket,
                    'rerun_inverse_permitted':False})
            except Exception as persistence_error:
                row.update(controller_stop=True,durable_failure_receipt_written=False,
                    receipt_error=str(persistence_error))
                print(json.dumps(s.plain(row),ensure_ascii=True),flush=True)
            return row
        try:
            result=cache.persist(ticket,original_out,raw)
        except Exception as error:
            row.update(status='PARAMETER_PERSISTENCE_FAILED_ENDPOINT_PRESERVATION_REQUIRED',
                score_status='NOT_SCOREABLE',controller_stop=True,error_type=type(error).__name__,error=str(error),
                original_endpoint_returned=True,inference_numerical_certified=bool(original_out.get('numerical_valid',False)),
                compact_status=cache.inspect())
            print(json.dumps(s.plain(row),ensure_ascii=True),flush=True)
            return row
        row['inference_numerical_certified']=bool(original_out.get('numerical_valid',False))
    else:
        result=cache.resume(raw)
        row.update(inference_reused=True,new_inverse_calls_panel_total=int(called.exists()))
        if called.exists(): read_sealed(store,called)
    row.update(compact_result=result,compact_report_root=str(cache.report),compact_data_root=str(cache.data))
    if result.get('status')!='LOSSLESS_READY' or not result.get('score_allowed'):
        row.update(status='FINITE_PRESERVED_REPRESENTATION_FAIL_CLOSED' if result.get('original_fallback_available') else 'INFERENCE_INTERRUPTED_OR_FAIL_CLOSED',
            score_status='NOT_SCOREABLE',finite_output=bool(result.get('original_fallback_available')),
            production_numerical_certified=False,original_fallback_available=bool(result.get('original_fallback_available')))
        row['controller_stop']=True
        if result.get('durable_failure_receipt_written') is False:
            row.update(durable_failure_receipt_written=False,
                original_finite_endpoint_not_durably_preserved=not result.get('original_fallback_available',False))
            print(json.dumps(s.plain(row),ensure_ascii=True),flush=True)
        try:
            seal(store,report/('CACHE_FAILURE_'+uuid.uuid4().hex+'.json'),{'time':s.now(),'row':row,'rerun_inverse_permitted':False})
        except Exception as persistence_error:
            row.update(durable_failure_receipt_written=False,receipt_error=str(persistence_error))
            print(json.dumps(s.plain(row),ensure_ascii=True),flush=True)
        return row
    try:
        out=cache.load_for_score(raw)  # Always parameter readback + byte verification before truth/scoring.
    except Exception as error:
        row.update(status='PARAMETER_READBACK_FAILED_NO_REFIT',score_status='NOT_SCOREABLE',controller_stop=True,
            error_type=type(error).__name__,error=str(error))
        return row
    row.update(finite_output=True,production_numerical_certified=bool(out.get('numerical_valid',False)),
        inference_numerical_certified=bool(out.get('numerical_valid',False)),
        status='FINITE_NUMERICALLY_CERTIFIED' if out.get('numerical_valid',False) else 'FINITE_NUMERICALLY_UNCERTIFIED')
    try:
        score=score_action(out)
        if (not isinstance(score,dict) or score.get('status')=='SCORING_FAILED'
                or score.get('score_status')=='SCORING_FAILED'):
            raise PanelIntegrityError('External score execution failed; retry scoring from the existing cache only')
        cache_score=cache.record_score('panel_external_v001',score,raw)
        row.update(metrics=score.get('metrics'),score_status='SCORED',external_score=score,
            compact_score_receipt=cache_score,truth_loaded_after_lossless_cache=True)
        seal(store,success,{'time':s.now(),'row':row,'new_inverse_count_is_attempt_local':True})
    except Exception as error:
        row.update(score_status='SCORING_FAILED',score_error_type=type(error).__name__,score_error=str(error))
        try:
            seal(store,report/('SCORING_FAILURE_'+uuid.uuid4().hex+'.json'),{'time':s.now(),'row':row,
                'retry_scope':'score only; never another inference'})
        except Exception as persistence_error:
            row.update(controller_stop=True,durable_failure_receipt_written=False,
                receipt_error=str(persistence_error))
            print(json.dumps(s.plain(row),ensure_ascii=True),flush=True)
    return row


def run(device='cuda',*,store=None):
    if device!='cuda': raise PanelIntegrityError('Formal panel execution is root-owned native GPU only')
    execution_entry()
    from .smooth_calibration import physics,contract
    from .compact_joint_cache import runtime_identity
    from .numerical_strategy import repaired_infer
    from .development import cached_v3
    from . import experiment_registry as registry,metrics
    import torch
    store=production_store() if store is None else store
    registration,digest=verify_registration(store)
    qualification=read_sealed(store,Path(store.report_root)/QUALIFICATION_NAME)
    planned=registration['planned_cells'];rows=[]
    with phase_locks(store,digest,VERSION+'-run'):
        terminal=Path(store.report_root)/RESULT_NAME
        if terminal.exists(): return read_sealed(store,terminal)
        s.frozen_bridge()
        from research_v6 import inputs
        refs={r['case_id']:r for r in registration['references']}
        try:
            torch.set_num_threads(2);p=physics(device)
            if contract(p)!=registration['calibration_contract']: raise PanelIntegrityError('Runtime C1 contract differs')
            require_qualification(qualification,registration,digest,runtime_identity(p))
            # Recheck the qualified compact parameter/cache bytes on this
            # same runtime before issuing any of the eight inference tickets.
            for origin in registration['v002_origins']:
                cell=next(c for c in planned if c['case_id']==origin['case_id'] and c['method']==origin['method'])
                raw=inputs.load_raw(refs[cell['case_id']],p)
                preserved=cache_for(store,p,cell,registration,digest,qualification=True).load_for_score(raw)
                original_arrays=s.arrays(origin['output_cache']['path'])
                if not all(byte_matches(preserved,original_arrays).values()):
                    raise PanelIntegrityError('Native compact qualification no longer reproduces original v002 bytes')
            for cell in planned:
                if cell['execution_role']=='REUSE_V002':
                    origin=next(r for r in registration['v002_origins'] if r['case_id']==cell['case_id'] and r['method']==cell['method'])
                    row=reuse_row(cell,origin,digest);rows.append(row)
                    path=Path(store.report_root)/'reused_cells'/(cell_key(cell,digest)+'.json')
                    if path.exists(): read_sealed(store,path)
                    else: seal(store,path,row)
                    continue
                verify_registration(store)
                ref=refs[cell['case_id']];raw=inputs.load_raw(ref,p)
                q.verify_initializer_raw(ref,raw,registration['v3_initializer_provenance'][ref['case_id']])
                cache=cache_for(store,p,cell,registration,digest)
                def action(cell=cell,ref=ref,raw=raw):
                    state=cached_v3(ref,p)[0] if cell['initializer']=='v3' else None
                    out=repaired_infer(raw,p,initializer=cell['initializer'],v3_state=state,
                        physical_options=copy.deepcopy(CONFIG['physical_options']),
                        numerical_options=copy.deepcopy(CONFIG['numerical_options']))
                    return c2.wrap_output(out,cell,registration)
                def score_action(out,cell=cell,ref=ref,raw=raw):
                    tensors={key:torch.as_tensor(value,device=p.M.device) if isinstance(value,np.ndarray) else value for key,value in out.items()}
                    truth=inputs.load_diagnostic_truth(ref,p)
                    return metrics.score(tensors,truth,raw,p,detector=None,method=cell['method'],
                        application_domain='C1v3_small_panel_v003_development_regression')
                row=evaluate_new_cell(cell,cache=cache,raw=raw,action=action,score_action=score_action,
                    count_call=registry.count_call,store=store,digest=digest,verify_sources=lambda:verify_registration(store))
                rows.append(row)
                store.append(Path(store.report_root)/'progress.jsonl',{'time':s.now(),'row':row})
                if row.get('controller_stop') or row.get('compact_result',{}).get('status','').startswith('FAIL_CLOSED'):
                    raise PanelStorageStop('Representation integrity failure: retain all remaining cells as unexecuted')
        except Exception as error:
            available={(r['case_id'],r['method']) for r in rows}
            for cell in planned:
                if cell['execution_role']=='REUSE_V002' and (cell['case_id'],cell['method']) not in available:
                    origin=next(r for r in registration['v002_origins'] if r['case_id']==cell['case_id'] and r['method']==cell['method'])
                    rows.append(reuse_row(cell,origin,digest))
            value={'time':s.now(),'status':'PANEL_CONTROLLER_STOP',
                'registration_sha256':digest,'error_type':type(error).__name__,'error':str(error),
                'traceback':traceback.format_exc(),'summary':summary(planned,rows),'storage_ledger':store.budget().ledger(),
                'not_a_method_selection':True,'final_payloads_opened':False}
            return run_receipt(store,Path(store.report_root)/('RUN_STOP_'+uuid.uuid4().hex+'.json'),value,
                failure_status='PANEL_CONTROLLER_STOP_RECEIPT_FAILED')
        value={'time':s.now(),'status':'C1V3_SMALL_PANEL_COMPLETE','registration_sha256':digest,
            'summary':summary(planned,rows),'qualification':file_ref(Path(store.report_root)/QUALIFICATION_NAME),
            'storage_ledger':store.budget().ledger(),'planned_maximum_new_inverse_calls':8,
            'new_inverse_wall_budget_s_excluding_overhead':1440,'method_selection_performed':False,
            'scientific_acceptance_claimed':False,'historical_detector_threshold_transferred':False,
            'final_payloads_opened':False,'whole_closeout_complete':False}
        verify_registration(store)
        if value['summary']['scoring_failed_cells']:
            value['status']='C1V3_SMALL_PANEL_SCORING_RETRY_ONLY'
            return run_receipt(store,Path(store.report_root)/('SCORING_RETRY_'+uuid.uuid4().hex+'.json'),value,
                failure_status='PANEL_SCORING_RETRY_RECEIPT_FAILED')
        return run_receipt(store,terminal,value,failure_status='PANEL_TERMINAL_RECEIPT_FAILED')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    reg=sub.add_parser('register');reg.add_argument('--contract',type=Path,action='append',required=True)
    for name in ('qualify','run'):
        item=sub.add_parser(name);item.add_argument('--device',choices=('cuda',),default='cuda')
    args=parser.parse_args()
    if args.command=='register':
        _,digest=register(args.contract);value={'status':'PANEL_REGISTERED','registration_sha256':digest,
            'report_root':str(REPORT_DIR),'data_root':str(DATA_DIR),'new_complete_inverse_calls':0}
    elif args.command=='qualify':
        value=qualify(args.device)
    else: value=run(args.device)
    print(json.dumps(s.plain(value),ensure_ascii=True,indent=2),flush=True)
    failed=(value.get('all_passed') is False or any(word in value.get('status','') for word in ('STOP','RETRY','FAILED')))
    raise SystemExit(2 if failed else 0)


if __name__=='__main__': main()
