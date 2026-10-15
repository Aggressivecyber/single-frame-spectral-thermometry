"""Root-side, metadata-only release of verified unused namespace reserves.

This maintenance tool does not change any registered cache release method.
It never reconstructs, scores, fits, reads raw arrays, or removes a fallback.
Prepare a durable plan first; release exactly its single zero placeholder.
"""
from . import storage as s
from .explicit_store import ExplicitDStore, _checked_path
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat

VERSION='owned-namespace-zero-reserve-cleanup-v001'
SOURCE='research_closeout/cache_reservation_cleanup.py'
TEST_SOURCE='research_closeout/tests/cache_reservation_cleanup.py'
REPORT=s.REPORTS/'repair_001/cache_reservation_cleanup_001'
CACHE_SCHEMA='joint-float64-forward-lossless-cache-v002-explicit-store'
CANONICAL=('temperature_color_K','template_postblur_e_ref','continuum_postblur_bands',
    'predicted_raw_e','predicted_raw_DN','standardized_residual','valid','low_signal')
ORIGINAL_SIX=('T','u','C','predicted_raw_e','valid','low_signal')
HEX64=re.compile(r'[0-9a-f]{64}')
HEX32=re.compile(r'[0-9a-f]{32}')


class CleanupIntegrityError(RuntimeError): pass


def _require(condition,message):
    if not condition: raise CleanupIntegrityError(message)


def _ref(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':s.sha(path),'bytes':path.stat().st_size}


def _source_hashes():
    return {name:s.sha(s.ROOT/name) for name in (SOURCE,TEST_SOURCE,
        'research_closeout/explicit_store.py','research_closeout/compact_joint_cache.py','research_closeout/storage.py')}


def _read_path(store,path):
    # Each validation first uses the original write_path on the sole target,
    # checking owner markers and the live budget. Read-only proofs retain the
    # original lexical/resolved/no-reparse guard without re-enrolling per file.
    return _checked_path(path,store.roots)


def _maintenance_path(path):
    path=Path(path);root=REPORT.resolve()
    _require(path.is_absolute() and path.drive.upper() in ('E:','D:'),'Maintenance receipts require absolute E/D paths')
    lexical=Path(os.path.abspath(path));resolved=path.resolve()
    _require(root in lexical.parents and root in resolved.parents,'Maintenance receipt escapes its independent report root')
    for part in (path,*path.parents):
        if part.exists() or part.is_symlink():
            attrs=getattr(part.lstat(),'st_file_attributes',0)
            _require(not part.is_symlink() and not attrs & getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024),
                'Maintenance receipt path has a reparse point')
    return resolved


def _seal(path,value):
    path=_maintenance_path(path);side=_maintenance_path(path.with_name(path.name+'.sha256.json'))
    encoded=(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf8')
    _require(len(encoded)<128*2**10,'Maintenance record exceeds its small metadata bound')
    s.budget_guard();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as handle:
        handle.write(encoded);handle.flush();os.fsync(handle.fileno())
    receipt=_ref(path)
    with side.open('xb') as handle:
        handle.write((json.dumps(receipt,sort_keys=True)+'\n').encode('utf8'));handle.flush();os.fsync(handle.fileno())
    return receipt


def _read_sealed(path,checked):
    path=checked(path);side=checked(path.with_name(path.name+'.sha256.json'))
    seal=s.read(side)
    _require(isinstance(seal,dict) and seal==_ref(path),'Sealed body path, bytes or SHA differs from its sidecar')
    return s.read(path),{'body':_ref(path),'sidecar':_ref(side)}


def _instance(record,key,context):
    _require(isinstance(record,dict) and record.get('schema')==CACHE_SCHEMA
        and record.get('cache_key')==key and record.get('context_fingerprint')==context,
        'Sealed records do not name the same cache key/context/schema')


def _verify_ref(store,value,expected):
    _require(isinstance(value,dict),'Expected an exact file reference')
    path=_read_path(store,value.get('path',''))
    _require(path==expected.resolve() and value==_ref(path),'Referenced cache file changed or escaped its exact instance path')
    return value


def _phase(store,namespace,context,report,data,terminal):
    expected=store.report_root/('NATIVE_COMPACT_QUALIFICATION.json' if namespace=='qualification' else 'PANEL_RESULT.json')
    _require(Path(terminal).resolve()==expected.resolve(),'Wrong terminal record for the requested namespace')
    value,refs=_read_sealed(expected,lambda path:_read_path(store,path))
    registration=context['identity']['registration_sha256']
    _require(value.get('registration_sha256')==registration,'Terminal belongs to another registration')
    if namespace=='qualification':
        cells=value.get('cells',[])
        _require(value.get('all_passed') is True and value.get('planned_existing_endpoints')==4
            and value.get('audited_existing_endpoints')==4 and value.get('passed_existing_endpoints')==4
            and len(cells)==4 and all(row.get('passed') is True for row in cells),
            'Qualification has no complete passed terminal denominator')
        matched=[row for row in cells if row.get('compact_report_root')==str(report)
            and row.get('compact_data_root')==str(data)]
        _require(len(matched)==1,'Qualification terminal does not identify exactly this compact instance')
        cell=matched[0]
        _require(cell.get('case_id')==context['identity']['case_id'] and cell.get('method')==context['identity']['method'],
            'Qualification method/case does not match its cache')
        for field in ('original_canonical_byte_matches','compact_readback_original_byte_matches'):
            matches=cell.get(field,{})
            _require(set(matches)==set(ORIGINAL_SIX) and all(v is True for v in matches.values()),
                'Qualification does not retain every original canonical byte match')
    else:
        summary=value.get('summary',{})
        _require(value.get('status')=='C1V3_SMALL_PANEL_COMPLETE' and summary.get('planned_cells')==12
            and summary.get('returned_cell_rows')==12 and len(summary.get('cells',[]))==12,
            'Inference phase has no complete terminal denominator')
        matched=[row for row in summary['cells'] if row.get('case_id')==context['identity']['case_id']
            and row.get('method')==context['identity']['method']]
        _require(len(matched)==1,'Inference terminal does not identify the same case/method')
    _require(not (store.report_root/'PHASE_ACTIVE.json').exists(),'The owned task phase is still active')
    global_lock=s.REPORTS/'PHASE_ACTIVE.json'
    if global_lock.exists():
        _require(s.read(global_lock).get('registration_sha256')!=registration,'The registered root phase is still active')
    return refs


def _zero_ref(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        initial=os.fstat(handle.fileno())
        _require(stat.S_ISREG(initial.st_mode) and initial.st_size>0,'Reservation must be a nonempty regular zero placeholder')
        for block in iter(lambda:handle.read(1024*1024),b''):
            _require(block.count(0)==len(block),'Reservation contains nonzero data; never remove an actual fallback')
            digest.update(block)
        final=os.fstat(handle.fileno())
    fields=('st_dev','st_ino','st_size','st_mtime_ns')
    identity={name:getattr(initial,name) for name in fields}
    _require(identity=={name:getattr(final,name) for name in fields}
        =={name:getattr(path.stat(),name) for name in fields},'Reservation changed during validation')
    return {'path':str(path),'sha256':digest.hexdigest(),'bytes':initial.st_size},identity


def _validate(store,reservation,owner_token,context_fingerprint,terminal,*,present):
    _require(type(store) is ExplicitDStore,'Use the original ExplicitDStore, with its retained owner token')
    _require(isinstance(owner_token,str) and HEX32.fullmatch(owner_token),'A32-lowercase-hex cache owner token is required')
    _require(isinstance(context_fingerprint,str) and HEX64.fullmatch(context_fingerprint),'The expected64-hex context is required')
    path=store.write_path(reservation)
    try: parts=path.relative_to(store.data_root).parts
    except ValueError as error: raise CleanupIntegrityError('Reservation is outside its exact owned data root') from error
    _require(len(parts)==4 and parts[0] in ('qualification','inference') and parts[1]=='cells'
        and HEX64.fullmatch(parts[2]) and parts[3]=='FALLBACK_RESERVED_'+owner_token+'.bin',
        'Only qualification/inference/cells/64hexkey/FALLBACK_RESERVED_32hex.bin is permitted')
    namespace,_,key,_=parts;data=path.parent
    report=_read_path(store,store.report_root/namespace/'cells'/key)
    records={};refs={}
    for name in ('STARTED.json','RESERVATION.json','RESULT.json','PARAMETERS.json','PREPARED.json'):
        records[name],refs[name]=_read_sealed(report/name,lambda path:_read_path(store,path))
        _instance(records[name],key,context_fingerprint)
    started=records['STARTED.json'];reserved=records['RESERVATION.json'];result=records['RESULT.json']
    owner=started.get('owner')
    _require(isinstance(owner,dict) and owner.get('token')==owner_token and owner==reserved.get('owner')
        and isinstance(owner.get('pid'),int) and not isinstance(owner.get('pid'),bool) and owner['pid']>0,
        'STARTED and RESERVATION must share the exact original cache owner')
    _require('owner' not in result or result['owner']==owner,'RESULT names another owner')
    context=started.get('context')
    _require(isinstance(context,dict) and s.canonical(context)==context_fingerprint
        and context.get('storage_contract')==store.contract(),'Started context fingerprint or owned storage contract differs')
    _require(s.canonical({'identity':context['identity'],'raw_sha256':context['input_reference']['sha256'],
        'calibration':context['calibration']['fingerprint'],'config':context['config']})==key,
        'Cache key is not derived from its original context')
    _require(started.get('status')=='STARTED' and started.get('reserve_path')==str(path)
        and reserved.get('reservation',{}).get('path')==str(path)
        and started.get('reserve_bytes')==reserved['reservation'].get('bytes')
        ==context['frame_plan']['fallback_zip_bound_bytes'],'Reservation path or declared byte envelope differs')
    _require(result.get('status')=='LOSSLESS_READY' and result.get('lossless_representation_verified') is True
        and result.get('original_fallback_available') is False and result.get('fallback_retained') is False,
        'A lossless-ready result with no retained fallback is required')
    _require(not _read_path(store,data/'FALLBACK.npz').exists(),'An actual fallback is present; leave the whole instance untouched')
    for name in ('FAILURE.json','RECONSTRUCTION_FAILURE.json'):
        _require(not _read_path(store,report/name).exists(),'A failed or later-invalidated cache cannot release its reservation')
    prepared=records['PREPARED.json'];parameters=records['PARAMETERS.json']
    _require(prepared.get('context')==context,'Prepared parameters belong to another original context')
    _verify_ref(store,result.get('parameter_receipt'),report/'PARAMETERS.json')
    _verify_ref(store,parameters.get('prepared'),report/'PREPARED.json')
    parameter_ref=_verify_ref(store,parameters.get('parameter_cache'),data/'PARAMETERS.npz')
    verification=result.get('verification',{});matches=verification.get('per_array_byte_identical',{})
    _require(verification.get('all_byte_identical') is True and set(matches)==set(CANONICAL)|{'continuum_latent'}
        and all(v is True for v in matches.values()) and verification.get('forward_only') is True
        and all(verification.get(name)==0 for name in ('complete_inverse_calls','optimizer_calls','linear_solver_calls'))
        and verification.get('reconstructed_canonical_arrays')==prepared.get('canonical_arrays')
        and set(prepared.get('canonical_arrays',{}))==set(CANONICAL)
        and verification.get('reconstructed_continuum_latent')==prepared.get('continuum_latent'),
        'Every original canonical descriptor and forward-only byte verification must agree')
    release_failure=_read_path(store,report/'RESERVATION_RELEASE_FAILURE.json')
    if release_failure.exists():
        failure,refs['RESERVATION_RELEASE_FAILURE.json']=_read_sealed(release_failure,lambda path:_read_path(store,path))
        _instance(failure,key,context_fingerprint)
    phase_refs=_phase(store,namespace,context,report,data,terminal)
    identity={'schema':VERSION,'store_contract':store.contract(),'namespace':namespace,'cache_key':key,
        'owner':owner,'context_fingerprint':context_fingerprint,'reservation':reserved['reservation'],
        'sealed_cache_records':refs,'parameter_cache':parameter_ref,'phase_terminal':phase_refs,
        'source_sha256':_source_hashes(),
        'RESULT_owner_binding':'optional explicit owner; otherwise unique sealed STARTED/RESERVATION owner plus same key/context and parameter/verification chain'}
    if present:
        actual,file_identity=_zero_ref(path)
        _require(actual==reserved['reservation'],'Reservation bytes/hash differ from the sealed reservation reference')
        return identity,file_identity
    _require(not path.exists(),'A success receipt cannot be reused when its reservation exists again')
    return identity,None


def prepare(store,reservation,*,owner_token,context_fingerprint,phase_terminal):
    identity,file_identity=_validate(store,reservation,owner_token,context_fingerprint,phase_terminal,present=True)
    key=s.canonical(identity);folder=REPORT/'operations'/key;path=folder/'PLAN.json'
    value={'schema':VERSION+'-plan','time':s.now(),'identity':identity,'identity_sha256':key,
        'validated_zero_file_identity':file_identity,'only_delete':identity['reservation'],
        'preserve_all_other_files':True,'whole_inverse_calls':0,'forward_calls':0,'truth_used':False}
    if path.exists():
        previous,_=_read_sealed(path,_maintenance_path)
        _require(previous.get('identity')==identity and previous.get('identity_sha256')==key,
            'Another cleanup identity already owns this plan')
        return {'plan':previous,'plan_ref':_ref(path)}
    return {'plan':value,'plan_ref':_seal(path,value)}


def release(store,plan_path,*,owner_token,context_fingerprint):
    plan_path=_maintenance_path(plan_path);plan,plan_seals=_read_sealed(plan_path,_maintenance_path)
    identity=plan.get('identity',{});key=s.canonical(identity)
    _require(plan.get('schema')==VERSION+'-plan' and plan.get('identity_sha256')==key
        and plan_path==_maintenance_path(REPORT/'operations'/key/'PLAN.json')
        and plan.get('only_delete')==identity.get('reservation'),'Plan path or single-target identity differs')
    _require(identity.get('owner',{}).get('token')==owner_token
        and identity.get('context_fingerprint')==context_fingerprint,'Request owner/context differs from its durable plan')
    target=Path(identity['reservation']['path']);terminal=identity['phase_terminal']['body']['path']
    receipt_path=plan_path.parent/'RECEIPT.json';attempt_path=plan_path.parent/'ATTEMPT.json'
    if receipt_path.exists():
        receipt,_=_read_sealed(receipt_path,_maintenance_path)
        current,_=_validate(store,target,owner_token,context_fingerprint,terminal,present=False)
        _require(current==identity and receipt.get('schema')==VERSION+'-receipt'
            and receipt.get('status')=='RELEASED_ZERO_RESERVATION' and receipt.get('identity')==identity
            and receipt.get('identity_sha256')==key and receipt.get('plan_seals')==plan_seals
            and receipt.get('removed')==identity['reservation'] and receipt.get('unlink_count')==1,
            'Existing success receipt does not match this exact cleanup request')
        return receipt
    _require(not attempt_path.exists(),'A prior incomplete release attempt requires audit; never infer or repeat its deletion')
    current,file_identity=_validate(store,target,owner_token,context_fingerprint,terminal,present=True)
    _require(current==identity and file_identity==plan.get('validated_zero_file_identity'),
        'Sources, seals, context or zero reservation changed after preparation')
    _seal(attempt_path,{'schema':VERSION+'-attempt','time':s.now(),'identity_sha256':key,
        'plan_seals':plan_seals,'only_delete':identity['reservation'],'pid':os.getpid()})
    # Recheck all proof bytes, owner/path guards and zero bytes after the durable
    # attempt. This is the only unlink in this tool; no glob or recursive delete.
    current,file_identity=_validate(store,target,owner_token,context_fingerprint,terminal,present=True)
    _require(current==identity and file_identity==plan['validated_zero_file_identity'],'Reservation proof changed before unlink')
    approved=store.write_path(target)
    approved.unlink()
    receipt={'schema':VERSION+'-receipt','time':s.now(),'status':'RELEASED_ZERO_RESERVATION',
        'identity':identity,'identity_sha256':key,'plan_seals':plan_seals,'removed':identity['reservation'],
        'unlink_count':1,'preserve_all_other_files':True,'whole_inverse_calls':0,'forward_calls':0,'truth_used':False}
    _seal(receipt_path,receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','release'):
        item=sub.add_parser(command);item.add_argument('--task',required=True);item.add_argument('--store-owner',required=True)
        item.add_argument('--owner-token',required=True);item.add_argument('--context',required=True)
        if command=='prepare':
            item.add_argument('--reservation',type=Path,required=True);item.add_argument('--phase-terminal',type=Path,required=True)
        else: item.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();store=ExplicitDStore(args.task,owner_token=args.store_owner)
    value=prepare(store,args.reservation,owner_token=args.owner_token,context_fingerprint=args.context,
        phase_terminal=args.phase_terminal) if args.command=='prepare' else release(store,args.plan,
            owner_token=args.owner_token,context_fingerprint=args.context)
    print(json.dumps({'schema':VERSION,'status':value.get('status','PLAN_PREPARED'),
        'plan_ref':value.get('plan_ref'),'whole_inverse_calls':0,'forward_calls':0}),flush=True)


if __name__=='__main__': main()
