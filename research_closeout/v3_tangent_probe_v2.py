"""Two coupled-v2 linear solves; reuse immutable v1 old-block receipts.

This controller never reruns either v1 route, never writes parameter/step arrays,
and never treats CPU batch-dispatch contracts as native CUDA qualification.
"""
from . import storage as s
from . import v3_tangent_probe as v1
import argparse
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import time
import uuid

VERSION='v3-native-fixed-tangent-probe-v2-batch64'
ROUTE='coupled_fourier_v2_batch64'
REPORT_DIR=s.REPORTS/'repair_001/v3_tangent_probe_002'
HELPER_CONTRACT=s.REPORTS/'repair_001/v3_preconditioner_002/CONTRACTS.json'
OWN_SOURCES=('research_closeout/v3_tangent_probe_v2.py','research_closeout/tests/v3_tangent_probe_v2.py')
HELPER_SOURCES=('research_closeout/coupled_preconditioner_v2.py','research_closeout/tests/coupled_preconditioner_v2.py')
RUNTIME_SOURCES=('research_closeout/runtime_entry.py','research_closeout/tests/runtime_entry.py')
RUNTIME_CONTRACT=s.REPORTS/'repair_001/runtime_entry_001/CONTRACTS.json'
CONFIG={**copy.deepcopy(v1.CONFIG),'version':VERSION,'routes':[ROUTE],'planned_cells':2,
    'total_route_wall_s':240.,'backend_batch_limit':64,'reused_old_block_controls':2,
    'old_block_rerun':False,'v1_failed_coupled_rerun':False,
    'direct_old_new_step_difference_available':False,
    'comparison_bounds':'analytic conditional bounds from same-system true residual estimates and triangle inequality; no interval-arithmetic certificate',
    'runtime_entry':'research_closeout.runtime_entry'}


def source_hashes(): return {name:s.sha(s.ROOT/name) for name in OWN_SOURCES}
def write_json(path,value,*,folder=REPORT_DIR,reserve=True,cap=None):
    return v1.write_json(path,value,folder=folder,reserve=reserve,cap=cap)
def space_guard(*,registration=False,folder=REPORT_DIR,free_bytes=None):
    return v1.space_guard(registration=registration,folder=folder,free_bytes=free_bytes)


@contextmanager
def owner_locks(digest,paths=None):
    paths=paths or (s.REPORTS/'PHASE_ACTIVE.json',REPORT_DIR/'PHASE_ACTIVE.json')
    owner={'pid':os.getpid(),'owner_token':uuid.uuid4().hex,'phase':VERSION,'registration_sha256':digest,'time':s.now()}
    acquired=[]
    try:
        for path in map(Path,paths):
            path=s.write_path(path);path.parent.mkdir(parents=True,exist_ok=True)
            fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY);acquired.append(path)
            with os.fdopen(fd,'w',encoding='utf8') as handle:
                json.dump(owner,handle);handle.flush();os.fsync(handle.fileno())
        yield owner
    finally:
        for path in reversed(acquired):
            try:
                value=s.read(path)
                if value.get('pid')==owner['pid'] and value.get('owner_token')==owner['owner_token']: path.unlink()
            except (FileNotFoundError,ValueError,UnicodeError): pass


def reference_controls():
    registration,digest=v1.verify_registration()
    result=s.read(v1.REPORT_DIR/'RESULT.json')
    if result.get('registration_sha256')!=digest or result.get('planned_cells')!=4 or result.get('returned_cells')!=4:
        raise v1.IntegrityError('All four immutable v1 rows are required')
    for ref in result.get('cell_file_refs',[]): v1.verify_file(ref)
    controls=[]
    files=[v1.file_ref(v1.REPORT_DIR/name) for name in ('REGISTRATION.json','REGISTRATION_SHA256.json','RESULT.json')]
    for index,case in enumerate(registration['cases']):
        cell_path=v1.REPORT_DIR/f'CELL_{index:02d}_old_block_diagonal.json'
        system_path=v1.REPORT_DIR/f'SYSTEM_{index:02d}.json'
        old=s.read(cell_path);identity=s.read(system_path)['identity']
        if old.get('case_id')!=case['case_id'] or old.get('route')!='old_block_diagonal' or old.get('converged') is not True:
            raise v1.IntegrityError('The same-raw converged v1 old-block control is required')
        if not old.get('true_residual_recomputed') or old['true_relative_residual']>1e-10 or old.get('step_saved') is not False:
            raise v1.IntegrityError('Old control residual/step provenance is invalid')
        if old['system_fingerprint']!=identity['system_fingerprint']:
            raise v1.IntegrityError('Old control and frozen exact system disagree')
        refs={'cell':v1.file_ref(cell_path),'system':v1.file_ref(system_path)}
        files.extend(refs.values())
        controls.append({'case_id':case['case_id'],'old_row':old,'system_identity':identity,'files':refs})
    return registration,controls,files


def register(contract_path=None):
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    with owner_locks('registration-pending'):
        if (REPORT_DIR/'REGISTRATION.json').exists() or (REPORT_DIR/'REGISTRATION_SHA256.json').exists():
            raise FileExistsError('v2 registration is immutable')
        reserve=space_guard(registration=True)
        helper=v1._passed_contract(HELPER_CONTRACT,HELPER_SOURCES)
        own=v1._passed_contract(contract_path or REPORT_DIR/'CPU_CONTRACTS.json',OWN_SOURCES)
        if s.read(HELPER_CONTRACT).get('batch_limit')!=64: raise v1.IntegrityError('Fixed64-block helper contracts are required')
        runtime=s.read(RUNTIME_CONTRACT);checks=runtime.get('checks',[])
        if len(checks)!=5 or runtime.get('passed')!=5 or runtime.get('total')!=5 or any(c.get('passed') is not True for c in checks):
            raise v1.IntegrityError('All five runtime metadata contracts are required')
        for name in RUNTIME_SOURCES:
            if runtime.get('source_sha256',{}).get(name)!=s.sha(s.ROOT/name):
                raise v1.IntegrityError('Runtime entry contract source hash is stale: '+name)
        old,controls,upstream=reference_controls()
        sources=dict(old['source_sha256']);sources.update(source_hashes())
        sources.update({name:s.sha(s.ROOT/name) for name in HELPER_SOURCES+RUNTIME_SOURCES})
        value={'schema':VERSION+'-registration','time':s.now(),'config':copy.deepcopy(CONFIG),
            'source_sha256':sources,'contracts':[helper,own,v1.file_ref(RUNTIME_CONTRACT)]+copy.deepcopy(old['contracts']),'upstream_files':upstream,
            'cases':copy.deepcopy(old['cases']),'reused_controls':controls,'selection':old['selection'],
            'space_reserve':reserve,'truth_loaded':False,'whole_inverse_calls':0,'GPU_started':False,
            'v1_failed_coupled_rows_preserved':2,'direct_step_arrays_available':False}
        value['fingerprint']=s.canonical({name:value[name] for name in
            ('config','source_sha256','cases','reused_controls','selection')})
        write_json(REPORT_DIR/'REGISTRATION.json',value)
        write_json(REPORT_DIR/'REGISTRATION_SHA256.json',v1.file_ref(REPORT_DIR/'REGISTRATION.json'))
        return value


def verify_registration():
    side=s.read(REPORT_DIR/'REGISTRATION_SHA256.json')
    if Path(side['path']).resolve()!=(REPORT_DIR/'REGISTRATION.json').resolve(): raise v1.IntegrityError('Wrong v2 registration sidecar')
    v1.verify_file(side);value=s.read(REPORT_DIR/'REGISTRATION.json')
    if value.get('schema')!=VERSION+'-registration' or value['config']!=CONFIG: raise v1.IntegrityError('v2 configuration changed')
    v1.verify_sources(value['source_sha256'])
    for ref in value['contracts']+value['upstream_files']+[value['selection']]: v1.verify_file(ref)
    for case in value['cases']:
        for ref in (case['raw_file'],case['state_file'],case['provenance']['receipt_journal']): v1.verify_file(ref)
    if tuple(c['case_id'] for c in value['cases'])!=v1.EXPECTED_CASES: raise v1.IntegrityError('The two old raw denominator changed')
    if tuple(c['case_id'] for c in value['reused_controls'])!=v1.EXPECTED_CASES: raise v1.IntegrityError('The two old control denominator changed')
    fingerprint=s.canonical({name:value[name] for name in ('config','source_sha256','cases','reused_controls','selection')})
    if fingerprint!=value['fingerprint']: raise v1.IntegrityError('v2 source/input/control fingerprint changed')
    return value,side['sha256']


class NativeCase(v1.NativeCase):
    def build_preconditioner(self,route):
        if route!=ROUTE: raise v1.IntegrityError('v2 cannot rerun a v1 route')
        from .coupled_preconditioner_v2 import build
        self.check()
        pre,D,metadata=build(self.p,self.state,self.beta,self.W,self.strength,CONFIG['damping'])
        if not self.torch.equal(D,self.D): raise v1.IntegrityError('The exact damping diagonal changed')
        self.check()
        def masked(vector):
            self.check();self.counts['precondition']+=1
            answer=pre(vector*self.mask)*self.mask;self.check();return answer
        return masked,metadata


def same_system(current,reference):
    #The saved SYSTEM file is byte-hashed; this equality also verifies every
    #dtype/shape/payload array SHA and the complete original identity object.
    return current==reference and s.canonical(current)==s.canonical(reference)


def conditional_bounds(old,new,min_D_free,*,same_system_verified):
    result={'direct_solution_difference':None,'direct_linear_output_difference':None,
        'old_step_array_available':False,'analytic_condition':'same exact symmetric system A_F; recorded recomputed residual estimates represent its true residuals; finite-precision FFT/reduction roundoff is not enclosed',
        'interval_arithmetic_certified':False,'step_L2_upper_bound':None,'linear_output_RMS_upper_bounds':None,
        'proof':'A_F=F(J^T W J+Lambda L+mu D)F >= mu*min(D_free) I. A_F(x_old-x_new)=r_new-r_old, so norm(step_difference)<= (norm(r_old)+norm(r_new))/(mu*min(D_free)). For the same fixed linear output map L, RMS(L step_difference)<=RMS(L x_old)+RMS(L x_new) by the triangle inequality.',
        'bound_is_nonlinear_accuracy_or_uncertainty':False,'both_stopping_tests_passed':bool(old.get('converged') and new.get('converged'))}
    values=(min_D_free,old.get('true_residual_norm'),new.get('true_residual_norm'))
    if not same_system_verified or not all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in values) or min_D_free<=0:
        result['unavailable_reason']='same-system identity or finite residual/spectral-lower-bound evidence unavailable';return result
    if not old.get('true_residual_recomputed') or not new.get('true_residual_recomputed'):
        result['unavailable_reason']='both true residual recomputations are required';return result
    alpha=CONFIG['damping']*min_D_free
    result.update(spectral_lower_bound=alpha,minimum_free_D=min_D_free,
        step_L2_upper_bound=(old['true_residual_norm']+new['true_residual_norm'])/alpha)
    a=old.get('linear_outputs') or {};b=new.get('linear_outputs') or {};bounds={}
    for name in ('raw_linear_RMS_e','template_postblur_linear_RMS_e_ref','color_linear_RMS_K','latent_temperature_linear_RMS_K','beta_linear_L2'):
        va=a.get(name);vb=b.get(name)
        bounds[name]=va+vb if all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in (va,vb)) else None
    result['linear_output_RMS_upper_bounds']=bounds
    result['linear_bound_kind']='conservative same-map triangle bound; not an observed difference or a sensitivity-tightened bound'
    return result


def execute(registration,digest,*,folder=REPORT_DIR,device='cuda',case_factory=NativeCase,
            clock=time.perf_counter,guard_space=space_guard,verify=lambda:None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);terminal=folder/'RESULT.json'
    if terminal.exists():
        value=s.read(terminal)
        if value.get('registration_sha256')!=digest: raise v1.IntegrityError('Another registration owns the v2 terminal')
        for ref in value.get('cell_file_refs',[]): v1.verify_file(ref)
        return value
    started=clock();rows=[];errors=[];partial=(folder/'RUN_STARTED.json').exists()
    if partial and s.read(folder/'RUN_STARTED.json').get('registration_sha256')!=digest:
        raise v1.IntegrityError('Another registration owns the interrupted v2 attempt')
    try:
        if not partial:
            guard_space(folder=folder)
            write_json(folder/'RUN_STARTED.json',{'schema':VERSION,'registration_sha256':digest,
                'time':s.now(),'planned_new_cells':2,'reused_old_block_controls':2,'whole_inverse_calls':0},folder=folder)
        for index,(case,control) in enumerate(zip(registration['cases'],registration['reused_controls'])):
            if partial: break
            verify();guard_space(folder=folder);system=None
            row={'case_id':case['case_id'],'route':ROUTE,'registration_sha256':digest,'attempted':False,
                'converged':False,'iterations':0,'pcg_started':False,'counts':{},'whole_inverse_calls':0,
                'reused_old_control':control['files'],'direct_solution_difference':None}
            try:
                system=case_factory(case,device)
                identity_ok=same_system(system.identity,control['system_identity'])
                row.update(system_identity_match=identity_ok,system_fingerprint=system.identity['system_fingerprint'],setup_counts=system.setup_counts)
                if not identity_ok: raise v1.IntegrityError('Byte-level original system identity mismatch; solve not started')
                remaining=240.-sum(r.get('wall_s',0.) for r in rows)
                if remaining<=0: row['status']='NOT_ATTEMPTED_TOTAL_BUDGET'
                else:
                    row['attempted']=True
                    receipt,step=v1.route_cell(system,ROUTE,clock=clock,remaining_s=remaining);row.update(receipt)
                    min_D=float(system.D[system.mask].min()) if bool(system.mask.any()) else None
                    row['comparison']=conditional_bounds(control['old_row'],row,min_D,same_system_verified=True)
                    row['case_total_counts']=dict(system.counts)
                    row['native_GPU_qualified']=bool(str(device).startswith('cuda') and row['converged'])
                    del step
            except Exception as error:
                row.update(status='SYSTEM_IDENTITY_OR_SETUP_FAILURE' if not row['attempted'] else 'PROBE_FAILURE',
                    converged=False,error={'type':type(error).__name__,'message':str(error)[:500]})
                if system is not None: row['case_total_counts']=dict(system.counts)
            rows.append(row);write_json(folder/f'CELL_{index:02d}.json',row,folder=folder)
            if system is not None: system.guard=None
            del system
    except Exception as error:
        errors.append({'type':type(error).__name__,'message':str(error)[:500]})
    if partial:
        for path in sorted(folder.glob('CELL_*.json')):
            try:
                row=s.read(path)
                if row.get('registration_sha256')!=digest or row.get('route')!=ROUTE: raise v1.IntegrityError('Invalid interrupted v2 cell')
                rows.append(row)
            except Exception as error: errors.append({'type':type(error).__name__,'message':str(error)[:500]})
    supplied={r['case_id']:r for r in rows}
    complete=[supplied.get(case['case_id'],{'case_id':case['case_id'],'route':ROUTE,'attempted':False,'converged':False,
        'status':'NOT_ATTEMPTED_INTERRUPTED' if partial else 'NOT_ATTEMPTED_CONTROLLER_FAILURE','counts':{},'pcg_started':False})
        for case in registration['cases']]
    result={'schema':VERSION+'-result','time':s.now(),'registration_sha256':digest,
        'status':'INTERRUPTED_ATTEMPT_RETAINED' if partial else ('PROBE_COMPLETE' if not errors and all(r['converged'] for r in complete) else 'PROBE_COMPLETE_WITH_FAILURES'),
        'planned_new_cells':2,'returned_new_cells':len(complete),'attempted_new_cells':sum(r['attempted'] for r in complete),
        'converged_new_cells':sum(r['converged'] for r in complete),'failed_or_unattempted_new_cells':sum(not r['converged'] for r in complete),
        'rows':complete,'reused_old_block_controls':[{'case_id':c['case_id'],'old_row':c['old_row'],'files':c['files']} for c in registration['reused_controls']],
        'reused_old_routes_executed_this_run':0,'v1_failed_coupled_rows_repeated':0,'direct_step_difference_available':False,
        'PCG_calls':sum(bool(r.get('pcg_started')) for r in complete),'full_new_denominator_preserved':True,
        'errors':errors,'route_wall_limit_s':120.,'planned_total_new_route_wall_s':240.,
        'wall_s_including_overhead':clock()-started,'whole_inverse_calls':0,'count_call_used':False,
        'truth_used':False,'final_payloads_opened':False,'step_arrays_saved':False,'device':device,
        'recorded_new_operator_counts':{key:sum(r.get('case_total_counts',{}).get(key,0) for r in complete)
            for key in ('JVP','VJP','normal','precondition')},
        'operator_counts_complete':all('case_total_counts' in r for r in complete),
        'cell_file_refs':[v1.file_ref(path) for path in sorted(folder.glob('CELL_*.json'))]}
    write_json(terminal,result,folder=folder,reserve=False)
    return result


def launch_failure(error,*,operation='run',folder=REPORT_DIR):
    value={'schema':VERSION+'-launch-failure','time':s.now(),'status':'FAILED_BEFORE_PROBE','operation':operation,
        'planned_new_cells':2,'returned_new_cells':2,'attempted_new_cells':0,'failed_or_unattempted_new_cells':2,
        'rows':[{'case_id':c,'route':ROUTE,'attempted':False,'converged':False,'status':'NOT_STARTED_LAUNCH_REJECTED'} for c in v1.EXPECTED_CASES],
        'whole_inverse_calls':0,'old_controls_rerun':False,'error':{'type':type(error).__name__,'message':str(error)[:500]}}
    write_json(Path(folder)/('LAUNCH_FAILURE_'+uuid.uuid4().hex+'.json'),value,folder=folder,reserve=False)
    return value


def run(device='cuda'):
    if device!='cuda': raise ValueError('Native execution requires explicit --device cuda')
    try:
        registration,digest=verify_registration()
        space_guard()
        with owner_locks(digest):
            import torch
            torch.set_num_threads(2)
            return execute(registration,digest,device=device,verify=verify_registration)
    except Exception as error: return launch_failure(error)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    registration=sub.add_parser('register');registration.add_argument('--contract',type=Path)
    execution=sub.add_parser('run');execution.add_argument('--device',choices=['cuda'],required=True)
    args=parser.parse_args()
    try: result=register(args.contract) if args.command=='register' else run(args.device)
    except Exception as error: result=launch_failure(error,operation=args.command)
    print(json.dumps({'schema':result['schema'],'status':result.get('status','REGISTERED'),
        'planned_new_cells':2,'converged_new_cells':result.get('converged_new_cells'),'whole_inverse_calls':0}),flush=True)
    if result.get('status')=='FAILED_BEFORE_PROBE': raise SystemExit(2)


if __name__=='__main__': main()
