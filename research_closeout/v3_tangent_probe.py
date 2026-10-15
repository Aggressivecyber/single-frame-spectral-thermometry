"""Register/run a fixed native v3 tangent comparison; never a complete inverse.

Only the two old v002 development raws and their same-raw frozen v3 state/beta
are opened. The original piecewise-linear calibration and exact tangent system
remain fixed. No truth, final data, nonlinear update or step-array export exists.
"""
from . import storage as s
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import numpy as np

VERSION='v3-native-fixed-tangent-probe-v1'
REPORT_DIR=s.REPORTS/'repair_001/v3_tangent_probe_001'
V002=s.REPORTS/'repair_001/quick_v002'
HELPER_CONTRACT=s.REPORTS/'repair_001/v3_preconditioner_001/CONTRACTS.json'
OWN_SOURCES=('research_closeout/v3_tangent_probe.py','research_closeout/tests/v3_tangent_probe.py')
HELPER_SOURCES=('research_closeout/coupled_preconditioner.py','research_closeout/tests/coupled_preconditioner.py')
DEPENDENCY_SOURCES=OWN_SOURCES+HELPER_SOURCES+('research_closeout/storage.py','research_closeout/measurement.py',
    'physics.py','solver.py','repair_v3/solver.py','runtime.py','config.py')
ROUTES=('old_block_diagonal','coupled_fourier')
EXPECTED_CASES=('v5_development_selection_mother_0000_positive_normal_noise0',
                'v5_development_selection_mother_0013_negative_normal_noise0')
CONFIG={'version':VERSION,'calibration':'original promoted float32 piecewise-linear LUT; measurement.physics',
    'damping':1e-3,'rtol':1e-10,'maxiter':2000,'route_wall_s':120.,'total_route_wall_s':480.,
    'audit_reserve_s':2.,'routes':list(ROUTES),'raws':2,'planned_cells':4,
    'output_cap_bytes':2**20,'terminal_reserve_bytes':64*2**10,'lock_meta_reserve_bytes':4096,
    'free_floor_bytes':2*2**30,'whole_inverse_calls':0,'truth_used':False,
    'raw_or_state_arrays_written':False,'preconditioner_is_exact_Hessian_inverse':False,
    'rhs':'negative original weighted-raw-plus-latent-prior gradient, restricted to one fixed free-bound mask',
    'system':'F*(J^T W J+Lambda*periodic_laplacian+1e-3*D)*F; D is original p.preconditioner_diagonal',
    'free_mask':'active fields/beta except exact-bound outward-gradient coordinates; native-bound tolerance1e-12',
    'partial_recovery':'preserve completed cell receipts and terminate remaining cells; never rerun a partial attempt'}


class IntegrityError(RuntimeError): pass
class BudgetError(RuntimeError): pass


def source_hashes(): return {name:s.sha(s.ROOT/name) for name in OWN_SOURCES}
def file_ref(path): return {'path':str(Path(path).resolve()),'sha256':s.sha(path)}
def verify_file(ref):
    path=Path(ref['path'])
    if not path.is_file() or s.sha(path)!=ref['sha256']: raise IntegrityError('Frozen file changed: '+str(path))
def verify_sources(values):
    for name,digest in values.items():
        if s.sha(s.ROOT/name)!=digest: raise IntegrityError('Frozen source changed: '+name)
def array_sha(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256(str(a.dtype).encode()+json.dumps(list(a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()
def tensor_sha(value): return array_sha(value.detach().cpu().numpy())
def free_bound_mask(point,gradient,lower,upper,active):
    return active.bool()&~(((point<=lower+1e-12)&(gradient>0))|((point>=upper-1e-12)&(gradient<0)))
def bytes_used(folder): return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file()) if Path(folder).exists() else 0


def write_json(path,value,*,folder=REPORT_DIR,reserve=True,cap=None):
    path=Path(path).resolve();folder=Path(folder).resolve()
    if folder not in path.parents or path.suffix!='.json': raise IntegrityError('Probe writes must be small JSON inside its report folder')
    value=s.plain(value);size=len((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())
    cap=CONFIG['output_cap_bytes'] if cap is None else cap
    allowance=CONFIG['terminal_reserve_bytes'] if reserve else 0
    if bytes_used(folder)+size+allowance>cap: raise BudgetError('The1 MiB probe output budget would be exceeded')
    return s.write(path,value)


def space_guard(*,registration=False,folder=REPORT_DIR,free_bytes=None):
    free=shutil.disk_usage(s.ROOT).free if free_bytes is None else free_bytes
    remaining=CONFIG['output_cap_bytes'] if registration else max(0,CONFIG['output_cap_bytes']-bytes_used(folder))
    reserve=remaining+CONFIG['lock_meta_reserve_bytes']
    if free-reserve<CONFIG['free_floor_bytes']: raise BudgetError('Insufficient space above the2 GiB floor for the complete report reserve')
    return {'free_bytes':free,'reserved_bytes':reserve,'floor_bytes':CONFIG['free_floor_bytes']}


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
                current=s.read(path)
                if current.get('pid')==owner['pid'] and current.get('owner_token')==owner['owner_token']: path.unlink()
            except (FileNotFoundError,ValueError,UnicodeError): pass


def _inputs(case):
    """Read only raw_dn and the exact two cached parameter arrays."""
    verify_file(case['raw_file']);verify_file(case['state_file'])
    with np.load(case['raw_file']['path'],allow_pickle=False) as z: raw=z['raw_dn'].copy()
    provenance=case['provenance']
    if raw.shape!=(256,256) or str(raw.dtype)!='uint16' or array_sha(raw)!=case['raw_array_sha256']:
        raise IntegrityError('Original camera raw array hash/dtype/shape mismatch')
    source_dtype={'torch.float32':np.float32,'torch.float64':np.float64}.get(provenance['source_input_dtype'])
    if source_dtype is None or array_sha(raw.astype(source_dtype))!=provenance['inference_input_array_sha256']:
        raise IntegrityError('Cached v3 source-floating raw hash mismatch')
    with np.load(case['state_file']['path'],allow_pickle=False) as z:
        state=z['state'].copy();beta=z['beta'].copy()
    if state.shape!=(1,5,256,256) or beta.shape!=(1,3) or not (np.isfinite(state).all() and np.isfinite(beta).all()):
        raise IntegrityError('Cached same-raw state/beta shape or finite check failed')
    if provenance.get('truth_or_identity_used') is not False or provenance['raw_file_sha256']!=case['raw_file']['sha256']:
        raise IntegrityError('A raw-only same-raw frozen initializer receipt is required')
    return raw,state,beta


def _passed_contract(path,expected_sources,minimum_tests=1):
    value=s.read(path)
    tests=value.get('tests',value.get('checks',[]))
    if value.get('all_passed') is not True or len(tests)<minimum_tests or any(t.get('passed') is not True for t in tests):
        raise IntegrityError('Passed current CPU contracts are required')
    for name in expected_sources:
        if value.get('source_sha256',{}).get(name)!=s.sha(s.ROOT/name): raise IntegrityError('CPU contract source hash is stale: '+name)
    return file_ref(path)


def register(contract_path=None):
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    with owner_locks('registration-pending'):
        if (REPORT_DIR/'REGISTRATION.json').exists() or (REPORT_DIR/'REGISTRATION_SHA256.json').exists():
            raise FileExistsError('Probe registration is immutable')
        space=space_guard(registration=True)
        own_contract=_passed_contract(contract_path or REPORT_DIR/'CPU_CONTRACTS.json',OWN_SOURCES)
        helper_contract=_passed_contract(HELPER_CONTRACT,HELPER_SOURCES,15)
        old=s.read(V002/'QUICK_REGISTRATION.json');side=s.read(V002/'QUICK_REGISTRATION_SHA256.json')
        if Path(side['path']).resolve()!=(V002/'QUICK_REGISTRATION.json').resolve(): raise IntegrityError('Wrong v002 registration sidecar')
        verify_file(side);verify_sources(old['source_sha256'])
        terminal=s.read(V002/'QUICK_RESULT.json')
        summary=terminal['summary']
        if terminal.get('registration_sha256')!=side['sha256'] or summary['planned_cells']!=4 or summary['numerically_certified_cells']!=4:
            raise IntegrityError('Completed4/4-certified v002 metadata is required')
        refs=old['references']
        if tuple(v['case_id'] for v in refs)!=EXPECTED_CASES: raise IntegrityError('Only the same two old v002 development raws are allowed')
        cases=[]
        for ref in refs:
            if ref.get('cohort')!='development' or ref.get('input_kind')!='noise': raise IntegrityError('Only development noise raws are allowed')
            provenance=old['v3_initializer_provenance'][ref['case_id']]
            raw_path=(s.LEGACY/ref['raw_reference']['path']).resolve()
            case={'case_id':ref['case_id'],'mother_id':ref['mother_id'],
                'raw_file':{'path':str(raw_path),'sha256':ref['raw_reference']['sha256']},
                'raw_array_sha256':ref['raw_reference']['array_sha256'],
                'state_file':provenance['state_cache'],'provenance':provenance}
            raw,state,beta=_inputs(case)
            case['array_sha256']={'raw':array_sha(raw),'state':array_sha(state),'beta':array_sha(beta)}
            verify_file(provenance['receipt_journal']);cases.append(case)
        sources={name:s.sha(s.ROOT/name) for name in DEPENDENCY_SOURCES}
        selection=file_ref(s.ROOT/'reports/training_v4/run_001/backbone_selection.json')
        value={'schema':VERSION+'-registration','time':s.now(),'config':copy.deepcopy(CONFIG),
            'source_sha256':sources,'cases':cases,'contracts':[helper_contract,own_contract],
            'upstream_files':[file_ref(V002/name) for name in ('QUICK_REGISTRATION.json','QUICK_REGISTRATION_SHA256.json','QUICK_RESULT.json')],
            'selection':selection,'space_reserve':space,'new_complete_inverse_calls':0,
            'truth_loaded':False,'GPU_started':False,'final_payloads_opened':False}
        value['fingerprint']=s.canonical({'config':value['config'],'sources':sources,'cases':cases,'selection':selection})
        write_json(REPORT_DIR/'REGISTRATION.json',value)
        write_json(REPORT_DIR/'REGISTRATION_SHA256.json',file_ref(REPORT_DIR/'REGISTRATION.json'))
        return value


def verify_registration():
    side=s.read(REPORT_DIR/'REGISTRATION_SHA256.json')
    if Path(side['path']).resolve()!=(REPORT_DIR/'REGISTRATION.json').resolve(): raise IntegrityError('Wrong probe registration sidecar')
    verify_file(side);value=s.read(REPORT_DIR/'REGISTRATION.json')
    if value.get('schema')!=VERSION+'-registration' or value['config']!=CONFIG: raise IntegrityError('Frozen probe configuration changed')
    verify_sources(value['source_sha256'])
    for ref in value['contracts']+value['upstream_files']+[value['selection']]: verify_file(ref)
    for case in value['cases']:
        verify_file(case['raw_file']);verify_file(case['state_file']);verify_file(case['provenance']['receipt_journal'])
    if tuple(c['case_id'] for c in value['cases'])!=EXPECTED_CASES: raise IntegrityError('Probe case denominator changed')
    fingerprint=s.canonical({'config':value['config'],'sources':value['source_sha256'],'cases':value['cases'],'selection':value['selection']})
    if fingerprint!=value['fingerprint']: raise IntegrityError('Probe source/input/system registration fingerprint changed')
    return value,side['sha256']


class RouteBudget:
    def __init__(self,clock=time.perf_counter,synchronize=lambda:None,limit_s=120.):
        self.clock=clock;self.synchronize=synchronize;self.started=clock()
        self.end=self.started+min(limit_s,CONFIG['route_wall_s'])
        self.deadline=self.end-min(CONFIG['audit_reserve_s'],limit_s/2)
    def check(self):
        self.synchronize()
        if self.clock()>=self.deadline: raise TimeoutError('Fixed route wall budget exhausted')
    def audit(self): self.deadline=self.end


def bounded_pcg(normal,rhs,precondition,guard,*,rtol=1e-10,maxiter=2000):
    """A linear solve only; retain the latest finite x and exact failure count."""
    import torch
    x=torch.zeros_like(rhs);r=rhs.clone();iterations=0;history=[];restarts=0;status='MAX_ITERATIONS';error=None
    norm=float(torch.linalg.vector_norm(rhs));initial_norm=norm
    try:
        guard.check()
        if not np.isfinite(norm): raise ArithmeticError('Nonfinite tangent RHS')
        if norm==0: status='ZERO_RHS'
        else:
            z=precondition(r);guard.check();rz=(r*z).sum();direction=z.clone()
            if not torch.isfinite(rz) or float(rz)<=0: raise ArithmeticError('Nonpositive preconditioned residual')
            for iteration in range(1,maxiter+1):
                guard.check();product=normal(direction);guard.check();denominator=(direction*product).sum()
                if not torch.isfinite(denominator) or float(denominator)<=0: raise ArithmeticError('Nonpositive tangent curvature')
                candidate=x+(rz/denominator)*direction
                if not bool(torch.isfinite(candidate).all()): raise ArithmeticError('Nonfinite tangent iterate')
                x=candidate;iterations=iteration;r=r-(rz/denominator)*product
                recursive=float(torch.linalg.vector_norm(r))/norm
                if not np.isfinite(recursive): raise ArithmeticError('Nonfinite tangent residual')
                if iteration<=8 or iteration%100==0: history.append({'iteration':iteration,'recursive_relative_residual':recursive})
                if recursive<=rtol:
                    true=rhs-normal(x);guard.check()
                    if float(torch.linalg.vector_norm(true))/norm<=rtol:
                        status='CONVERGED_TRUE_RESIDUAL';break
                    r=true;z=precondition(r);rz=(r*z).sum();direction=z.clone();restarts+=1
                    if not torch.isfinite(rz) or float(rz)<=0: raise ArithmeticError('Residual-restart preconditioner breakdown')
                    continue
                z=precondition(r);guard.check();next_rz=(r*z).sum()
                if not torch.isfinite(next_rz) or float(next_rz)<=0: raise ArithmeticError('Preconditioner breakdown')
                direction=z+(next_rz/rz)*direction;rz=next_rz
    except (TimeoutError,ArithmeticError,RuntimeError,ValueError) as failure:
        status='TIMEOUT' if isinstance(failure,TimeoutError) else 'LINEAR_FAILURE'
        error={'type':type(failure).__name__,'message':str(failure)[:400]}
    guard.audit();true_norm=None;audit_error=None
    try:
        guard.check();true_norm=float(torch.linalg.vector_norm(rhs-normal(x)));guard.check()
    except (TimeoutError,ArithmeticError,RuntimeError) as failure: audit_error={'type':type(failure).__name__,'message':str(failure)[:400]}
    relative=None if true_norm is None else (true_norm/initial_norm if initial_norm else 0.)
    converged=status in ('CONVERGED_TRUE_RESIDUAL','ZERO_RHS') and relative is not None and relative<=rtol
    return x,{'status':status,'converged':converged,'iterations':iterations,'maxiter':maxiter,'rtol':rtol,
        'initial_rhs_norm':initial_norm,'true_residual_norm':true_norm,'true_relative_residual':relative,
        'true_residual_recomputed':true_norm is not None,'residual_restarts':restarts,
        'sampled_recursive_history':history,'error':error,'audit_error':audit_error}


class NativeCase:
    def __init__(self,case,device):
        import torch
        from . import measurement as m
        from solver import pack,unpack
        from repair_v3.solver import laplacian,regularization
        self.torch=torch;self.pack=pack;self.unpack=unpack;self.laplacian=laplacian
        self.p=m.physics(device);self.device=device;self.guard=None
        self.counts={'JVP':0,'VJP':0,'normal':0,'precondition':0}
        raw,state,beta=_inputs(case)
        for key,value in (('raw',raw),('state',state),('beta',beta)):
            if array_sha(value)!=case['array_sha256'][key]: raise IntegrityError('Registered input array changed: '+key)
        self.state=torch.as_tensor(state,device=device,dtype=torch.float64);self.beta=torch.as_tensor(beta,device=device,dtype=torch.float64)
        self.raw_e=self.p.raw_to_electrons(torch.as_tensor(raw,device=device,dtype=torch.float64)[None])
        self.W=m.fixed_weights(self.p,self.raw_e)
        self.strength=self.state.new_tensor(s.read(s.ROOT/'reports/training_v4/run_001/backbone_selection.json')['solver_config']['regularization'])
        original_jvp=self.p.jvp;original_vjp=self.p.vjp
        def jvp(*args):
            self.check();self.counts['JVP']+=1;value=original_jvp(*args);self.check();return value
        def vjp(*args):
            self.check();self.counts['VJP']+=1;value=original_vjp(*args);self.check();return value
        self.p.jvp=jvp;self.p.vjp=vjp
        d,db=self.p.preconditioner_diagonal(self.state,self.beta,self.W);self.D=pack(d,db)
        prediction=self.p(self.state,self.beta);g,gb=self.p.vjp(self.state,self.beta,self.W*(prediction-self.raw_e))
        _,reg_g=regularization(self.state,self.strength);self.gradient=pack(g+reg_g,gb)
        limits=self.p.cfg['temperature_range_K'];lo=self.state.new_tensor([(limits[0]-self.p.T0)/self.p.Tscale,-4.,-20.,-20.,-20.])
        hi=self.state.new_tensor([(limits[1]-self.p.T0)/self.p.Tscale,.3,20.,20.,20.])
        lower=pack(lo[None,:,None,None].expand_as(self.state),torch.full_like(self.beta,-20.))
        upper=pack(hi[None,:,None,None].expand_as(self.state),torch.full_like(self.beta,20.))
        point=pack(self.state,self.beta)
        if bool(((point<lower-1e-7)|(point>upper+1e-7)).any()): raise IntegrityError('Cached original state lies outside original bounds')
        active=torch.cat([self.state.new_ones(2),self.p.active]);active=pack(active[None,:,None,None].expand_as(self.state),torch.ones_like(self.beta)).bool()
        self.mask=free_bound_mask(point,self.gradient,lower,upper,active)
        self.rhs=-self.gradient*self.mask
        self.identity={'linearization_array_sha256':{name:tensor_sha(v) for name,v in
            (('state',self.state),('beta',self.beta),('W',self.W),('D',self.D),('rhs',self.rhs),('free_mask',self.mask),('strength',self.strength))},
            'free_coordinates':int(self.mask.sum()),'packed_coordinates':point.numel(),'active_coordinates':int(active.sum()),
            'damping':CONFIG['damping'],'calibration':CONFIG['calibration'],'same_system_both_routes':True}
        self.identity['system_fingerprint']=s.canonical(self.identity)
        self.setup_counts=dict(self.counts)
    def sync(self):
        if str(self.device).startswith('cuda'): self.torch.cuda.synchronize(self.device)
    def check(self):
        if self.guard is not None: self.guard.check()
    def normal(self,vector):
        self.check();self.counts['normal']+=1;v=vector*self.mask;dz,db=self.unpack(v,self.p.cfg['shape'])
        j=self.p.jvp(self.state,self.beta,dz,db);g,gb=self.p.vjp(self.state,self.beta,self.W*j)
        g+=self.strength[None,:,None,None]*self.laplacian(dz)
        answer=(self.pack(g,gb)+CONFIG['damping']*self.D*v)*self.mask
        self.check();return answer
    def build_preconditioner(self,route):
        from .coupled_preconditioner import build
        from repair_v3.solver import block_preconditioner
        self.check()
        if route=='coupled_fourier':
            pre,D,metadata=build(self.p,self.state,self.beta,self.W,self.strength,CONFIG['damping'])
            if not self.torch.equal(D,self.D): raise IntegrityError('The new helper changed the exact damping diagonal')
        elif route=='old_block_diagonal':
            active=self.state.new_tensor([1.,1.,0.,0.,1.])
            pre,_,=block_preconditioner(self.p,self.state,self.beta,self.W,active,self.strength,CONFIG['damping'])
            metadata={'method':'original repair_v3.solver.block_preconditioner','approximation_only':True}
        else: raise IntegrityError('Unknown frozen preconditioner route')
        self.check()
        def masked(vector):
            self.check();self.counts['precondition']+=1
            answer=pre(vector*self.mask)*self.mask;self.check();return answer
        return masked,metadata
    def linear_outputs(self,vector):
        torch=self.torch;self.check();dz,db=self.unpack(vector*self.mask,self.p.cfg['shape'])
        raw=self.p.jvp(self.state,self.beta,dz,db)
        jac=self.p.spectral_jacobian(self.state,self.beta)
        dC=self.p.op.object_blur((jac[:,:2]*dz[:,:2,None]).sum(1))
        C=self.p.op.object_blur(self.p.components(self.state,self.beta)[0])
        ratio=C[:,-2]/C[:,-1];idx=torch.searchsorted(self.p.color_ratio_lut,ratio.contiguous()).clamp(1,len(self.p.temperatures)-1)
        slope=(self.p.temperatures[idx]-self.p.temperatures[idx-1])/(self.p.color_ratio_lut[idx]-self.p.color_ratio_lut[idx-1])
        color=slope*(dC[:,-2]*C[:,-1]-C[:,-2]*dC[:,-1])/C[:,-1].square()
        available=(C[:,-1]>1)&(ratio>self.p.color_ratio_lut[0])&(ratio<self.p.color_ratio_lut[-1])
        template=self.p.op.object_blur(dz[:,4]*self.p.Sscale)
        rms=lambda a:float(a.square().mean().sqrt())
        answer={'raw_linear_RMS_e':rms(raw),'template_postblur_linear_RMS_e_ref':rms(template),
            'color_linear_RMS_K':rms(color) if bool(available.all()) else None,
            'color_linear_available_fraction':float(available.double().mean()),
            'latent_temperature_linear_RMS_K':rms(dz[:,0]*self.p.Tscale),'beta_linear_L2':float(db.norm())}
        self.check();return answer


def route_cell(system,route,*,clock=time.perf_counter,remaining_s=480.):
    before=dict(system.counts);guard=RouteBudget(clock,system.sync,min(120.,remaining_s));system.guard=guard
    pcg_started=False
    step=None;metadata=None;receipt={'status':'BUILD_FAILED','converged':False,'iterations':0,'pcg_started':False,
        'rtol':CONFIG['rtol'],'maxiter':CONFIG['maxiter'],'true_residual_norm':None,
        'true_relative_residual':None,'true_residual_recomputed':False}
    try:
        guard.check();pre,metadata=system.build_preconditioner(route)
        pcg_started=True
        step,receipt=bounded_pcg(system.normal,system.rhs,pre,guard,rtol=CONFIG['rtol'],maxiter=CONFIG['maxiter'])
    except (TimeoutError,ArithmeticError,RuntimeError,ValueError) as error:
        receipt.update(status='TIMEOUT' if isinstance(error,TimeoutError) else 'BUILD_FAILED',
            error={'type':type(error).__name__,'message':str(error)[:400]})
    guard.audit();outputs=None
    if step is not None:
        try: outputs=system.linear_outputs(step)
        except (TimeoutError,ArithmeticError,RuntimeError,ValueError) as error:
            receipt['linear_output_error']={'type':type(error).__name__,'message':str(error)[:400]}
    system.sync();elapsed=clock()-guard.started
    if elapsed>min(120.,remaining_s): receipt.update(converged=False,wall_budget_exceeded=True)
    receipt.update(route=route,pcg_started=pcg_started,wall_s=elapsed,wall_limit_s=min(120.,remaining_s),
        counts={name:system.counts[name]-before.get(name,0) for name in system.counts},
        preconditioner_metadata=metadata,linear_outputs=outputs,
        step_array_sha256=None if step is None else tensor_sha(step),step_saved=False,
        system_fingerprint=system.identity['system_fingerprint'])
    system.guard=None
    return receipt,step


def execute(registration,digest,*,folder=REPORT_DIR,device='cuda',case_factory=NativeCase,
            clock=time.perf_counter,guard_space=space_guard,verify=lambda:None):
    """Small orchestration seam also exercised by injected CPU fake operators."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);terminal=folder/'RESULT.json'
    if terminal.exists():
        value=s.read(terminal)
        if value.get('registration_sha256')!=digest: raise IntegrityError('Another registration owns the terminal result')
        for ref in value.get('cell_file_refs',[]): verify_file(ref)
        return value
    rows=[];pairs=[];errors=[];started=clock();recovered=(folder/'RUN_STARTED.json').exists()
    if recovered and s.read(folder/'RUN_STARTED.json').get('registration_sha256')!=digest:
        raise IntegrityError('Another registration owns the interrupted attempt')
    try:
        if not recovered:
            guard_space(folder=folder)
            write_json(folder/'RUN_STARTED.json',{'schema':VERSION,'registration_sha256':digest,'time':s.now(),
                'planned_cells':4,'whole_inverse_calls':0},folder=folder)
        for case_index,case in enumerate(registration['cases']):
            if recovered: break
            verify();guard_space(folder=folder);system=None;steps={};case_setup=clock()
            try: system=case_factory(case,device)
            except Exception as error:
                errors.append({'case_id':case['case_id'],'phase':'fixed_system_setup','type':type(error).__name__,'message':str(error)[:400]})
            if system is not None:
                write_json(folder/f'SYSTEM_{case_index:02d}.json',{'case_id':case['case_id'],'identity':system.identity,
                    'setup_counts':system.setup_counts,'setup_wall_s':clock()-case_setup},folder=folder)
            for route in ROUTES:
                row={'case_id':case['case_id'],'route':route,'attempted':system is not None,'registration_sha256':digest}
                if system is None: row.update(status='SYSTEM_SETUP_FAILED',converged=False,iterations=0,counts={})
                else:
                    remaining=480.-sum(r.get('wall_s',0.) for r in rows)
                    if remaining<=0:
                        row.update(status='NOT_ATTEMPTED_TOTAL_BUDGET',attempted=False,converged=False,iterations=0,counts={})
                    else:
                        receipt,step=route_cell(system,route,clock=clock,remaining_s=remaining);row.update(receipt);steps[route]=step
                rows.append(row);write_json(folder/f'CELL_{case_index:02d}_{route}.json',row,folder=folder)
            comparison={'case_id':case['case_id'],'registration_sha256':digest,'both_converged':all(r.get('converged',False) for r in rows[-2:]),
                'solution_difference':None,'linear_output_difference':None,'comparison_is_truth_error':False}
            if system is not None and all(steps.get(r) is not None for r in ROUTES):
                try:
                    difference=steps[ROUTES[0]]-steps[ROUTES[1]]
                    comparison['solution_difference']={'L2':float(difference.norm()),
                        'relative_to_old_L2':float(difference.norm()/steps[ROUTES[0]].norm().clamp_min(1e-30))}
                    comparison['linear_output_difference']=system.linear_outputs(difference)
                except Exception as error: comparison['error']={'type':type(error).__name__,'message':str(error)[:400]}
            if system is not None: comparison['case_total_counts']=dict(system.counts)
            pairs.append(comparison);write_json(folder/f'PAIR_{case_index:02d}.json',comparison,folder=folder)
            if system is not None:
                system.guard=None
                del steps,system
    except Exception as error:
        errors.append({'phase':'controller','type':type(error).__name__,'message':str(error)[:400]})
    if recovered:
        for path in sorted(folder.glob('CELL_*.json')):
            try:
                value=s.read(path)
                if value.get('registration_sha256')!=digest: raise IntegrityError('Interrupted cell belongs to another registration')
                if (value.get('case_id'),value.get('route')) in {(c['case_id'],r) for c in registration['cases'] for r in ROUTES}: rows.append(value)
            except Exception as error:
                errors.append({'phase':'interrupted_receipt','path':path.name,'type':type(error).__name__,'message':str(error)[:400]})
        pairs=[s.read(path) for path in sorted(folder.glob('PAIR_*.json')) if s.read(path).get('registration_sha256')==digest]
    supplied={(r['case_id'],r['route']):r for r in rows}
    complete=[supplied.get((case['case_id'],route),{'case_id':case['case_id'],'route':route,'attempted':False,
        'status':'NOT_ATTEMPTED_INTERRUPTED' if recovered else 'NOT_ATTEMPTED_CONTROLLER_FAILURE',
        'converged':False,'iterations':0,'counts':{}}) for case in registration['cases'] for route in ROUTES]
    value={'schema':VERSION+'-result','time':s.now(),'registration_sha256':digest,
        'status':'INTERRUPTED_ATTEMPT_RETAINED' if recovered else ('PROBE_COMPLETE_WITH_FAILURES' if errors or not all(r['converged'] for r in complete) else 'PROBE_COMPLETE'),
        'planned_cells':4,'returned_cells':len(complete),'attempted_cells':sum(r['attempted'] for r in complete),
        'converged_cells':sum(r['converged'] for r in complete),'failed_or_unattempted_cells':sum(not r['converged'] for r in complete),
        'rows':complete,'pairs':pairs,'errors':errors,'full_denominator_preserved':True,
        'route_wall_limit_s':120.,'planned_total_route_wall_s':480.,'wall_s_including_overhead':clock()-started,
        'whole_inverse_calls':0,'count_call_used':False,'truth_used':False,'final_payloads_opened':False,
        'device':device,'step_arrays_saved':False,'output_bytes_before_terminal':bytes_used(folder)}
    value['linear_error_scope']='differences of fixed-linearization outputs between the two steps; not truth error or nonlinear estimator accuracy'
    value['operator_counts_complete']=len(pairs)==len(registration['cases']) and all('case_total_counts' in pair for pair in pairs)
    value['total_operator_counts']=({key:sum(pair['case_total_counts'].get(key,0) for pair in pairs)
        for key in ('JVP','VJP','normal','precondition')} if value['operator_counts_complete'] else None)
    value['recorded_cell_operator_counts']={key:sum(row.get('counts',{}).get(key,0) for row in complete)
        for key in ('JVP','VJP','normal','precondition')}
    value['PCG_calls']=sum(bool(row.get('pcg_started')) for row in complete)
    value['cell_file_refs']=[file_ref(path) for path in sorted(folder.glob('*.json'))
        if path.name.startswith(('CELL_','SYSTEM_','PAIR_'))]
    write_json(terminal,value,folder=folder,reserve=False)
    return value


def launch_failure(error,*,folder=REPORT_DIR,operation='run'):
    value={'schema':VERSION+'-launch-failure','time':s.now(),'status':'FAILED_BEFORE_PROBE','operation':operation,
        'planned_cells':4,'returned_cells':4,'attempted_cells':0,'failed_or_unattempted_cells':4,
        'rows':[{'case_id':case,'route':route,'attempted':False,'converged':False,'status':'NOT_STARTED_LAUNCH_REJECTED'}
            for case in EXPECTED_CASES for route in ROUTES],
        'error':{'type':type(error).__name__,'message':str(error)[:400]},'whole_inverse_calls':0,
        'truth_used':False,'step_arrays_saved':False,'final_payloads_opened':False}
    write_json(Path(folder)/('LAUNCH_FAILURE_'+uuid.uuid4().hex+'.json'),value,folder=folder,reserve=False)
    return value


def run(device='cuda'):
    if device!='cuda': raise ValueError('The registered native probe requires explicit --device cuda; CPU tests use injected fixtures')
    try:
        registration,digest=verify_registration()
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
        'registration_sha256':s.sha(REPORT_DIR/'REGISTRATION.json') if (REPORT_DIR/'REGISTRATION.json').exists() else None,'planned_cells':4,
        'converged_cells':result.get('converged_cells'),'whole_inverse_calls':0}),flush=True)
    if result.get('status')=='FAILED_BEFORE_PROBE': raise SystemExit(2)


if __name__=='__main__': main()
