"""Small CPU controller fixtures; no actual development arrays are opened."""
from .. import storage as s
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .. import v3_tangent_probe as probe
from config import default_config
from physics import SpectralPhysics
from unittest.mock import patch
import copy
import inspect
import json
from pathlib import Path
import tempfile
import time
import torch
import numpy as np


class Clock:
    def __init__(self): self.value=0.
    def __call__(self): return self.value
    def advance(self,value): self.value+=value


class FakeCase:
    def __init__(self,case=None,device=None,*,fail_route=None,clock=None):
        self.A=torch.tensor([[4.,1.,0.],[1.,3.,.2],[0.,.2,2.]],dtype=torch.float64)
        self.rhs=torch.tensor([[1.,2.,3.]],dtype=torch.float64);self.guard=None
        self.counts={'JVP':0,'VJP':0,'normal':0,'precondition':0};self.setup_counts=dict(self.counts)
        self.identity={'system_fingerprint':'same-fixed-system'};self.fail_route=fail_route;self.clock=clock
    def sync(self): pass
    def normal(self,vector):
        if self.guard: self.guard.check()
        self.counts['normal']+=1;self.counts['JVP']+=1;self.counts['VJP']+=1
        value=vector@self.A
        if self.guard: self.guard.check()
        return value
    def build_preconditioner(self,route):
        if route==self.fail_route: raise ArithmeticError('injected preconditioner build failure')
        def apply(value):
            self.counts['precondition']+=1
            if self.guard: self.guard.check()
            return torch.linalg.solve(self.A,value.T).T if route=='coupled_fourier' else value
        return apply,{'fixture_only':True,'route':route}
    def linear_outputs(self,vector): return {'raw_linear_RMS_e':float(vector.norm()),'fixture_only':True}


def run():
    started=time.perf_counter();torch.set_num_threads(2);probe.REPORT_DIR.mkdir(parents=True,exist_ok=True)
    upstream=s.read(probe.V002/'QUICK_REGISTRATION.json')
    frozen_before={name:s.sha(s.ROOT/name) for name in upstream['source_sha256']}
    helper_before={name:s.sha(s.ROOT/name) for name in probe.HELPER_SOURCES}
    tests=[]
    def check(name,condition,**evidence): tests.append({'name':name,'passed':bool(condition),**s.plain(evidence)})
    def raises(kind,fn):
        try: fn()
        except kind: return True
        return False
    with tempfile.TemporaryDirectory(prefix='_cpu_',dir=probe.REPORT_DIR) as temporary:
        base=Path(temporary)
        locks=(base/'global.json',base/'local.json')
        with probe.owner_locks('fixture',locks):
            lock_exists=all(path.exists() for path in locks)
        check('owner_locks_acquire_and_remove_only_their_own_files',lock_exists and not any(path.exists() for path in locks))
        s.write(locks[1],{'pid':123,'owner_token':'foreign'})
        check('partial_lock_failure_preserves_foreign_owner_and_releases_own_global',
            raises(FileExistsError,lambda:_enter_locks(locks)) and not locks[0].exists()
            and s.read(locks[1])['owner_token']=='foreign')
        locks[1].unlink()
        with probe.owner_locks('fixture',locks):
            replacement={'pid':999,'owner_token':'replacement'}
            s.write(locks[1],replacement,replace=True)
        check('replaced_lock_not_deleted_by_old_owner',s.read(locks[1])==replacement and not locks[0].exists())
        locks[1].unlink()
        check('complete_report_and_lock_meta_reserve_enforced',
            raises(probe.BudgetError,lambda:probe.space_guard(registration=True,
                free_bytes=probe.CONFIG['free_floor_bytes']+2**20+4095))
            and probe.space_guard(registration=True,free_bytes=probe.CONFIG['free_floor_bytes']+2**20+4096)['reserved_bytes']==2**20+4096)
        check('oversized_write_refused_before_creating_a_payload',raises(probe.BudgetError,
            lambda:probe.write_json(base/'too_large.json',{'value':'x'*2000},folder=base,reserve=False,cap=100))
            and not (base/'too_large.json').exists())
        check('non_JSON_payload_and_escape_are_rejected',raises(probe.IntegrityError,
            lambda:probe.write_json(base/'step.npz',{},folder=base)) and raises(probe.IntegrityError,
            lambda:probe.write_json(base.parent/'escape.json',{},folder=base)))

        system=FakeCase();guard=probe.RouteBudget()
        answer,receipt=probe.bounded_pcg(system.normal,system.rhs,lambda r:r,guard)
        reference=torch.linalg.solve(system.A,system.rhs.T).T
        check('bounded_PCG_matches_dense3_and_recomputes_actual_residual',receipt['converged']
            and receipt['true_residual_recomputed'] and receipt['true_relative_residual']<=1e-10
            and torch.allclose(answer,reference,atol=1e-12,rtol=1e-12),PCG=receipt)
        _,zero=probe.bounded_pcg(system.normal,torch.zeros_like(system.rhs),lambda r:r,probe.RouteBudget())
        check('zero_rhs_is_explicit_and_still_true_residual_checked',zero['status']=='ZERO_RHS'
            and zero['iterations']==0 and zero['true_relative_residual']==0.)
        _,negative=probe.bounded_pcg(lambda r:-r,system.rhs,lambda r:r,probe.RouteBudget())
        check('nonpositive_curvature_retains_failed_linear_denominator',negative['status']=='LINEAR_FAILURE'
            and not negative['converged'] and negative['true_residual_recomputed'])
        clock=Clock();timeout_guard=probe.RouteBudget(clock)
        def slow_pre(vector): clock.advance(119.);timeout_guard.check();return vector
        _,timed=probe.bounded_pcg(system.normal,system.rhs,slow_pre,timeout_guard)
        check('work_timeout_reserves_final_true_residual_audit',timed['status']=='TIMEOUT'
            and not timed['converged'] and timed['true_residual_recomputed'] and clock()==119.,PCG=timed)
        _,limited=probe.bounded_pcg(system.normal,system.rhs,lambda r:r,probe.RouteBudget(),maxiter=1)
        check('iteration_cap_is_not_relabelled_as_convergence',limited['status']=='MAX_ITERATIONS'
            and limited['iterations']==1 and not limited['converged'] and limited['true_residual_recomputed'])

        registration={'cases':[{'case_id':name} for name in probe.EXPECTED_CASES]}
        no_space=lambda **kwargs:None
        good_folder=base/'good'
        value=probe.execute(registration,'fixture',folder=good_folder,device='cpu-fixture',case_factory=FakeCase,guard_space=no_space)
        check('full_two_raw_two_route_denominator_and_same_system_kept',value['planned_cells']==value['returned_cells']==4
            and value['converged_cells']==4 and value['PCG_calls']==4 and value['operator_counts_complete']
            and all(row['system_fingerprint']=='same-fixed-system' for row in value['rows'])
            and len(value['pairs'])==2 and all(pair['both_converged'] for pair in value['pairs'])
            and value['whole_inverse_calls']==0 and not value['count_call_used'])
        called=[0]
        def forbidden_factory(*args): called[0]+=1;raise AssertionError('terminal rerun attempted work')
        reused=probe.execute(registration,'fixture',folder=good_folder,device='cpu-fixture',case_factory=forbidden_factory,guard_space=no_space)
        check('immutable_terminal_repeat_performs_no_more_linear_work',called[0]==0 and reused==value)
        cell=next(good_folder.glob('CELL_*.json'));original=cell.read_bytes();cell.write_bytes(original+b' ')
        check('terminal_repeat_detects_modified_cell_receipt',raises(probe.IntegrityError,
            lambda:probe.execute(registration,'fixture',folder=good_folder,case_factory=FakeCase,guard_space=no_space)))
        cell.write_bytes(original)
        mixed=probe.execute(registration,'mixed',folder=base/'mixed',device='cpu-fixture',
            case_factory=lambda *args:FakeCase(fail_route='coupled_fourier'),guard_space=no_space)
        check('preconditioner_failure_does_not_remove_cells_or_other_route',mixed['returned_cells']==4
            and mixed['converged_cells']==2 and mixed['failed_or_unattempted_cells']==2
            and all(row['status']=='BUILD_FAILED' for row in mixed['rows'] if row['route']=='coupled_fourier'))
        def failed_factory(*args): raise ValueError('injected setup failure')
        failed=probe.execute(registration,'failed',folder=base/'failed',device='cpu-fixture',case_factory=failed_factory,guard_space=no_space)
        check('all_setup_failures_retain_four_planned_cells',failed['returned_cells']==4 and failed['attempted_cells']==0
            and failed['failed_or_unattempted_cells']==4 and not failed['operator_counts_complete'])
        interrupted=base/'interrupted';interrupted.mkdir()
        s.write(interrupted/'RUN_STARTED.json',{'registration_sha256':'interrupted'})
        s.write(interrupted/'CELL_00_old_block_diagonal.json',{'registration_sha256':'interrupted',
            'case_id':probe.EXPECTED_CASES[0],'route':'old_block_diagonal','attempted':True,
            'converged':False,'status':'MAX_ITERATIONS','iterations':3,'counts':{'normal':4}})
        recovered=probe.execute(registration,'interrupted',folder=interrupted,device='cpu-fixture',case_factory=forbidden_factory,guard_space=no_space)
        check('partial_attempt_is_closed_without_rerun_or_hidden_extra_budget',called[0]==0
            and recovered['status']=='INTERRUPTED_ATTEMPT_RETAINED' and recovered['returned_cells']==4
            and recovered['attempted_cells']==1 and recovered['failed_or_unattempted_cells']==4
            and recovered['recorded_cell_operator_counts']['normal']==4 and not recovered['operator_counts_complete'])
        check('partial_attempt_from_another_registration_is_rejected',raises(probe.IntegrityError,
            lambda:probe.execute(registration,'other',folder=interrupted,guard_space=no_space)))
        def fail_space(**kwargs): raise probe.BudgetError('injected low disk')
        disk=probe.execute(registration,'disk',folder=base/'disk',device='cpu-fixture',case_factory=forbidden_factory,guard_space=fail_space)
        check('low_disk_failure_still_delivers_all_four_failure_rows',called[0]==0 and disk['returned_cells']==4
            and disk['attempted_cells']==0 and disk['failed_or_unattempted_cells']==4)
        rejected=probe.launch_failure(probe.IntegrityError('injected registration hash failure'),folder=base/'launch')
        check('launch_rejection_keeps_four_not_started_cells_without_terminal_overwrite',
            rejected['planned_cells']==rejected['returned_cells']==4 and rejected['attempted_cells']==0
            and rejected['failed_or_unattempted_cells']==4 and not (base/'launch/RESULT.json').exists())
        damaged=base/'damaged';damaged.mkdir()
        s.write(damaged/'RUN_STARTED.json',{'registration_sha256':'damaged'})
        (damaged/'CELL_bad.json').write_text('{bad',encoding='utf8')
        damaged_result=probe.execute(registration,'damaged',folder=damaged,device='cpu-fixture',
            case_factory=forbidden_factory,guard_space=no_space)
        check('damaged_partial_receipt_is_quarantined_without_dropping_denominator',
            called[0]==0 and damaged_result['returned_cells']==4 and damaged_result['attempted_cells']==0
            and damaged_result['failed_or_unattempted_cells']==4 and len(damaged_result['errors'])==1)

        #Exercise registration/verification with tiny JSON stand-ins and an
        #in-memory input injector. No native raw or parameter array is stored.
        reg_folder=base/'registration';upstream_folder=base/'upstream';upstream_folder.mkdir()
        raw_file=base/'raw_standin.json';state_file=base/'state_standin.json';journal=base/'journal_standin.json'
        for path in (raw_file,state_file,journal): s.write(path,{'fixture_only':True})
        tiny_raw=np.zeros((2,2),dtype=np.uint16);tiny_state=np.zeros((1,5,2,2),dtype=np.float32);tiny_beta=np.zeros((1,3),dtype=np.float32)
        refs=[];provenance={}
        for index,cid in enumerate(probe.EXPECTED_CASES):
            refs.append({'case_id':cid,'mother_id':f'fixture_{index}','cohort':'development','input_kind':'noise',
                'raw_reference':{'path':str(raw_file),'sha256':s.sha(raw_file),'array_sha256':probe.array_sha(tiny_raw)}})
            provenance[cid]={'state_cache':probe.file_ref(state_file),'receipt_journal':probe.file_ref(journal),
                'raw_file_sha256':s.sha(raw_file),'truth_or_identity_used':False,'source_input_dtype':'torch.float32',
                'inference_input_array_sha256':probe.array_sha(tiny_raw.astype(np.float32))}
        old={'source_sha256':{'physics.py':s.sha(s.ROOT/'physics.py')},'references':refs,'v3_initializer_provenance':provenance}
        s.write(upstream_folder/'QUICK_REGISTRATION.json',old)
        old_sha=s.sha(upstream_folder/'QUICK_REGISTRATION.json')
        s.write(upstream_folder/'QUICK_REGISTRATION_SHA256.json',probe.file_ref(upstream_folder/'QUICK_REGISTRATION.json'))
        s.write(upstream_folder/'QUICK_RESULT.json',{'registration_sha256':old_sha,'summary':{'planned_cells':4,'numerically_certified_cells':4}})
        helper_fixture=base/'helper_contract.json'
        s.write(helper_fixture,{'all_passed':True,'tests':[{'passed':True}]*15,'source_sha256':helper_before})
        own_fixture=base/'own_contract.json'
        s.write(own_fixture,{'all_passed':True,'tests':[{'passed':True}],'source_sha256':probe.source_hashes()})
        original_locks=probe.owner_locks
        with patch.object(probe,'REPORT_DIR',reg_folder),patch.object(probe,'V002',upstream_folder),\
             patch.object(probe,'HELPER_CONTRACT',helper_fixture),\
             patch.object(probe,'_inputs',return_value=(tiny_raw,tiny_state,tiny_beta)),\
             patch.object(probe,'owner_locks',side_effect=lambda digest:original_locks(digest,(base/'register_global.json',base/'register_local.json'))):
            registered=probe.register(own_fixture)
            checked,digest=probe.verify_registration()
            duplicate_rejected=raises(FileExistsError,lambda:probe.register(own_fixture))
        check('metadata_only_registration_roundtrip_freezes_exact_dependencies_and_is_immutable',
            registered==checked and len(registered['cases'])==2 and len(registered['contracts'])==2
            and set(registered['source_sha256'])==set(probe.DEPENDENCY_SOURCES)
            and digest==s.sha(reg_folder/'REGISTRATION.json') and duplicate_rejected)

        #Actual32-pixel-side physics only tests the fixed masked operator and
        #its adjoint. It runs no full3075-coordinate PCG or complete inverse.
        p=SpectralPhysics(default_config(32),dtype=torch.float64)
        state=torch.zeros((1,5,32,32),dtype=torch.float64);state[:,0]=.1023;state[:,1]=np.log(.53)
        beta=torch.tensor([[.02,.001,-.002]],dtype=torch.float64)
        raw=(p(state,beta)/p.adc_step+p.cfg['black_level_DN'])[0].detach().numpy()
        case={'array_sha256':{key:probe.array_sha(v) for key,v in (('raw',raw),('state',state.numpy()),('beta',beta.numpy()))}}
        from .. import measurement
        with patch.object(probe,'_inputs',return_value=(raw,state.numpy(),beta.numpy())),patch.object(measurement,'physics',return_value=p):
            native=probe.NativeCase(case,'cpu')
        generator=torch.Generator(device='cpu').manual_seed(20261015)
        a=torch.randn(native.rhs.shape,dtype=torch.float64,generator=generator)*native.mask
        b=torch.randn(native.rhs.shape,dtype=torch.float64,generator=generator)*native.mask
        Aa=native.normal(a);Ab=native.normal(b)
        symmetry=abs(float((a*Ab).sum()-(Aa*b).sum()))/max(abs(float((a*Ab).sum())),1.)
        check('actual_CPU32_exact_masked_tangent_is_positive_and_self_adjoint',float((a*Aa).sum())>0 and symmetry<1e-11,
            self_adjoint_relative_error=symmetry,packed_dimension=a.numel(),full_inverse_calls=0)
        point=torch.tensor([[-1.,1.,0.,0.]],dtype=torch.float64)
        gradient=torch.tensor([[2.,-2.,-3.,3.]],dtype=torch.float64)
        mask=probe.free_bound_mask(point,gradient,torch.full_like(point,-1.),torch.ones_like(point),torch.tensor([[1,1,1,0]]))
        check('fixed_free_bound_mask_uses_outward_gradient_sign_and_active_fields',mask.tolist()==[[False,False,True,False]])
        #A single active bound is removed to test F P F on the actual coupled
        #operator. No constrained nonlinear update is performed.
        native.mask[0,0]=False
        pre,_=native.build_preconditioner('coupled_fourier')
        outside=torch.zeros_like(a);outside[0,0]=1.
        masked_result=pre(a)
        check('coupled_application_is_mask_P_mask_not_one_sided_mask',torch.count_nonzero(pre(outside))==0
            and masked_result[0,0]==0 and torch.count_nonzero(masked_result[~native.mask])==0)
        check('no_step_array_payloads_written_and_fixture_reports_small',not list(base.rglob('*.npz'))
            and not list(base.rglob('*.npy')) and probe.bytes_used(base)<2**20)

    frozen_after={name:s.sha(s.ROOT/name) for name in upstream['source_sha256']}
    check('upstream_frozen_sources_and_coupled_helper_unchanged',frozen_before==frozen_after
        and all(frozen_after[name]==digest for name,digest in upstream['source_sha256'].items())
        and helper_before=={name:s.sha(s.ROOT/name) for name in probe.HELPER_SOURCES})
    check('formal_protocol_constants_remain_fixed',probe.CONFIG['rtol']==1e-10 and probe.CONFIG['maxiter']==2000
        and probe.CONFIG['damping']==1e-3 and probe.CONFIG['route_wall_s']==120.
        and probe.CONFIG['total_route_wall_s']==480. and probe.CONFIG['output_cap_bytes']==2**20
        and list(inspect.signature(probe.NativeCase).parameters)==['case','device'])
    result={'schema':probe.VERSION+'-CPU-contracts','time':s.now(),'all_passed':all(t['passed'] for t in tests),
        'tests':tests,'source_sha256':probe.source_hashes(),'helper_source_sha256':helper_before,
        'upstream_source_sha256':frozen_after,'GPU_used':False,'whole_inverse_calls':0,
        'real_development_raw_or_parameter_arrays_read':False,'truth_arrays_read':False,
        'scientific_probe_started':False,'fixture_max_native_side':32,'fixture_dense_reference_dimension':3,
        'CPU_wall_s':time.perf_counter()-started}
    number=len(list(probe.REPORT_DIR.glob('CPU_CONTRACTS_ATTEMPT_*.json')))+1
    probe.write_json(probe.REPORT_DIR/f'CPU_CONTRACTS_ATTEMPT_{number:03d}.json',result)
    if result['all_passed']:
        path=probe.REPORT_DIR/'CPU_CONTRACTS.json'
        if path.exists():
            #Prior attempts remain immutable; replace only this task-local
            #latest-pass contract pointer after a source repair.
            s.write(path,result,replace=True)
        else: probe.write_json(path,result)
    print(json.dumps({'all_passed':result['all_passed'],'tests':len(tests),
        'failures':[t for t in tests if not t['passed']],
        'source_sha256':result['source_sha256'],'CPU_wall_s':result['CPU_wall_s']},indent=2),flush=True)
    return result


def _enter_locks(paths):
    with probe.owner_locks('fixture',paths): pass


if __name__=='__main__': raise SystemExit(0 if run()['all_passed'] else 2)
