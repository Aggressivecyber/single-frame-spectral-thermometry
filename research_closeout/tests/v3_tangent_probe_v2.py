"""Small CPU contracts for two new routes and immutable reused controls."""
from .. import storage as s
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .. import v3_tangent_probe_v2 as probe
from .. import coupled_preconditioner_v2 as helper
from unittest.mock import patch
import argparse
import copy
import inspect
import json
from pathlib import Path
import tempfile
import time
import torch


class FakeCase:
    def __init__(self,case,device,*,routes=None,fail_build=False):
        self.A=torch.tensor([[4.,1.,0.],[1.,3.,.2],[0.,.2,2.]],dtype=torch.float64)
        self.rhs=torch.tensor([[1.,2.,3.]],dtype=torch.float64)
        self.D=torch.ones_like(self.rhs);self.mask=torch.ones_like(self.rhs,dtype=torch.bool)
        self.guard=None;self.routes=routes;self.fail_build=fail_build
        self.identity=copy.deepcopy(case['fixture_identity'])
        self.counts={'JVP':0,'VJP':0,'normal':0,'precondition':0};self.setup_counts=dict(self.counts)
    def sync(self): pass
    def normal(self,vector):
        if self.guard: self.guard.check()
        self.counts['normal']+=1;self.counts['JVP']+=1;self.counts['VJP']+=1
        return vector@self.A
    def build_preconditioner(self,route):
        if self.routes is not None: self.routes.append(route)
        if route!=probe.ROUTE: raise AssertionError('A reused v1 route was executed')
        if self.fail_build: raise ArithmeticError('injected v2 backend build failure')
        def precondition(vector):
            self.counts['precondition']+=1
            if self.guard: self.guard.check()
            return torch.linalg.solve(self.A,vector.T).T
        return precondition,{'fixture_only':True,'backend_batch_limit':64}
    def linear_outputs(self,vector):
        rms=float(vector.square().mean().sqrt())
        return {'raw_linear_RMS_e':rms,'template_postblur_linear_RMS_e_ref':2*rms,
            'color_linear_RMS_K':3*rms,'latent_temperature_linear_RMS_K':4*rms,
            'beta_linear_L2':float(vector.norm())}


def fixture_registration(base):
    cases=[];controls=[]
    for index,cid in enumerate(probe.v1.EXPECTED_CASES):
        identity={'fixture_only':True,'system_fingerprint':'fixed-system-'+str(index),
            'linearization_array_sha256':{'rhs':'same','D':'same','free_mask':'same'}}
        case={'case_id':cid,'fixture_identity':identity};cases.append(case)
        old={'case_id':cid,'route':'old_block_diagonal','converged':True,
            'true_relative_residual':1e-12,'true_residual_norm':1e-9,
            'true_residual_recomputed':True,'step_saved':False,
            'system_fingerprint':identity['system_fingerprint'],
            'linear_outputs':{'raw_linear_RMS_e':1.,'template_postblur_linear_RMS_e_ref':2.,
                'color_linear_RMS_K':3.,'latent_temperature_linear_RMS_K':4.,'beta_linear_L2':5.}}
        cell=base/f'old_control_{index}.json';system=base/f'old_system_{index}.json'
        s.write(cell,old);s.write(system,{'identity':identity})
        controls.append({'case_id':cid,'old_row':old,'system_identity':identity,
            'files':{'cell':probe.v1.file_ref(cell),'system':probe.v1.file_ref(system)}})
    return {'cases':cases,'reused_controls':controls}


def run():
    started=time.perf_counter();torch.set_num_threads(2);tests=[]
    probe.REPORT_DIR.mkdir(parents=True,exist_ok=True)
    upstream=s.read(probe.v1.REPORT_DIR/'REGISTRATION.json')
    frozen_before={name:s.sha(s.ROOT/name) for name in upstream['source_sha256']}
    receipt_paths=[probe.v1.REPORT_DIR/name for name in ('REGISTRATION.json','REGISTRATION_SHA256.json','RESULT.json',
        'CELL_00_old_block_diagonal.json','CELL_01_old_block_diagonal.json','SYSTEM_00.json','SYSTEM_01.json')]
    receipt_before={str(path):s.sha(path) for path in receipt_paths}
    runtime_before={name:s.sha(s.ROOT/name) for name in probe.RUNTIME_SOURCES}
    def check(name,passed,**evidence): tests.append({'name':name,'passed':bool(passed),**s.plain(evidence)})
    def raises(kind,fn):
        try: fn()
        except kind: return True
        return False
    no_space=lambda **kwargs:{'fixture_only':True}
    with tempfile.TemporaryDirectory(prefix='_cpu_',dir=probe.REPORT_DIR) as temporary:
        base=Path(temporary);registration=fixture_registration(base)
        locks=(base/'global.json',base/'local.json')
        def enter():
            with probe.owner_locks('fixture',locks): pass
        with probe.owner_locks('fixture',locks): owned=all(path.exists() for path in locks)
        check('owner_locks_release_only_owned_global_and_local_files',owned and not any(path.exists() for path in locks))
        foreign={'pid':123,'owner_token':'foreign'};s.write(locks[1],foreign)
        check('foreign_local_lock_refuses_execution_and_releases_own_global',raises(FileExistsError,enter)
            and not locks[0].exists() and s.read(locks[1])==foreign)
        locks[1].unlink()
        with probe.owner_locks('fixture',locks): s.write(locks[1],foreign,replace=True)
        check('replaced_owner_lock_is_preserved',s.read(locks[1])==foreign and not locks[0].exists());locks[1].unlink()
        floor=probe.CONFIG['free_floor_bytes'];reserve=2**20+4096
        check('registration_requires_full1MiB_plus_lock_meta_above2GiB_floor',
            raises(probe.v1.BudgetError,lambda:probe.space_guard(registration=True,free_bytes=floor+reserve-1))
            and probe.space_guard(registration=True,free_bytes=floor+reserve)['reserved_bytes']==reserve)
        check('oneMiB_output_cap_and_JSON_only_paths_are_enforced',
            raises(probe.v1.BudgetError,lambda:probe.write_json(base/'large.json',{'x':'a'*2000},folder=base,cap=100,reserve=False))
            and raises(probe.v1.IntegrityError,lambda:probe.write_json(base/'step.npy',{},folder=base))
            and raises(probe.v1.IntegrityError,lambda:probe.write_json(base.parent/'escape.json',{},folder=base))
            and not (base/'large.json').exists())

        routes=[];good_folder=base/'good'
        value=probe.execute(registration,'good',folder=good_folder,device='cpu-fixture',guard_space=no_space,
            case_factory=lambda *args:FakeCase(*args,routes=routes))
        check('exactly_two_new_v2_routes_solve_dense3_with_true_residuals_and_old_controls_reused',
            routes==[probe.ROUTE]*2 and value['PCG_calls']==2 and value['converged_new_cells']==2
            and value['returned_new_cells']==2 and value['reused_old_routes_executed_this_run']==0
            and value['v1_failed_coupled_rows_repeated']==0 and len(value['reused_old_block_controls'])==2
            and all(row['true_residual_recomputed'] and row['true_relative_residual']<=1e-10
                and row['system_identity_match'] and not row['native_GPU_qualified'] for row in value['rows']))
        check('CPU_success_does_not_claim_CUDA_or_direct_step_difference',
            value['direct_step_difference_available'] is False and not value['step_arrays_saved']
            and all(row['comparison']['direct_solution_difference'] is None
                and row['comparison']['direct_linear_output_difference'] is None
                and not row['comparison']['interval_arithmetic_certified'] for row in value['rows']))
        calls=[0]
        def forbidden_factory(*args): calls[0]+=1;raise AssertionError('An immutable/partial attempt was rerun')
        repeated=probe.execute(registration,'good',folder=good_folder,device='cpu-fixture',
            case_factory=forbidden_factory,guard_space=no_space)
        check('terminal_repeat_returns_immutable_receipt_without_any_new_operator',repeated==value and calls[0]==0)
        check('terminal_other_registration_is_rejected',raises(probe.v1.IntegrityError,
            lambda:probe.execute(registration,'other',folder=good_folder,guard_space=no_space)))
        cell=good_folder/'CELL_00.json';original=cell.read_bytes();cell.write_bytes(original+b' ')
        check('modified_terminal_cell_hash_is_detected',raises(probe.v1.IntegrityError,
            lambda:probe.execute(registration,'good',folder=good_folder,guard_space=no_space)))
        cell.write_bytes(original)

        def mismatched(*args):
            system=FakeCase(*args,routes=routes);system.identity['linearization_array_sha256']['rhs']='changed'
            return system
        count=len(routes)
        mismatch=probe.execute(registration,'mismatch',folder=base/'mismatch',device='cpu-fixture',
            case_factory=mismatched,guard_space=no_space)
        check('any_original_system_array_hash_mismatch_refuses_both_solves_before_build',
            len(routes)==count and mismatch['PCG_calls']==0 and mismatch['returned_new_cells']==2
            and mismatch['failed_or_unattempted_new_cells']==2
            and all(not row['system_identity_match'] and not row['attempted'] for row in mismatch['rows']))
        def failed_factory(*args): raise ValueError('injected setup failure')
        failed=probe.execute(registration,'failed',folder=base/'failed',device='cpu-fixture',
            case_factory=failed_factory,guard_space=no_space)
        check('all_setup_failures_keep_two_new_and_two_reused_control_denominators',
            failed['returned_new_cells']==failed['failed_or_unattempted_new_cells']==2
            and failed['attempted_new_cells']==failed['PCG_calls']==0
            and len(failed['reused_old_block_controls'])==2 and not failed['operator_counts_complete'])
        build_failed=probe.execute(registration,'buildfailed',folder=base/'buildfailed',device='cpu-fixture',
            case_factory=lambda *args:FakeCase(*args,fail_build=True),guard_space=no_space)
        check('build_failure_is_retained_with_zero_PCG_and_no_old_route_retry',
            build_failed['attempted_new_cells']==2 and build_failed['PCG_calls']==0
            and build_failed['failed_or_unattempted_new_cells']==2
            and all(row['status']=='BUILD_FAILED' and row['counts']['JVP']==0 for row in build_failed['rows']))

        interrupted=base/'interrupted';interrupted.mkdir()
        s.write(interrupted/'RUN_STARTED.json',{'registration_sha256':'partial'})
        partial=copy.deepcopy(value['rows'][0]);partial.update(registration_sha256='partial',converged=False,status='MAX_ITERATIONS')
        s.write(interrupted/'CELL_00.json',partial)
        recovered=probe.execute(registration,'partial',folder=interrupted,device='cpu-fixture',
            case_factory=forbidden_factory,guard_space=no_space)
        check('partial_attempt_preserves_completed_row_and_closes_remaining_denominator_without_rerun',
            calls[0]==0 and recovered['status']=='INTERRUPTED_ATTEMPT_RETAINED'
            and recovered['returned_new_cells']==recovered['failed_or_unattempted_new_cells']==2
            and recovered['attempted_new_cells']==1 and recovered['PCG_calls']==1)
        check('foreign_partial_registration_is_rejected',raises(probe.v1.IntegrityError,
            lambda:probe.execute(registration,'other',folder=interrupted,guard_space=no_space)))
        damaged=base/'damaged';damaged.mkdir();s.write(damaged/'RUN_STARTED.json',{'registration_sha256':'damaged'})
        (damaged/'CELL_bad.json').write_text('{bad',encoding='utf8')
        damaged_result=probe.execute(registration,'damaged',folder=damaged,device='cpu-fixture',
            case_factory=forbidden_factory,guard_space=no_space)
        check('damaged_partial_JSON_retains_both_failed_cells_and_error',calls[0]==0
            and damaged_result['returned_new_cells']==damaged_result['failed_or_unattempted_new_cells']==2
            and len(damaged_result['errors'])==1)
        def fail_space(**kwargs): raise probe.v1.BudgetError('injected low disk')
        disk=probe.execute(registration,'disk',folder=base/'disk',device='cpu-fixture',
            case_factory=forbidden_factory,guard_space=fail_space)
        check('low_disk_before_setup_keeps_full_failure_denominator_without_GPU',calls[0]==0
            and disk['returned_new_cells']==disk['failed_or_unattempted_new_cells']==2 and disk['attempted_new_cells']==0)
        rejected=probe.launch_failure(probe.v1.IntegrityError('injected launch hash failure'),folder=base/'launch')
        check('launch_rejection_has_two_rows_without_overwriting_a_terminal',
            rejected['returned_new_cells']==rejected['failed_or_unattempted_new_cells']==2
            and rejected['attempted_new_cells']==0 and not (base/'launch/RESULT.json').exists())
        def exhausted_route(system,route,**kwargs):
            return {'route':route,'wall_s':240.,'converged':False,'pcg_started':True,
                'true_residual_norm':None,'true_residual_recomputed':False,'status':'TIMEOUT'},None
        with patch.object(probe.v1,'route_cell',side_effect=exhausted_route) as call:
            exhausted=probe.execute(registration,'exhausted',folder=base/'exhausted',device='cpu-fixture',
                case_factory=FakeCase,guard_space=no_space)
        check('total240second_route_budget_prevents_second_route_after_exhaustion',call.call_count==1
            and exhausted['rows'][1]['status']=='NOT_ATTEMPTED_TOTAL_BUDGET'
            and exhausted['PCG_calls']==1 and exhausted['returned_new_cells']==2)

        #Analytic bounds use a4-coordinate SPD system and its two true residuals.
        J=torch.tensor([[1.,2.,0.,1.],[0.,1.,1.,0.],[2.,0.,1.,1.]],dtype=torch.float64)
        D=torch.tensor([2.,3.,4.,5.],dtype=torch.float64)
        L=torch.tensor([[1.,-1.,0.,0.],[-1.,2.,-1.,0.],[0.,-1.,2.,-1.],[0.,0.,-1.,1.]],dtype=torch.float64)
        A=J.T@J+.2*L+probe.CONFIG['damping']*torch.diag(D)
        rhs=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
        xold=torch.tensor([.1,-.2,.3,.4],dtype=torch.float64);xnew=torch.tensor([.2,.1,-.3,.7],dtype=torch.float64)
        rms=lambda v:float(v.square().mean().sqrt())
        old={'converged':False,'true_residual_recomputed':True,'true_residual_norm':float((rhs-A@xold).norm()),
            'linear_outputs':{'raw_linear_RMS_e':rms(J@xold)}}
        new={'converged':False,'true_residual_recomputed':True,'true_residual_norm':float((rhs-A@xnew).norm()),
            'linear_outputs':{'raw_linear_RMS_e':rms(J@xnew)}}
        bound=probe.conditional_bounds(old,new,float(D.min()),same_system_verified=True)
        check('residual_and_spectral_lower_bound_controls_step_difference_on_same_dense_SPD_system',
            float((xold-xnew).norm())<=bound['step_L2_upper_bound']
            and torch.linalg.eigvalsh(A).min()>=bound['spectral_lower_bound']
            and bound['spectral_lower_bound']==probe.CONFIG['damping']*float(D.min())
            and bound['direct_solution_difference'] is None and not bound['both_stopping_tests_passed'],
            dense_dimension=4,actual_step_L2=float((xold-xnew).norm()),conditional_upper_bound=bound['step_L2_upper_bound'])
        check('same_fixed_linear_output_triangle_bound_has_correct_RMS_units',
            rms(J@(xold-xnew))<=bound['linear_output_RMS_upper_bounds']['raw_linear_RMS_e']
            and bound['linear_output_RMS_upper_bounds']['color_linear_RMS_K'] is None
            and not bound['bound_is_nonlinear_accuracy_or_uncertainty'])
        bad=copy.deepcopy(new);bad['true_residual_recomputed']=False
        unavailable=[probe.conditional_bounds(old,new,0.,same_system_verified=True),
            probe.conditional_bounds(old,new,float('nan'),same_system_verified=True),
            probe.conditional_bounds(old,new,2.,same_system_verified=False),
            probe.conditional_bounds(old,bad,2.,same_system_verified=True)]
        check('bounds_are_unavailable_without_positive_D_same_system_and_both_true_residuals',
            all(item['step_L2_upper_bound'] is None and item['direct_solution_difference'] is None for item in unavailable))

        native=object.__new__(probe.NativeCase);native.torch=torch;native.guard=None;native.p=object()
        native.state=native.beta=native.W=native.strength=None;native.D=torch.ones((1,3),dtype=torch.float64)
        native.mask=torch.tensor([[False,True,True]]);native.counts={'precondition':0}
        M=torch.tensor([[4.,1.,.3],[1.,3.,.2],[.3,.2,2.]],dtype=torch.float64)
        with patch.object(helper,'build',return_value=(lambda x:x@M,native.D,{'fixture_only':True})):
            pre,_=native.build_preconditioner(probe.ROUTE)
            outside=(~native.mask).double();a=torch.tensor([[1.,2.,3.]],dtype=torch.float64);b=torch.tensor([[3.,2.,1.]],dtype=torch.float64)
            check('native_v2_application_is_mask_P_mask_and_self_adjoint_on_free_subspace',
                torch.count_nonzero(pre(outside))==0 and torch.count_nonzero(pre(a)[~native.mask])==0
                and torch.allclose((a*pre(b)).sum(),(pre(a)*b).sum()) and float((a*pre(a)).sum())>0)
        with patch.object(helper,'build',return_value=(lambda x:x,2*native.D,{})):
            check('changed_original_damping_D_is_refused',raises(probe.v1.IntegrityError,
                lambda:native.build_preconditioner(probe.ROUTE)))
        check('native_v2_refuses_old_routes',raises(probe.v1.IntegrityError,
            lambda:native.build_preconditioner('old_block_diagonal')))

        #Metadata-only registration verifies the dependency closure and frozen
        #receipts using tiny stand-ins, without opening real raw/cache arrays.
        reg_folder=base/'registration';old=copy.deepcopy(upstream)
        standin=base/'standin.json';s.write(standin,{'fixture_only':True});ref=probe.v1.file_ref(standin)
        old['cases']=copy.deepcopy(registration['cases'])
        for case in old['cases']:
            case.update(raw_file=ref,state_file=ref,provenance={'receipt_journal':ref})
        old['selection']=ref
        own_contract=base/'own_contract.json';s.write(own_contract,{'all_passed':True,'tests':[{'passed':True}],
            'source_sha256':probe.source_hashes()})
        helper_contract=base/'helper_contract.json';s.write(helper_contract,{'all_passed':True,
            'batch_limit':64,'tests':[{'passed':True}],
            'source_sha256':{name:s.sha(s.ROOT/name) for name in probe.HELPER_SOURCES}})
        original_locks=probe.owner_locks
        with patch.object(probe,'REPORT_DIR',reg_folder),patch.object(probe,'HELPER_CONTRACT',helper_contract),\
             patch.object(probe,'space_guard',side_effect=no_space),\
             patch.object(probe,'reference_controls',return_value=(old,registration['reused_controls'],[ref])),\
             patch.object(probe,'owner_locks',side_effect=lambda digest:original_locks(digest,(base/'regglobal.json',base/'reglocal.json'))):
            registered=probe.register(own_contract);verified,digest=probe.verify_registration()
            duplicate=raises(FileExistsError,lambda:probe.register(own_contract))
        expected=set(upstream['source_sha256'])|set(probe.OWN_SOURCES+probe.HELPER_SOURCES+probe.RUNTIME_SOURCES)
        check('registration_freezes_exact_dependencies_runtime5_and_old_contracts_without_rerun',
            registered==verified and set(registered['source_sha256'])==expected and len(expected)==17
            and len(registered['contracts'])==3+len(old['contracts']) and digest==s.sha(reg_folder/'REGISTRATION.json')
            and duplicate and registered['GPU_started'] is False and registered['whole_inverse_calls']==0,
            dependency_count=len(expected))
        with patch.object(probe,'REPORT_DIR',reg_folder):
            original=standin.read_bytes();standin.write_bytes(original+b' ')
            stale=raises(probe.v1.IntegrityError,probe.verify_registration);standin.write_bytes(original)
        check('frozen_input_or_upstream_receipt_byte_change_is_rejected',stale)
        check('all_CPU_fixture_outputs_are_small_JSON_without_step_or_native_vector_caches',
            not list(base.rglob('*.npz')) and not list(base.rglob('*.npy')) and probe.v1.bytes_used(base)<2**20,
            fixture_bytes=probe.v1.bytes_used(base))

    frozen_after={name:s.sha(s.ROOT/name) for name in frozen_before}
    check('all11_old_dependencies_and_v1_system_control_receipts_remain_byte_identical',
        len(frozen_before)==11 and frozen_before==frozen_after and frozen_before==upstream['source_sha256']
        and receipt_before=={str(path):s.sha(path) for path in receipt_paths})
    check('native_exact_system_rhs_mask_linear_outputs_objective_and_constraints_inherited_unchanged',
        all(getattr(probe.NativeCase,name) is getattr(probe.v1.NativeCase,name)
            for name in ('__init__','normal','linear_outputs','check','sync'))
        and list(inspect.signature(probe.NativeCase).parameters)==['case','device'])
    check('fixed_mu_rtol_iterations_budgets_and_full_inverse_truth_zero',probe.CONFIG['damping']==1e-3
        and probe.CONFIG['rtol']==1e-10 and probe.CONFIG['maxiter']==2000
        and probe.CONFIG['route_wall_s']==120. and probe.CONFIG['total_route_wall_s']==240.
        and probe.CONFIG['planned_cells']==2 and probe.CONFIG['routes']==[probe.ROUTE]
        and probe.CONFIG['output_cap_bytes']==2**20 and not probe.CONFIG['truth_used']
        and probe.CONFIG['whole_inverse_calls']==0 and not probe.CONFIG['old_block_rerun'])
    result={'schema':probe.VERSION+'-CPU-contracts','time':s.now(),'all_passed':all(t['passed'] for t in tests),
        'tests':tests,'source_sha256':probe.source_hashes(),
        'helper_source_sha256':{name:s.sha(s.ROOT/name) for name in probe.HELPER_SOURCES},
        'runtime_source_sha256':runtime_before,'runtime_contract':probe.v1.file_ref(probe.RUNTIME_CONTRACT),
        'v1_source_sha256':frozen_after,'v1_receipt_sha256':receipt_before,
        'GPU_used':False,'native_GPU_qualified':False,'whole_inverse_calls':0,'truth_arrays_read':False,
        'real_development_raw_or_parameter_arrays_opened':False,'scientific_probe_started':False,
        'fixture_max_dense_dimension':4,'CPU_wall_s':time.perf_counter()-started}
    number=len(list(probe.REPORT_DIR.glob('CPU_CONTRACTS_ATTEMPT_*.json')))+1
    probe.write_json(probe.REPORT_DIR/f'CPU_CONTRACTS_ATTEMPT_{number:03d}.json',result)
    if result['all_passed']: s.write(probe.REPORT_DIR/'CPU_CONTRACTS.json',result,replace=(probe.REPORT_DIR/'CPU_CONTRACTS.json').exists())
    print(json.dumps({'all_passed':result['all_passed'],'tests':len(tests),'failures':[t for t in tests if not t['passed']],
        'source_sha256':result['source_sha256'],'CPU_wall_s':result['CPU_wall_s']},indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-helper-contracts',action='store_true');args=parser.parse_args()
    if args.include_helper_contracts:
        from . import coupled_preconditioner_v2 as contracts
        if not contracts.run()['all_passed']: raise SystemExit(2)
    raise SystemExit(0 if run()['all_passed'] else 2)


if __name__=='__main__': main()
