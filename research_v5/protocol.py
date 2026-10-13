"""Predeclare the finite v5 decision study after complete CPU implementation QA."""
import ast
import importlib
import os
import stat
import subprocess
from .common import *


REQUIRED_MODULES = ('__init__.py', 'common.py', 'planning.py', 'run.py', 'reports.py', 'protocol.py',
    'stage1.py', 'continuum_solver.py', 'signed_linear_solver.py', 'reconstruct.py', 'uncertainty.py',
    'identifiability.py', 'oracle.py', 'data.py', 'family_data.py', 'baselines.py', 'metrics.py',
    'statistics.py', 'development.py', 'candidate_review.py', 'signed_gain_denoiser.py', 'train.py', 'final.py',
    'deploy.py', 'tests_deploy.py', 'tests_adjoint.py', 'tests_data_oracle.py', 'tests_information.py')


def _source_name(name):
    path = Path(name)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return None
    text = str(name).replace('\\', '/')
    if '..' in Path(text).parts:
        return None
    if '/' not in text:
        return text if (ROOT/text).is_file() else 'research_v5/' + text
    return text


def implementation_preflight():
    """Inspect real passing small-check receipts and reject stale/missing source.

    Import/AST/dispatch checks alone are not solver/information numerical QA.
    Required current source must also appear in actual CPU receipt manifests.
    """
    configure_runtime()
    missing = [name for name in REQUIRED_MODULES if not (CODE / name).is_file()]
    assert not missing, 'Implement every phase before freeze; missing: ' + ', '.join(missing)
    sources = source_hashes()
    for name in REQUIRED_MODULES:
        tree = ast.parse((CODE / name).read_text(encoding='utf-8-sig'), filename=name)
        assert not any(isinstance(node, ast.Raise) and node.exc is not None
                       and ((isinstance(node.exc, ast.Name) and node.exc.id == 'NotImplementedError')
                            or (isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name)
                                and node.exc.func.id == 'NotImplementedError'))
                       for node in ast.walk(tree)), 'Placeholder phase is forbidden: ' + name
    api = {'stage1': ('run_oracle', 'run_identifiability', 'run_linear_solver'),
           'development': ('run_development',), 'candidate_review': ('review_candidate',),
           'train': ('train_denoiser',), 'final': ('run_final_once',),
           'reconstruct': ('reconstruct',), 'deploy': ('reliable_reconstruct', 'main'),
           'reports': ('write_scientific_decision', 'write_train_report')}
    imported = []
    for name, functions in api.items():
        module = importlib.import_module('research_v5.' + name)
        for function in functions:
            assert callable(getattr(module, function, None)), 'Missing connected API: ' + name + '.' + function
        imported.append({'module': name, 'callable_API': list(functions)})
    # These are pure small controls for missing evidence and nonconvergence.
    from .reports import contract_checks
    orchestration_checks = contract_checks()
    receipt_paths = set()
    for directory in (REPORTS.parent / 'implementation_checks', REPORTS.parent / 'core_tests'):
        if directory.exists():
            receipt_paths.update(directory.rglob('*.json'))
    receipt_paths.update(path for path in (REPORTS / 'IMPLEMENTATION_CHECKS.json',
                                          REPORTS / 'INTEGRATION_CHECKS.json') if path.exists())
    coverage = set()
    accepted = []
    rejected = []
    for path in sorted(receipt_paths):
        receipt = read_json(path)
        if not isinstance(receipt, dict) or not (receipt.get('all_passed') is True or receipt.get('success') is True):
            continue
        hashes = receipt.get('source_sha256', receipt.get('source_hashes', {}))
        if not isinstance(hashes, dict) or not hashes:
            continue
        normalized = {_source_name(name): digest for name, digest in hashes.items()}
        normalized.pop(None, None)
        stale = [name for name, digest in normalized.items() if not (ROOT / name).is_file()
                 or sha(ROOT / name) != digest]
        if stale:
            rejected.append({'path': relative(path), 'stale_sources': stale})
            continue
        # Formal scientific receipts cannot substitute for the pre-freeze CPU QA.
        device = str(receipt.get('device', receipt.get('execution', receipt.get('scope', '')))).lower()
        if 'cpu' not in device and receipt.get('fixture_only_not_scientific_acceptance') is not True:
            rejected.append({'path': relative(path), 'reason': 'CPU fixture scope not declared'})
            continue
        covered = {name for name in normalized if name in sources}
        coverage.update(covered)
        accepted.append({'path': relative(path), 'sha256': sha(path), 'source_coverage': sorted(covered),
                         'scientific_acceptance': False})
    # Native-resolution checks run on the real GPU and keep that device label.
    # Do not relabel their numerical evidence as a CPU fixture for registration.
    native_controls = []
    for path in sorted((REPORTS.parent/'core_tests').glob('native_256_*.json')):
        receipt = read_json(path)
        hashes = receipt.get('source_sha256', receipt.get('sources', {}))
        if (receipt.get('all_passed') is not True or receipt.get('device') != 'cuda'
                or receipt.get('shape') != [256,256] or receipt.get('scientific_cohort_acceptance') is not False
                or receipt.get('new_or_sealed_mother_arrays_generated') is not False
                or 'research_v5/tests_native.py' not in hashes):
            continue
        if any(not (ROOT/name).is_file() or sha(ROOT/name) != digest for name,digest in hashes.items()):
            continue
        native_controls.append({'path':relative(path),'sha256':sha(path),'device':'cuda',
                                'source_coverage':sorted(hashes),'all_passed':True,
                                'scientific_acceptance':False})
        coverage.update(name for name in hashes if name in sources)
    assert native_controls, 'A current-source actual 256x256 GPU operator/solver control is required'
    required = set(sources)
    uncovered = sorted(required - coverage)
    assert not uncovered, ('Current-source CPU checks are required before registration; uncovered: '
                           + ', '.join(uncovered) + '; stale receipts=' + str(len(rejected)))
    assert source_hashes() == sources, 'Source changed during pre-registration QA'
    return {'schema': 'v5-registration-source-CPU-QA-001', 'time': now(), 'all_passed': True,
            'all_phase_APIs_implemented': True, 'required_modules': list(REQUIRED_MODULES),
            'source_sha256': sources, 'CPU_test_receipts': accepted, 'rejected_stale_receipts': rejected,
            'native_resolution_GPU_controls': native_controls,
            'API_import_checks': imported, 'gate_contract_checks': orchestration_checks,
            'device': 'cpu', 'formal_arrays_generated': False, 'training_executed': False,
            'scientific_acceptance': False}


def register():
    configure_runtime()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()==BASE
    assert not (CODE/'protocol.json').exists(), 'A frozen protocol is never rewritten'
    qa = implementation_preflight()
    from .data import build_diagnostic_plan
    from .planning import build_family_plans
    from .continuum_solver import DEFAULT_OPTIONS as continuum_defaults
    from .signed_linear_solver import DEFAULT_OPTIONS as linear_defaults
    from .identifiability import FISHER_DEFAULTS, NULLSPACE_DEFAULTS
    from .final import frozen_evaluation_fields
    # Plans contain metadata only. All generation is after this registration.
    ladder=build_diagnostic_plan(ROOT/'reports/training_v4/run_001/data_manifest.json')
    families=build_family_plans()
    assert len(ladder) == 180 and len({case['mother_id'] for case in ladder}) == 12
    final_fields=frozen_evaluation_fields(families)
    assert len(final_fields['final_MC_case_ids']) == len(set(final_fields['final_MC_case_ids'])) == 12
    assert final_fields['case_counts']['signed_confirmation'] == len(families['plans']['signed_confirmation']) == 288
    assert final_fields['case_counts']['final_once'] == len(families['plans']['final_once']) == 512
    inventory=tracked_baseline_inventory()
    assert len(inventory)==1452, 'Baseline tracked-file count differs from the reviewed commit'
    for item in inventory:
        path=ROOT/item['path']
        item['was_readonly']=bool(getattr(path.stat(),'st_file_attributes',0)&stat.FILE_ATTRIBUTE_READONLY)
        os.chmod(path,stat.S_IREAD)
        assert getattr(path.stat(),'st_file_attributes',0)&stat.FILE_ATTRIBUTE_READONLY
    write_json(REPORTS/'baseline_readonly_inventory.json',{'baseline_head':BASE,'files':inventory,
        'old_source_weights_reports_readonly':True,'mutated_bytes':0,'old_checkpoint_payloads_loaded':False})
    # Frozen raw/label/cache arrays already exist. Attribute-protect them without
    # opening labels from any prospective final/confirmation partition.
    frozen_arrays=[]
    for version in ('training_v4','training_v4_1','training_v4_2','training_v4_3','training_v4_4','training_v4_5','training_v4_6'):
        folder=ROOT/'outputs'/version
        if folder.exists():
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix.lower()=='.npz' and '_runtime' not in p.parts:
                    before=p.stat()
                    os.chmod(p,stat.S_IREAD)
                    frozen_arrays.append({'path':relative(p),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,
                                          'readonly':True})
    write_json(REPORTS/'old_array_readonly_inventory.json',{'files':frozen_arrays,
        'array_payloads_opened':0,'sha256_scope':'old generation/manifest hashes retained; no unopened final payloads touched'})
    write_json(CODE/'diagnostic_plan.json',ladder)
    write_json(CODE/'family_plans.json',families)
    write_json(REPORTS/'SOURCE_REGISTRATION_CHECKS.json',qa)
    config=physics_config()
    protocol={'schema':'single-frame-spectral-thermometry-v5-decisive-study-v1','registered_at':now(),
        'baseline_head':BASE,'task_spec_sha256':sha(CODE/'TASK_SPEC.md'),
        'physics_config':config,'physics_config_sha256':canonical_hash(config),
        'calibration_source':'fixed externally supplied synthetic engineering configuration; no physical instrument calibration',
        'old_weights_sha256':{'old_B4':sha(ROOT/'checkpoints/selected_checkpoint.pt'),
            'v4.4_final_continuum_fixed':sha(ROOT/'training_v4_4/checkpoints/fixed_continuum/update_0500.pt')},
        'baselines':['B1_tikhonov','v3_physics','old_B4','v4.4_final_continuum_fixed'],
        'v44_internal_continuum_still_varies':True,'traditional_preselected_regularization':.1,
        'continuum_options':continuum_defaults,'linear_defaults':linear_defaults,
        'oracle_linear_options':{k:linear_defaults[k] for k in ('alpha_g','lambda0_factor','lambda_beta_factor','rtol','max_iterations')},
        'alpha_g_development_grid':[0.,.01,.1,1.],
        'diagnostic_plan_sha256':sha(CODE/'diagnostic_plan.json'),'diagnostic_cases':180,'diagnostic_mothers':12,
        'family_plans_sha256':sha(CODE/'family_plans.json'),'family_plan_hashes':families['plan_hashes'],
        'mother_partitions':{name:[m['mother_id'] for m in values] for name,values in families['mothers'].items()},
        'sealed_signed_confirmation_mothers':36,'sealed_final_mothers':64,
        'final_MC_case_ids':final_fields['final_MC_case_ids'],
        'source_sha256':source_hashes(),
        'source_CPU_QA_receipt_sha256':sha(REPORTS/'SOURCE_REGISTRATION_CHECKS.json'),
        'source_QA_passed_before_freeze':True,
        'fixed_input_sha256':{item['path']:item['sha256'] for item in inventory},
        'old_readonly_inventory_sha256':sha(REPORTS/'baseline_readonly_inventory.json'),
        'old_array_readonly_inventory_sha256':sha(REPORTS/'old_array_readonly_inventory.json'),
        'phase_budgets_s':{'register':300.,'oracle':14400.,'identifiability':3600.,'linear-solver':1200.,
            'development-compare':21600.,'train-denoiser':14400.,'candidate-review':2400.,'final-once':43200.},
        'hardware':'root-controlled local RTX4060 Laptop; one GPU job; earlier direct local-GPU authorization persists',
        'subagents_model':'gpt-6.1-sol','subagents_reasoning':'max',
        'CG':{'true_rtol':1e-4,'max_iterations':200,'not_a_recursive_residual_certificate':True},
        'oracle_noiseless_gate':{'input_kind':'continuous','coefficients_e':[180.,-180.],
            'pooled_RMSE_e_max':.5,'pooled_L2_max':.02,'support_sign_recall_min':.99,
            'all_CG_converged':True,'interpretation':'finite numerical recovery criterion for declared postblur synthetic normal signal'},
        'oracle_crosstalk_screen':{'pooled_MSE_factor_over_O0':1.25,'absolute_MSE_allowance_e2':1.,
            'collapsed_noiseless_normal_RMSE_e':5.,'not_a_universal_causal_proof':True},
        'oracle_baseline_subset':{'cases':84,'methods':4,'rows':336,'budget_s':7200.,
            'rule':'all 60 noise cases plus 24 continuous +/-180 e normal cases',
            'remaining_96_cases_excluded':True,'not_claimed_as_180_case_baseline_comparison':True},
        'fisher':{'snr_detectable':3.,'finite_modes_only':True,'not_full_image_exact_CRLB':True},
        'fisher_options':FISHER_DEFAULTS,'optical_options':NULLSPACE_DEFAULTS,
        'fisher_case_rule':'same oracle noise inputs: every one of 12 mothers at +180 e and -180 e; exactly 24 linearizations',
        'output_schemas':{'oracle':'v5-oracle-ladder-evidence-001','information':'v5-stage1-information-evidence-001',
            'stage1_decision':'v5-decision-1-001','phase_receipt':'v5-phase-execution-001'},
        'development_gates':{'nominal_color_RMSE_K_max':20.,'signed_normal_RMSE_factor':1.05,
            'valid_coverage_min':.99,'null_evaluation_frame_FP_max':.05,
            'MC_nominal_parent_mean_interval_coverage_min':.90,
            'weak_L2_not_required_below_one_when_undetectable':True},
        'selection_rule':'Among converged alpha grid points, minimize paired-parent mean normal +/- template MSE with color<=20K; choose one best traditional method by same development normal pooled MSE; freeze one configuration before null thresholding.',
        'null_detection':{'calibration_mothers':12,'evaluation_mothers':12,'repeats':8,
            'quantile':.95,'method':'higher quantile of independent calibration-frame maximum absolute standardized template; evaluated on disjoint null mothers',
            'threshold_scope':'simulated nominal calibration only; no instrument or distribution-free guarantee'},
        'MC':{'repeats':8,'seed':53125,'confidence_level':.95,'selected_development_cases':12,
            'rerun_continuum_each_repeat':True,'truth_used_only_for_external_coverage':True},
        'Hessian':{'probes':8,'seed':53126,'selected_cases':1,'packed_latent_coordinates_only':True},
        'bootstrap':{'unit':'mother_scene','draws':2000,'seed':53130,'confidence_level':.95},
        'training':{'model':'SignedBoundedGainDenoiser','width':8,'gain_bound':.3,'last_head_zero':True,
            'seed':53131,'optimizer':'Adam','lr':3e-4,'max_updates':1000,'checkpoints':[0,200,500,1000],
            'train_mothers':36,'geometry_factorial':[6,6],'reuse_old_bias_heads':False,
            'acceptance':{'normal_MSE_factor_vs_physics':1.05,'detectable_weak_MSE_factor':.95,
                'both_signs_required':True,'null_frame_FP_max':.05},
            'skip_if_no_above_limit_weak_subset':'pure physical plus uncertainty remains the candidate; learning cannot create missing information'},
        'final_once':{'requires_candidate_and_thresholds_and_statistics_frozen':True,
            'joint_signed_confirmation_and_final_evaluations':1,'retuning_permitted':False,**final_fields},
        'strict_stop_conditions':['O0 noiseless gate fails: stop ML, inspect solver/forward/optical information',
            'O0 passes but O1/O2 collapse: stop ML, quantify continuum/background identifiability',
            'O3 signed normal not comparable to selected traditional reference: stop ML until physical deployment works',
            'Single bounded neural control fails: terminate neural branch and keep pure physical results',
            'Final test fails: retain failure without retuning'],
        'raw_single_likelihood':True,'truth_in_deployment':False,'negative_concentration_claimed':False,
        'real_gas_temperature_or_10kHz_claimed':False,'final_or_confirmation_arrays_opened':False}
    # JSON canonicalization converts tuples to lists. Reopened immutable records
    # must equal the in-memory object rather than failing on Python tuple types.
    import json
    protocol=json.loads(json.dumps(protocol,ensure_ascii=False,allow_nan=False))
    assert protocol['source_sha256'] == qa['source_sha256'], 'Source drift before freeze'
    write_json(CODE/'protocol.json',protocol)
    write_json(REPORTS/'REGISTRATION_RECEIPT.json',{'status':'REGISTERED_NO_SCIENTIFIC_ARRAYS_GENERATED',
        'time':now(),'protocol_sha256':sha(CODE/'protocol.json'),'baseline_files_readonly':len(inventory),
        'old_arrays_readonly':len(frozen_arrays),'old_files_bytes_unchanged':True,
        'final_or_confirmation_payloads_opened':0,'GPU_calls':0})
    return protocol
