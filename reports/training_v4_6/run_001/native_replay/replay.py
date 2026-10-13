"""Pre-registered raw-route replay of the fixed v4.6 500-update endpoint.

--register and --self-test use only the standard library and JSON metadata.
--execute starts one root-controlled CUDA worker under a 180-second watchdog.
No ground-truth arrays or final-test inputs are opened by either replay branch.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
OUT = SCRIPT.parent
ROOT = SCRIPT.parents[4]
PROTOCOL = OUT / 'protocol.json'
EXECUTION = OUT / 'execution'
CHECKPOINT = ROOT / 'training_v4_6/checkpoints/negative_noisy_repair/update_0500.pt'
TRAIN = ROOT / 'reports/training_v4_6/run_001/negative_noisy_repair'
TRAIN_PROTOCOL = ROOT / 'reports/training_v4_6/run_001/execution_protocol.json'
MANIFEST = ROOT / 'reports/training_v4_1/run_001/data_manifest.json'
BACKBONE = ROOT / 'reports/training_v4/run_001/backbone_selection.json'
CALIBRATION = ROOT / 'reports/training_v4_3/run_001/residual_scaled/coefficient_calibration.json'
STRATA = (('negative_normal', 'noise'), ('weak', 'noise'), ('blank', 'quantization'))
CASE_IDS = ('v4_1_signed_validation_mother_0000_negative_normal_noise',
            'v4_validation_mother_0000_weak_noise0', 'v4_validation_mother_0000_blank_quantization')
FIELDS = ('state', 'beta', 'template_e', 'temperature_color_K', 'species_signed_e',
          'band_images_e', 'continuum_images_e', 'expected_e', 'raw_residual_e', 'valid',
          'correction_coefficient', 'initial_weighted_raw_RMS',
          'unrestricted_proposal_state', 'unrestricted_scaled_state')
SOURCES = ('training_v4_4/infer.py', 'training_v4_4/model.py',
           'training_v4_3/residual_scaling.py', 'training_v4/model.py',
           'training_v4/scalar_engine.py', 'repair_v3/solver.py', 'solver.py',
           'physics.py', 'config.py', 'runtime.py', 'evaluation_v2/common.py',
           'training_v4/common.py', 'training_v4_1/common.py',
           'training_v4/__init__.py', 'training_v4_1/__init__.py',
           'training_v4_3/__init__.py', 'training_v4_4/__init__.py',
           'evaluation_v2/__init__.py', 'repair_v3/__init__.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rel(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def rooted(relative):
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'Only repository-relative paths are allowed')
    path = (ROOT / path).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.drive.upper() in ('E:', 'D:'), 'Path escaped the E/D checkout')
    return path


def write(path, value, *, new=False):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT.resolve()) and path.drive.upper() in ('E:', 'D:'), 'Replay writes must stay in native_replay on E/D')
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if new:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(content)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            handle.write(content)
        temporary.replace(path)
    require(read(path) == value, 'JSON reopen verification failed: ' + str(path))


def expected_cases():
    """Read only the old manifest and score/cache JSON identity metadata."""
    rows = read(MANIFEST)['rows']
    require(len(rows) == 3352 and sum(row['split'] == 'validation' for row in rows) == 776, 'Old development manifest changed')
    result = []
    for condition, kind in STRATA:
        row = next(row for row in rows if row['split'] == 'validation'
                   and row['condition'] == condition and row['input_kind'] == kind)
        score_path = ROOT / 'outputs/training_v4_1/run_001/physical_scores' / (row['case_id'] + '.json')
        score = read(score_path)
        cache_path = rooted(score['cache_path'])
        metadata_path = cache_path.with_suffix('.json')
        metadata = read(metadata_path)
        require(score['raw_hash'] == metadata['raw_hash'] == row['raw_hash']
                and score['cache_key'] == metadata['cache_key']
                and score['cache_sha256'] == metadata['cache_sha256']
                and sha(metadata_path) == score['cache_metadata_sha256']
                and metadata['labels_used'] is False, 'Old raw/cache identity mismatch')
        # Neither truth arrays nor truth-derived score values enter this record.
        result.append({key: row[key] for key in ('case_id', 'mother_id', 'condition', 'input_kind',
                                                'raw_path', 'raw_file_sha256', 'raw_hash')} | {
            'split': 'validation', 'cache_path': rel(cache_path), 'cache_key': score['cache_key'],
            'cache_sha256': score['cache_sha256'], 'cache_metadata_path': rel(metadata_path),
            'cache_metadata_sha256': score['cache_metadata_sha256'],
            'score_identity_sidecar_path': rel(score_path), 'score_identity_sidecar_sha256': sha(score_path),
            'score_values_used': False})
    require(tuple(row['case_id'] for row in result) == CASE_IDS, 'Registered first-case order changed')
    return result


def register():
    require(ROOT.drive.upper() in ('E:', 'D:'), 'Replay registration must be on E/D')
    require(not EXECUTION.exists(), 'Cannot register after replay execution has started')
    if PROTOCOL.exists():
        return validate_protocol(sha(PROTOCOL))
    configuration = {key: read(BACKBONE)[key] for key in ('method', 'solver_config', 'physics_config')}
    value = {
        'schema': 'v4.6-fixed-endpoint-raw-native-replay-v1', 'registered_at': now(),
        'status': 'PRE_REGISTERED_NOT_EXECUTED', 'authorized_by': 'root',
        'endpoint': {'checkpoint': rel(CHECKPOINT), 'optimizer_updates': 500,
                     'checkpoint_sha256_at_registration': None,
                     'checkpoint_binding_rule': 'After COMPLETE500, bind the unique exact path in training_receipt.checkpoints at update 500 and its parameter SHA256. Never rank or substitute checkpoints.'},
        'training_status_path': rel(TRAIN / 'status.json'),
        'training_receipt_path': rel(TRAIN / 'training_receipt.json'),
        'training_execution_protocol_path': rel(TRAIN_PROTOCOL),
        'training_execution_protocol_sha256': sha(TRAIN_PROTOCOL),
        'training_endpoint_source_sha256': sha(ROOT / 'training_v4_6/train.py'),
        'maximum_total_wall_seconds': 180, 'cpu_threads': 2, 'cuda_workers': 1,
        'case_selection_rule': 'For the three listed strata, use the first validation row in the frozen old manifest order.',
        'cases': expected_cases(), 'case_count': 3,
        'old_manifest': rel(MANIFEST), 'old_manifest_sha256': sha(MANIFEST),
        'backbone': rel(BACKBONE), 'backbone_sha256': sha(BACKBONE),
        'backbone_configuration_sha256': canonical(configuration),
        'calibration': rel(CALIBRATION), 'calibration_sha256': sha(CALIBRATION),
        'replay_source_sha256': sha(SCRIPT),
        'dependency_files_sha256': {relative: sha(ROOT / relative) for relative in SOURCES},
        'raw_branch_inputs': ['raw_dn-only NPZ', 'explicit fixed checkpoint', 'frozen backbone configuration'],
        'cache_branch_inputs': ['the same raw_dn', 'historical state', 'historical beta', 'the same checkpoint'],
        'cache_arrays_opened_only_after_each_independent_raw_branch': True,
        'required_prediction_fields': list(FIELDS), 'prediction_key_sets_must_match': True,
        'physical_inverse_fields': ['state', 'beta'],
        'comparison_rule': 'Exact shape, dtype and finite element equality for both physical fields and every prediction field, including the boolean valid mask; maximum numeric absolute difference must be zero.',
        'numeric_absolute_tolerance': 0.0, 'numeric_relative_tolerance': 0.0,
        'valid_fraction_is_reported_not_a_candidate_acceptance_gate': True,
        'physical_inverse_and_cnn_times_measured_separately_with_cuda_synchronization': True,
        'timing_limit': 'Single-call stage timings exclude model load, I/O and transfer. The 180-second parent watchdog covers the whole execution; no hardware real-time acceptance claim.',
        'outputs_directory': rel(EXECUTION), 'failure_outputs_are_retained': True,
        'scope': 'Old DEV raw-route reproduction of one fixed endpoint only.',
        'candidate_acceptance': False, 'independent_confirmation': False,
        'ground_truth_arrays_opened': 0, 'labels_enter_inference': False,
        'optimizer_updates_in_this_replay': 0, 'final_test_used': False,
        'new_or_locked_development_used': False, 'checkpoint_selection_performed': False,
    }
    write(PROTOCOL, value, new=True)
    return validate_protocol(sha(PROTOCOL))


def validate_protocol(protocol_sha256):
    require(isinstance(protocol_sha256, str) and len(protocol_sha256) == 64 and sha(PROTOCOL) == protocol_sha256,
            'Supply the exact frozen native-replay protocol SHA256')
    value = read(PROTOCOL)
    require(value['schema'] == 'v4.6-fixed-endpoint-raw-native-replay-v1' and value['authorized_by'] == 'root', 'Unknown replay protocol')
    require(value['endpoint']['checkpoint'] == rel(CHECKPOINT) and value['endpoint']['optimizer_updates'] == 500,
            'Only the registered v4.6 update500 endpoint is allowed')
    require(value['maximum_total_wall_seconds'] == 180 and value['cpu_threads'] == 2 and value['cuda_workers'] == 1,
            'Resource budget changed')
    require(value['case_count'] == 3 and value['cases'] == expected_cases(), 'Frozen three-case selection changed')
    require(sha(MANIFEST) == value['old_manifest_sha256'] and sha(BACKBONE) == value['backbone_sha256']
            and sha(CALIBRATION) == value['calibration_sha256'], 'Frozen metadata/configuration changed')
    require(sha(TRAIN_PROTOCOL) == value['training_execution_protocol_sha256']
            and sha(ROOT / 'training_v4_6/train.py') == value['training_endpoint_source_sha256'],
            'Pre-registered training execution protocol/source changed')
    require(value['replay_source_sha256'] == sha(SCRIPT) and set(value['dependency_files_sha256']) == set(SOURCES),
            'Replay source or declared dependency set changed')
    for relative, expected in value['dependency_files_sha256'].items():
        require(sha(rooted(relative)) == expected, 'Frozen replay dependency changed: ' + relative)
    require(value['required_prediction_fields'] == list(FIELDS) and value['numeric_absolute_tolerance'] == 0.
            and value['numeric_relative_tolerance'] == 0. and value['prediction_key_sets_must_match'] is True,
            'Exact output comparison rule changed')
    require(not any(value[key] for key in ('candidate_acceptance', 'independent_confirmation', 'final_test_used',
                                         'new_or_locked_development_used', 'checkpoint_selection_performed'))
            and value['labels_enter_inference'] is False and value['ground_truth_arrays_opened'] == 0,
            'Replay scope changed')
    return value


def bind_endpoint(receipt):
    require(receipt['updates'] == 500 and receipt['microbatches'] == 2000, 'Training must complete exactly 500 updates')
    checkpoints = receipt['checkpoints']
    require(sorted(item['update'] for item in checkpoints) == [0, 100, 200, 300, 400, 500], 'Training checkpoint ledger differs')
    matches = [item for item in checkpoints if item['update'] == 500 and item['path'] == rel(CHECKPOINT)]
    require(len(matches) == 1, 'The explicit update500 checkpoint must occur exactly once')
    item = matches[0]
    require(item['Adam_steps'] == [500] and item['parameters_sha256'] == receipt['final_parameters_sha256'],
            'Training receipt does not bind the final Adam/CNN state')
    require(receipt['final_test_used'] is False and receipt['locked_development_generated'] is False,
            'Unexpected training receipt scope')
    return {key: item[key] for key in ('update', 'path', 'sha256', 'parameters_sha256', 'Adam_steps')}


def input_binding(protocol_sha256):
    protocol = validate_protocol(protocol_sha256)
    status_path, receipt_path = TRAIN / 'status.json', TRAIN / 'training_receipt.json'
    status, receipt = read(status_path), read(receipt_path)
    require(status['stage'] == 'COMPLETE' and status['optimizer_updates'] == 500, 'Root must finish v4.6 training before replay')
    endpoint = bind_endpoint(receipt)
    require(sha(CHECKPOINT) == endpoint['sha256'], 'Endpoint bytes do not match the completed training receipt')
    execution_hash = sha(TRAIN_PROTOCOL)
    require(execution_hash == receipt['protocol_sha256'] == protocol['training_execution_protocol_sha256'],
            'Training receipt execution protocol changed')
    require(sha(ROOT / 'training_v4_6/train.py') == receipt['source_sha256'] == protocol['training_endpoint_source_sha256'],
            'Completed trainer source changed')
    training_protocol = read(TRAIN_PROTOCOL)
    require(training_protocol['train_updates'] == 500 and not training_protocol['final_test_used']
            and not training_protocol['locked_development_generated'], 'Training execution scope changed')
    for relative, expected in protocol['dependency_files_sha256'].items():
        if relative in receipt['source_hashes_at_start']:
            require(receipt['source_hashes_at_start'][relative] == expected, 'Model dependency differs from the trained source')
    files = {rel(path): sha(path) for path in (PROTOCOL, SCRIPT, status_path, receipt_path, TRAIN_PROTOCOL,
                                               CHECKPOINT, MANIFEST, BACKBONE, CALIBRATION)}
    for case in protocol['cases']:
        for path_key, hash_key in (('raw_path', 'raw_file_sha256'), ('cache_path', 'cache_sha256'),
                                   ('cache_metadata_path', 'cache_metadata_sha256'),
                                   ('score_identity_sidecar_path', 'score_identity_sidecar_sha256')):
            actual = sha(rooted(case[path_key]))
            require(actual == case[hash_key], 'Frozen case file changed: ' + case[path_key])
            files[case[path_key]] = actual
    return protocol, {'protocol_sha256': protocol_sha256, 'endpoint': endpoint,
                      'training_receipt_sha256': sha(receipt_path), 'training_status_sha256': sha(status_path),
                      'training_execution_protocol_sha256': execution_hash,
                      'input_files_sha256': files, 'dependencies_sha256': protocol['dependency_files_sha256'],
                      'labels_enter_inference': False, 'ground_truth_arrays_opened': 0,
                      'candidate_acceptance': False, 'independent_confirmation': False, 'final_test_used': False}


def configure_worker():
    runtime_root = OUT / '_runtime'
    require(runtime_root.resolve().drive.upper() in ('E:', 'D:'), 'Runtime must stay on E/D')
    for child in ('tmp', 'torch', 'cache', 'mpl', 'cuda'):
        (runtime_root / child).mkdir(parents=True, exist_ok=True)
    for key in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[key] = str(runtime_root / 'tmp')
    os.environ['THERMOMETRY_RUNTIME'] = str(runtime_root)
    os.environ['CUDA_CACHE_PATH'] = str(runtime_root / 'cuda')
    os.environ['TORCH_HOME'] = str(runtime_root / 'torch')
    os.environ['XDG_CACHE_HOME'] = str(runtime_root / 'cache')
    os.environ['MPLCONFIGDIR'] = str(runtime_root / 'mpl')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    sys.path.insert(0, str(ROOT))


def worker(protocol_sha256, binding_sha256):
    started, records = time.monotonic(), []
    binding_path = EXECUTION / 'input_binding.json'
    require(sha(binding_path) == binding_sha256, 'Worker must use the exact parent-bound inputs')
    protocol, binding = input_binding(protocol_sha256)
    require(binding == read(binding_path), 'Parent-bound training/input identity changed before the worker')
    configure_worker()
    receipt = {'status': 'RUNNING', 'started_at': now(), **binding, 'records': records,
               'fixed_checkpoint_path': rel(CHECKPOINT), 'scope': protocol['scope'],
               'optimizer_updates_in_this_replay': 0, 'checkpoint_selection_performed': False,
               'worker_started_at_monotonic': started}
    write(EXECUTION / 'receipt.json', receipt)
    try:
        import numpy as np
        import torch
        from evaluation_v2.common import array_hash, tensor_digest
        from training_v4_4.infer import run as raw_infer, load_backbone, load_model, load_raw, save_arrays
        require(torch.cuda.is_available(), 'Root execution requires a CUDA worker')
        torch.set_num_threads(protocol['cpu_threads'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        configuration = load_backbone(BACKBONE)
        require(canonical(configuration) == protocol['backbone_configuration_sha256'], 'Frozen inference configuration changed')
        model, saved = load_model(CHECKPOINT, configuration, 'cuda')
        require(saved['optimizer_updates'] == 500 and saved['parameters_sha256'] == binding['endpoint']['parameters_sha256']
                and saved['execution_protocol_sha256'] == binding['training_execution_protocol_sha256']
                and saved['model_source_sha256'] == protocol['dependency_files_sha256']['training_v4_4/model.py']
                and saved['training_source_sha256'] == protocol['training_endpoint_source_sha256']
                and saved['coefficient_calibration'] == read(CALIBRATION),
                'Native checkpoint metadata differs from the completed training receipt')
        parameters_before = tensor_digest(model.named_parameters())
        buffers_before = tensor_digest(model.named_buffers())

        def arrays(path):
            with np.load(path, allow_pickle=False) as values:
                return {key: values[key].copy() for key in values.files}

        def compare(actual, expected):
            shape_equal, dtype_equal = actual.shape == expected.shape, actual.dtype == expected.dtype
            finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
            exact = shape_equal and dtype_equal and finite and bool(np.array_equal(actual, expected))
            difference = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if shape_equal and finite else None
            return {'shape_equal': shape_equal, 'dtype_equal': dtype_equal, 'all_finite': finite,
                    'exact': exact, 'maximum_absolute_difference': difference,
                    'actual_shape': list(actual.shape), 'expected_shape': list(expected.shape),
                    'actual_dtype': str(actual.dtype), 'expected_dtype': str(expected.dtype),
                    'actual_array_sha256': array_hash(actual), 'expected_array_sha256': array_hash(expected)}

        for case in protocol['cases']:
            case_folder = EXECUTION / 'cases' / case['case_id']
            case_folder.mkdir(parents=True, exist_ok=False)
            raw_folder = case_folder / 'raw_branch'
            print(json.dumps({'stage': 'independent_raw_branch', 'case_id': case['case_id']}), flush=True)
            # No case descriptor, cache, score, condition, label or truth enters
            # the deployable branch. Historical cache arrays remain unopened.
            raw_receipt = raw_infer(rooted(case['raw_path']), CHECKPOINT, raw_folder,
                                    backbone=BACKBONE, device='cuda')
            require(raw_receipt['status'] == 'SUCCESS' and raw_receipt['inverse_recomputed_from_raw'] is True
                    and raw_receipt['labels_loaded'] is False and raw_receipt['condition_loaded'] is False
                    and raw_receipt['development_cache_loaded'] is False, 'Raw branch violated its inference scope')
            require(raw_receipt['checkpoint_sha256'] == binding['endpoint']['sha256']
                    and raw_receipt['parameters_sha256'] == binding['endpoint']['parameters_sha256']
                    and raw_receipt['checkpoint_optimizer_updates'] == 500
                    and raw_receipt['raw_hash'] == case['raw_hash']
                    and raw_receipt['raw_file_sha256'] == case['raw_file_sha256'], 'Raw branch input/model identity differs')
            actual = arrays(raw_folder / 'prediction.npz')
            inverse = arrays(raw_folder / 'physical_inverse.npz')
            # Only the outer QA branch now opens the historical physical cache.
            cached = arrays(rooted(case['cache_path']))
            require(set(cached) == {'state', 'beta'} and set(inverse) == {'state', 'beta'}, 'Unexpected physical-cache fields')
            raw = load_raw(rooted(case['raw_path']), configuration)
            require(array_hash(raw) == case['raw_hash'], 'Outer QA raw differs from the independent branch')
            raw_tensor = torch.as_tensor(raw.astype(np.float32), device='cuda')[None]
            state = torch.as_tensor(cached['state'], device='cuda')[None]
            beta = torch.as_tensor(cached['beta'], device='cuda')[None]
            with torch.no_grad():
                torch.cuda.synchronize()
                reference_started = time.perf_counter()
                expected_tensors = model(raw_tensor, state, beta)
                torch.cuda.synchronize()
                reference_seconds = time.perf_counter() - reference_started
            expected = {key: value.detach().cpu().numpy() for key, value in expected_tensors.items()}
            require(set(actual) == set(expected) == set(FIELDS), 'Prediction field coverage differs from registration')
            physical = {'state': compare(inverse['state'], cached['state'][None]),
                        'beta': compare(inverse['beta'], cached['beta'][None])}
            output_checks = {key: compare(actual[key], expected[key]) for key in FIELDS}
            reference_path = case_folder / 'cached_reference_prediction.npz'
            reference_hash = save_arrays(reference_path, expected)
            valid_fraction = float(expected['valid'].astype(np.float64).mean())
            valid_fraction_matches = raw_receipt['valid_fraction'] == valid_fraction
            passed = valid_fraction_matches and all(value['exact'] for value in [*physical.values(), *output_checks.values()])
            record = {'case_id': case['case_id'], 'split': 'old_DEV', 'passed': passed,
                      'raw_hash': case['raw_hash'], 'raw_file_sha256': case['raw_file_sha256'],
                      'cache_key': case['cache_key'], 'cache_sha256': case['cache_sha256'],
                      'cache_metadata_sha256': case['cache_metadata_sha256'],
                      'checkpoint_sha256': binding['endpoint']['sha256'],
                      'parameters_sha256': binding['endpoint']['parameters_sha256'],
                      'physical_inverse_vs_cache': physical, 'all_prediction_fields': output_checks,
                      'all_14_prediction_fields_exact': all(value['exact'] for value in output_checks.values()),
                      'physical_state_and_beta_exact': all(value['exact'] for value in physical.values()),
                      'physical_inverse_seconds': raw_receipt['physical_inverse_seconds'],
                      'cnn_seconds': raw_receipt['cnn_seconds'], 'cache_reference_cnn_seconds': reference_seconds,
                      'raw_branch_valid_fraction': raw_receipt['valid_fraction'],
                      'cache_reference_valid_fraction': valid_fraction,
                      'valid_fraction_matches': valid_fraction_matches, 'valid_mask_is_candidate_acceptance_gate': False,
                      'raw_branch_receipt': rel(raw_folder / 'inference_receipt.json'),
                      'raw_branch_receipt_sha256': sha(raw_folder / 'inference_receipt.json'),
                      'raw_branch_prediction_sha256': sha(raw_folder / 'prediction.npz'),
                      'raw_branch_physical_inverse_sha256': sha(raw_folder / 'physical_inverse.npz'),
                      'cached_reference_prediction': rel(reference_path), 'cached_reference_prediction_sha256': reference_hash,
                      'labels_enter_inference': False, 'ground_truth_arrays_opened': 0,
                      'cache_arrays_opened_after_raw_branch': True, 'candidate_acceptance': False,
                      'independent_confirmation': False, 'final_test_used': False}
            write(case_folder / 'comparison.json', record, new=True)
            records.append(record)
            write(EXECUTION / 'progress.json', {'completed': len(records), 'total': 3, 'records': records,
                                               'elapsed_worker_seconds': time.monotonic() - started, 'final_test_used': False})
            print(json.dumps({'case_id': case['case_id'], 'exact_replay_passed': passed,
                              'physical_inverse_seconds': record['physical_inverse_seconds'],
                              'cnn_seconds': record['cnn_seconds']}), flush=True)
        require(tensor_digest(model.named_parameters()) == parameters_before == binding['endpoint']['parameters_sha256']
                and tensor_digest(model.named_buffers()) == buffers_before, 'Reference model changed during replay')
        _, after_binding = input_binding(protocol_sha256)
        require(after_binding == binding, 'A bound input, dependency or completed training receipt changed')
        require(len(records) == 3 and all(record['passed'] for record in records), 'At least one registered exact replay failed')
        receipt.update(status='SUCCESS', passed=True, completed_at=now(), records=records, case_count=3,
                       all_input_model_dependency_hashes_unchanged=True, CNN_and_physical_buffers_unchanged=True,
                       source_sha256=sha(SCRIPT), protocol_sha256=protocol_sha256,
                       source_runtime_versions={'torch': str(torch.__version__), 'numpy': np.__version__,
                                                'python': sys.version, 'cuda': torch.version.cuda,
                                                'GPU': torch.cuda.get_device_name()},
                       elapsed_worker_seconds=time.monotonic() - started,
                       claim_limit='Old DEV raw-route reproduction only; not candidate acceptance, independent confirmation or final test.')
        write(EXECUTION / 'receipt.json', receipt)
    except BaseException as exc:
        receipt.update(status='FAILED', passed=False, completed_at=now(), records=records,
                       error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                       elapsed_worker_seconds=time.monotonic() - started, failed_outputs_retained=True)
        write(EXECUTION / 'receipt.json', receipt)
        raise


def execute(protocol_sha256):
    started = time.monotonic()
    protocol, binding = input_binding(protocol_sha256)
    require(not EXECUTION.exists(), 'Replay was already attempted; retain it and do not overwrite or automatically rerun')
    EXECUTION.mkdir(parents=True, exist_ok=False)
    binding_path = EXECUTION / 'input_binding.json'
    write(binding_path, binding, new=True)
    deadline = started + protocol['maximum_total_wall_seconds']
    log_path = EXECUTION / 'worker.log'
    environment = dict(os.environ)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONUNBUFFERED'] = '1'
    args = [sys.executable, '-B', str(SCRIPT), '--worker', '--protocol-sha256', protocol_sha256,
            '--binding-sha256', sha(binding_path)]
    result = {'status': 'STARTING', 'started_at': now(), 'maximum_total_wall_seconds': 180,
              'protocol_sha256': protocol_sha256, 'input_binding_sha256': sha(binding_path),
              'worker_log': rel(log_path), 'subprocess_count': 1, 'final_test_used': False}
    write(EXECUTION / 'watchdog_receipt.json', result)
    process = None
    try:
        require(time.monotonic() < deadline, 'Metadata verification exhausted the replay time budget')
        with log_path.open('x', encoding='utf-8') as log:
            process = subprocess.Popen(args, cwd=str(ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            print(json.dumps({'worker_started': True, 'pid': process.pid, 'maximum_total_wall_seconds': 180}), flush=True)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('The registered 180-second total replay budget expired')
                try:
                    exit_code = process.wait(timeout=min(15., remaining))
                    break
                except subprocess.TimeoutExpired:
                    print(json.dumps({'worker_running': True, 'elapsed_total_seconds': time.monotonic() - started}), flush=True)
        require(exit_code == 0, 'Native replay worker failed; retain worker.log and partial receipts')
        receipt = read(EXECUTION / 'receipt.json')
        require(receipt['status'] == 'SUCCESS' and receipt['passed'] is True and receipt['case_count'] == 3,
                'Worker did not complete all three exact replays')
        require(time.monotonic() <= deadline, 'The registered replay budget was exceeded')
        result.update(status='SUCCESS', passed=True, exit_code=exit_code, completed_at=now(),
                      elapsed_total_seconds=time.monotonic() - started,
                      worker_receipt_sha256=sha(EXECUTION / 'receipt.json'),
                      worker_log_sha256=sha(log_path), watchdog_termination=False)
        write(EXECUTION / 'watchdog_receipt.json', result)
        print(json.dumps({'raw_native_replays': 3, 'passed': True,
                          'receipt': str(EXECUTION / 'receipt.json'),
                          'receipt_sha256': result['worker_receipt_sha256'],
                          'elapsed_total_seconds': result['elapsed_total_seconds'],
                          'candidate_acceptance': False, 'independent_confirmation': False, 'final_test_used': False}), flush=True)
        return result
    except BaseException as exc:
        terminated = process is not None and process.poll() is None
        if terminated:
            process.kill()
            process.wait(timeout=10.)
        result.update(status='FAILED', passed=False, completed_at=now(), error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc(), elapsed_total_seconds=time.monotonic() - started,
                      watchdog_termination=terminated, worker_exit_code=None if process is None else process.poll(),
                      partial_artifacts_retained=True)
        if log_path.exists():
            result['worker_log_sha256'] = sha(log_path)
        write(EXECUTION / 'watchdog_receipt.json', result)
        raise


def self_test():
    """Only parse code, inspect JSON metadata and exercise endpoint rejection."""
    for path in [SCRIPT, *[ROOT / relative for relative in SOURCES]]:
        ast.parse(path.read_text(encoding='utf-8-sig'))
    require(tuple(case['case_id'] for case in expected_cases()) == CASE_IDS, 'First-case selection self-test failed')
    fixture = {'updates': 500, 'microbatches': 2000, 'final_parameters_sha256': 'fixed',
               'final_test_used': False, 'locked_development_generated': False,
               'checkpoints': [{'update': update, 'path': rel(CHECKPOINT) if update == 500 else 'other',
                                'Adam_steps': [update], 'parameters_sha256': 'fixed', 'sha256': 'bytes'}
                               for update in (0, 100, 200, 300, 400, 500)]}
    require(bind_endpoint(fixture)['path'] == rel(CHECKPOINT), 'Explicit endpoint fixture failed')
    failures = []
    for mutation in ('different_path', 'wrong_Adam_step', 'wrong_parameter_digest', 'missing_checkpoint'):
        value = json.loads(json.dumps(fixture))
        if mutation == 'different_path':
            value['checkpoints'][-1]['path'] = 'some_selected_candidate.pt'
        elif mutation == 'wrong_Adam_step':
            value['checkpoints'][-1]['Adam_steps'] = [400]
        elif mutation == 'wrong_parameter_digest':
            value['final_parameters_sha256'] = 'different'
        else:
            value['checkpoints'].pop()
        try:
            bind_endpoint(value)
        except ValueError:
            failures.append(mutation)
        else:
            raise AssertionError('Invalid endpoint binding was accepted: ' + mutation)
    if PROTOCOL.exists():
        validate_protocol(sha(PROTOCOL))
    result = {'syntax_parsed': 1 + len(SOURCES), 'fixed_first_old_DEV_case_ids': list(CASE_IDS),
              'endpoint_binding_rejections': failures, 'only_JSON_metadata_read': True,
              'array_files_opened': 0, 'model_loads': 0, 'GPU_calls': 0,
              'torch_imported': 'torch' in sys.modules, 'numpy_imported': 'numpy' in sys.modules,
              'runtime_imported': 'runtime' in sys.modules, 'files_written': 0,
              'execution_authorized_by_self_test': False, 'final_test_used': False}
    require(not any(result[key] for key in ('torch_imported', 'numpy_imported', 'runtime_imported')),
            'Metadata self-test imported a model/array runtime')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--binding-sha256', help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.register:
        value = register()
        print(json.dumps({'registered': True, 'protocol': str(PROTOCOL), 'protocol_sha256': sha(PROTOCOL),
                          'source_sha256': sha(SCRIPT), 'case_ids': [case['case_id'] for case in value['cases']],
                          'model_loads': 0, 'GPU_calls': 0, 'final_test_used': False}))
    elif arguments.self_test:
        self_test()
    elif arguments.worker:
        worker(arguments.protocol_sha256, arguments.binding_sha256)
    else:
        execute(arguments.protocol_sha256)
