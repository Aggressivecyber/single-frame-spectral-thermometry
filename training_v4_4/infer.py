"""Raw-only online physical inversion followed by FixedContinuumCNN.

An explicit checkpoint is required.  This entry point uses the frozen v4
backbone, does not select a candidate, and never loads labels or development
state caches.  It saves the full configuration, inverse history and separate
synchronized inversion/CNN timings.  These one-call timings are not a real-time
throughput measurement.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def storage_path(path):
    """Reject C, network paths and junctions resolving outside E/D before writes."""
    resolved = Path(path).expanduser().resolve()
    if resolved.drive.upper() not in ('E:', 'D:'):
        raise ValueError(f'Output and runtime paths must resolve onto E or D: {resolved}')
    return resolved


# Validate before importing runtime/torch: runtime creates its directories on
# import, and the CUDA driver can initialize a disk cache at first device use.
_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME = storage_path(os.environ.get('THERMOMETRY_RUNTIME', _ROOT / '_runtime'))
for _child in ('tmp', 'torch', 'cache', 'mpl', 'cuda'):
    storage_path(_RUNTIME / _child)
os.environ['THERMOMETRY_RUNTIME'] = str(_RUNTIME)
os.environ['CUDA_CACHE_PATH'] = str(_RUNTIME / 'cuda')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import runtime
import numpy as np
import torch

from evaluation_v2.common import array_hash, canonical_hash, tensor_digest
from training_v4.model import physical_output
from training_v4.scalar_engine import invert_single
from training_v4_4.model import FixedContinuumCNN


DEFAULT_BACKBONE = runtime.ROOT / 'reports/training_v4/run_001/backbone_selection.json'
# Hash only the consumed configuration: a byte-for-byte copy or a compact
# inference-only JSON in a delivery bundle describes the same frozen solver.
FROZEN_BACKBONE_CONFIG_SHA256 = 'e8a00514ef2b761d1dbd531adbda957d09d4f2883382574dc6a4f75ff35bf667'
SOURCE_PATHS = (
    'training_v4_4/infer.py', 'training_v4_4/model.py',
    'training_v4_3/residual_scaling.py', 'training_v4/model.py',
    'training_v4/scalar_engine.py', 'repair_v3/solver.py', 'solver.py',
    'physics.py', 'config.py', 'runtime.py', 'evaluation_v2/common.py',
)


def load_backbone(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    configuration = {key: value[key] for key in ('method', 'solver_config', 'physics_config')}
    if canonical_hash(configuration) != FROZEN_BACKBONE_CONFIG_SHA256:
        raise ValueError('Backbone physics/method/solver differ from the frozen training configuration')
    return configuration


def load_model(checkpoint, configuration, device='cpu'):
    """Load complete CNN weights; require the exact trained physics and digest."""
    saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    required = {'cnn_state_dict', 'physics_config', 'coefficient_calibration', 'parameters_sha256'}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError('Checkpoint must contain CNN weights, physics, calibration and parameter SHA256')
    if canonical_hash(saved['physics_config']) != canonical_hash(configuration['physics_config']):
        raise ValueError('Checkpoint physics differ from the frozen backbone')
    calibration = saved['coefficient_calibration']
    tau, power = calibration['tau'], calibration['power']
    if (not isinstance(power, int) or isinstance(power, bool) or power < 2
            or not math.isfinite(float(tau)) or float(tau) <= 0):
        raise ValueError('Checkpoint residual coefficient is not a valid frozen calibration')
    model = FixedContinuumCNN(configuration['physics_config'], calibration)
    own, supplied = model.cnn_state_dict(), saved['cnn_state_dict']
    if not isinstance(supplied, dict) or set(supplied) != set(own):
        raise ValueError('Checkpoint must provide every CNN tensor and no extra CNN keys')
    for name, value in supplied.items():
        if (not isinstance(value, torch.Tensor) or value.shape != own[name].shape
                or value.dtype != own[name].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Invalid CNN tensor: {name}')
    model.load_cnn(supplied)
    if tensor_digest(model.named_parameters()) != saved['parameters_sha256']:
        raise ValueError('Loaded CNN parameters do not match checkpoint parameters_sha256')
    return model.to(device).eval(), saved


def load_raw(path, configuration):
    """A single H x W raw_dn-only NPZ; reject mixed raw/label containers."""
    path = Path(path)
    if path.suffix.lower() != '.npz':
        raise ValueError('Input must be an NPZ containing raw_dn only')
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'raw_dn'}:
            raise ValueError('Input NPZ must contain exactly raw_dn; other fields are not read')
        raw = arrays['raw_dn'].copy()
    cfg = configuration['physics_config']
    if raw.ndim != 2 or list(raw.shape) != list(cfg['shape']):
        raise ValueError(f'Expected a single raw_dn frame with shape {cfg["shape"]}')
    if (not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw)
            or not np.isfinite(raw).all()):
        raise ValueError('raw_dn must be a finite real numeric array')
    maximum = 2 ** cfg['adc_bits'] - 1
    if np.any(raw < 0) or np.any(raw > maximum):
        raise ValueError('raw_dn is outside the configured ADC range')
    if np.any(raw >= maximum):
        raise ValueError('The frozen inverse does not support a saturated raw frame')
    return raw


def source_hashes():
    return {name: runtime.sha256(runtime.ROOT / name) for name in SOURCE_PATHS}


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(torch.device(device))


def save_arrays(path, arrays):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    with np.load(path, allow_pickle=False) as reopened:
        if set(reopened.files) != set(arrays) or any(
                not np.array_equal(reopened[name], value) for name, value in arrays.items()):
            raise RuntimeError(f'Array reopen verification failed: {path}')
    return runtime.sha256(path)


def run(raw_path, checkpoint, output, *, backbone=DEFAULT_BACKBONE, device='cpu'):
    """No truth, condition, cached-state or candidate-selection input exists."""
    destination = storage_path(output)
    if destination.exists():
        raise FileExistsError(f'Refusing to replace an inference directory: {destination}')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    raw_path, checkpoint, backbone = map(lambda value: Path(value).resolve(),
                                         (raw_path, checkpoint, backbone))
    configuration = load_backbone(backbone)
    raw_array = load_raw(raw_path, configuration)
    fingerprints = {name: runtime.sha256(path) for name, path in (
        ('raw_file_sha256', raw_path), ('checkpoint_sha256', checkpoint),
        ('backbone_file_sha256', backbone))}
    sources = source_hashes()
    torch.set_num_threads(2)
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model, saved = load_model(checkpoint, configuration, device)
    if runtime.sha256(checkpoint) != fingerprints['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed while it was being loaded')
    raw = torch.as_tensor(np.array(raw_array, dtype=np.float32, copy=True), device=device)[None]
    raw_e = model.physics.raw_to_electrons(raw)
    receipt = {
        'schema': 'fixed-continuum-raw-only-inference-v4.4', 'status': 'RUNNING',
        'started_at': runtime.now(), 'model': 'FixedContinuumCNN',
        'parameters': sum(value.numel() for value in model.parameters()),
        'parameters_sha256': tensor_digest(model.named_parameters()),
        'checkpoint': str(checkpoint), 'checkpoint_model_name': saved.get('model_name'),
        'checkpoint_schema': saved.get('schema'),
        'checkpoint_optimizer_updates': saved.get('optimizer_updates'),
        'checkpoint_declared_model_source_sha256': saved.get('model_source_sha256'),
        'raw_path': str(raw_path), 'raw_hash': array_hash(raw_array),
        'raw_dtype': str(raw_array.dtype), 'raw_shape': list(raw_array.shape),
        'raw_float32_hash': array_hash(raw[0].cpu().numpy()),
        'backbone': str(backbone), **fingerprints,
        'backbone_configuration_sha256': canonical_hash(configuration),
        'physics_config_sha256': canonical_hash(configuration['physics_config']),
        'solver_config_sha256': canonical_hash(configuration['solver_config']),
        'solver_engine_sha256': sources['training_v4/scalar_engine.py'],
        'source_hashes': sources,
        'configuration': {**configuration, 'coefficient_calibration': saved['coefficient_calibration']},
        'coefficient_calibration_sha256': canonical_hash(saved['coefficient_calibration']),
        'device': device, 'dtype': 'float32', 'torch_version': str(torch.__version__),
        'numpy_version': np.__version__, 'python_version': sys.version,
        'cuda_runtime_version': torch.version.cuda if device == 'cuda' else None,
        'device_name': torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
        'labels_loaded': False, 'condition_loaded': False, 'development_cache_loaded': False,
        'checkpoint_selected_by_entrypoint': False, 'optimizer_updates_in_this_call': 0,
        'inverse_recomputed_from_raw': True, 'batched_inverse': False,
        'inverse_then_post_CNN_not_differentiably_unrolled': True,
        'internal_continuum_proposal_still_active': True,
        'reproducibility_scope': (
            'Online inversion follows the same float32 scalar algorithm and frozen configuration. '
            'CPU/CUDA, library and floating-point differences can change PCG stopping and LM '
            'acceptance; equality to a historical cache needs a separate paired replay.'),
        'solver_budget_metadata_enforced_by_scalar_engine': False,
        'timing_scope': (
            'One synchronized call per stage; physical inverse and CNN measured separately. '
            'Model loading, host/device input transfer, diagnostics and file writes are excluded. '
            'This is not a warmed throughput or hardware real-time acceptance measurement.'),
    }
    destination.mkdir(parents=True, exist_ok=False)
    runtime.write_json(destination / 'inference_receipt.json', receipt)
    try:
        receipt['saved_input_sha256'] = save_arrays(destination / 'input_raw.npz', {'raw_dn': raw_array})
        print(json.dumps({'stage': 'physical_inverse', 'device': device}), flush=True)
        with torch.no_grad():
            synchronize(device)
            started = time.perf_counter()
            state, beta, history, engine_seconds = invert_single(
                model.physics, raw_e, configuration['method'], configuration['solver_config'])
            synchronize(device)
            inversion_seconds = time.perf_counter() - started
            print(json.dumps({'stage': 'cnn', 'physical_inverse_seconds': inversion_seconds}), flush=True)
            synchronize(device)
            started = time.perf_counter()
            predicted = model.forward_electrons(raw_e, state, beta)
            synchronize(device)
            cnn_seconds = time.perf_counter() - started
            baseline = physical_output(model.physics, state, beta, raw_e)
        if not torch.equal(predicted['state'][:, :2], state[:, :2]):
            raise RuntimeError('Final continuum state differs from the online physical inverse')
        for name in ('temperature_color_K', 'continuum_images_e'):
            if not torch.equal(predicted[name], baseline[name]):
                raise RuntimeError(f'Final {name} differs from the online physical inverse')
        arrays = {name: value.detach().cpu().numpy() for name, value in predicted.items()}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError('Nonfinite final prediction or proposal diagnostic')
        inverse_arrays = {'state': state.cpu().numpy(), 'beta': beta.cpu().numpy()}
        receipt.update(
            physical_inverse_seconds=inversion_seconds, cnn_seconds=cnn_seconds,
            inverse_plus_cnn_seconds=inversion_seconds + cnn_seconds,
            scalar_engine_reported_unsynchronized_seconds=engine_seconds,
            inverse_history=history, final_continuum_and_color_equal_physics=True,
            valid_fraction=float(predicted['valid'].double().mean()),
            prediction_sha256=save_arrays(destination / 'prediction.npz', arrays),
            physical_inverse_sha256=save_arrays(destination / 'physical_inverse.npz', inverse_arrays),
            prediction_array_hashes={name: array_hash(value) for name, value in arrays.items()},
            physical_inverse_array_hashes={name: array_hash(value) for name, value in inverse_arrays.items()},
            output_arrays_reopened_exactly=True,
        )
        after = {name: runtime.sha256(path) for name, path in (
            ('raw_file_sha256', raw_path), ('checkpoint_sha256', checkpoint),
            ('backbone_file_sha256', backbone))}
        if after != fingerprints or source_hashes() != sources:
            raise RuntimeError('An input, checkpoint, configuration or source changed during inference')
        if tensor_digest(model.named_parameters()) != saved['parameters_sha256']:
            raise RuntimeError('CNN parameters changed during inference')
        receipt.update(status='SUCCESS', completed_at=runtime.now(),
                       input_checkpoint_configuration_sources_unchanged=True)
    except BaseException as exc:
        receipt.update(status='FAILED', completed_at=runtime.now(),
                       error_type=type(exc).__name__, error=str(exc))
        runtime.write_json(destination / 'inference_receipt.json', receipt)
        raise
    runtime.write_json(destination / 'inference_receipt.json', receipt)
    reopened = json.loads((destination / 'inference_receipt.json').read_text(encoding='utf-8'))
    if reopened != receipt:
        raise RuntimeError('JSON receipt reopen verification failed')
    print(json.dumps({'status': receipt['status'], 'output': str(destination),
                      'physical_inverse_seconds': receipt['physical_inverse_seconds'],
                      'cnn_seconds': receipt['cnn_seconds'],
                      'valid_fraction': receipt['valid_fraction'],
                      'receipt_sha256': runtime.sha256(destination / 'inference_receipt.json')}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True, help='NPZ containing a single H x W raw_dn array only')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Explicit v4.3/v4.4 CNN checkpoint; never auto-selected')
    parser.add_argument('--backbone', type=Path, default=DEFAULT_BACKBONE, help='Frozen backbone configuration or its bundle copy')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True, help='New E/D output directory; existing paths are rejected')
    args = parser.parse_args()
    run(args.raw, args.checkpoint, args.output, backbone=args.backbone, device=args.device)


if __name__ == '__main__':
    main()
