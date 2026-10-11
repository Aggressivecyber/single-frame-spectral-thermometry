"""Inference accepts measurements only; scoring remains outside this module."""
import time
import numpy as np
import torch
from evaluation_v2.common import MeasurementInput
from repair_v3.solver import (DEFAULTS, consistent_initialize, joint_data_step,
                              old_step_with_subblocks, delta_statistics, regularization)

METHODS = ('old_init__joint_update_v3', 'consistent_init__old_update_v3',
           'consistent_init__joint_update_v3')
SELECTED = METHODS[-1]


def prediction(p, state, beta):
    continuum, species, background, _ = p.components(state, beta)
    continuum_images = p.op.object_blur(continuum)
    temperature, valid = p.color_temperature(continuum_images)
    expected = p(state, beta)
    template = p.op.object_blur(state[:, 2:]*p.active[None, :, None, None]*p.Sscale)[:, 2]
    bands = p.op.object_blur(continuum+species+background)
    valid = valid & torch.isfinite(expected) & (expected >= 0) & (bands.amin(1) >= -1e-3)
    return {'temperature_K': temperature[0].cpu().numpy(), 'template_e': template[0].cpu().numpy(),
            'valid': valid[0].cpu().numpy(), 'latent_state': state.cpu().numpy(),
            'beta': beta.cpu().numpy(), 'expected_e': expected[0].cpu().numpy()}


class PhysicalReconstructor:
    def __init__(self, physics, method=SELECTED, config=None):
        if method not in METHODS:
            raise ValueError('Unknown registered v3 method')
        self.physics, self.method = physics, method
        self.config = dict(DEFAULTS if config is None else config)

    def reconstruct(self, inputs, *, trace=False):
        if type(inputs) is not MeasurementInput:
            raise TypeError('Inference takes MeasurementInput only, never truth or scene metadata')
        return self._infer(inputs.raw_dn, 'raw_DN', trace)

    def diagnose_continuous_electrons(self, electrons, *, trace=False):
        return self._infer(electrons, 'oracle_continuous_e', trace)

    def _infer(self, array, kind, trace):
        p, config = self.physics, self.config
        raw = torch.as_tensor(np.array(array, copy=True), device=p.M.device, dtype=p.M.dtype)[None]
        if tuple(raw.shape) != (1, *p.cfg['shape']) or not torch.isfinite(raw).all():
            raise ValueError('Finite H x W measurement required')
        if kind == 'raw_DN':
            if raw.min() < 0 or raw.max() > 4095:
                raise ValueError('Measurement outside ADC bounds')
            if torch.any(raw >= 4095):
                return {'status': 'SATURATION_REJECTED', 'metrics': None}
            raw = p.raw_to_electrons(raw)
        if p.M.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        snapshots, curve, warm = [], [], []
        weights = (raw.clamp_min(0)+p.extra_variance).reciprocal()
        strength = raw.new_tensor(config['regularization'])
        def snapshot(state, beta, label, info=None):
            pred = prediction(p, state, beta)
            raw_cost = .5*((torch.as_tensor(pred['expected_e'], device=raw.device)[None]-raw).square()*weights).sum()
            reg, _ = regularization(state, strength)
            snapshots.append({'label': label, 'prediction': pred, 'solver': info,
                              'raw_objective': float(raw_cost), 'regularization': float(reg),
                              'total_objective': float(raw_cost+reg),
                              'raw_RMSE_e': float((torch.as_tensor(pred['expected_e'], device=raw.device)[None]-raw).square().mean().sqrt())})
        with torch.no_grad():
            state, beta = p.initialize(raw)
            if trace:
                snapshot(state, beta, 'old_initializer_seed')
            if self.method.startswith('consistent_init'):
                state, beta, warm = consistent_initialize(p, raw, config, deadline=start+config['per_frame_wall_s'])
            if trace:
                snapshot(state, beta, 'initialization')
            damping = config['LM_initial']
            for iteration in range(config['outer_iterations']):
                if time.perf_counter()-start > config['per_frame_wall_s']:
                    raise TimeoutError('Physical v3 frame wall budget')
                before, before_beta = state, beta
                if '__joint_update' in self.method:
                    state, beta, info = joint_data_step(p, raw, state, beta, config, damping)
                    damping = info['next_damping']
                    if trace and iteration == 0:
                        # Component decomposition is a scoring-only preview of
                        # the atomic accepted joint step, not a second solver.
                        spectral = state.clone()
                        spectral[:, 0] = before[:, 0]
                        snapshot(spectral, beta, 'joint_spectral_components_preview_only',
                                 {'feeds_solver': False, **delta_statistics(p, before, spectral, before_beta, beta)})
                        snapshot(state, beta, 'joint_temperature_components_added_atomic_step', info)
                    curve.append({'iteration': iteration+1, **info})
                else:
                    state, beta, blocks = old_step_with_subblocks(p, raw, state, beta)
                    if trace and iteration == 0:
                        for zs, bs, label, detail in blocks:
                            snapshot(zs, bs, label, detail)
                    curve.append({'iteration': iteration+1, 'old_blocks': [block[3] for block in blocks]})
                if trace:
                    snapshot(state, beta, f'iteration_{iteration+1}', curve[-1])
                if p.M.device.type == 'cuda' and torch.cuda.max_memory_allocated()/2**20 > config['max_GPU_allocated_MiB']:
                    raise MemoryError('v3 GPU allocation budget')
                if '__joint_update' in self.method and not info['accepted']:
                    break
            pred = prediction(p, state, beta)
            if not all(np.isfinite(v).all() for v in pred.values()):
                raise ValueError('Nonfinite v3 output')
            if p.M.device.type == 'cuda':
                torch.cuda.synchronize()
            return {'status': 'SUCCESS', 'prediction': pred, 'trace': snapshots,
                    'warm_start': warm, 'curve': curve, 'iterations': len(curve),
                    'seconds': time.perf_counter()-start, 'input_kind': kind,
                    'peak_GPU_allocated_MiB': torch.cuda.max_memory_allocated()/2**20 if p.M.device.type == 'cuda' else 0,
                    'stop_reason': 'STEP_REJECTED_NO_STATIONARITY_CERTIFICATE' if '__joint_update' in self.method and not curve[-1]['accepted'] else 'REGISTERED_ITERATION_BUDGET',
                    'optimizer_updates': 0, 'CNN_used': False, 'original_CNN_preserved': True}
