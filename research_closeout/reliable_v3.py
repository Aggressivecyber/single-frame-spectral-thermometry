"""Bounded refinement of the unchanged v3 physical objective.

The historical twelve-step estimator and its outputs remain immutable. This
explicitly versioned numerical route uses analytic gradients, avoiding truncated
normal-equation CG directions. Convergence is tested on observable outputs of
successively tighter solves, not on a solver success flag alone.
"""
from . import storage as s
from . import measurement as m
from .baselines import from_state
import copy
import time
import numpy as np
import torch
from scipy.optimize import minimize
from repair_v3.solver import regularization

VERSION = 'v3_objective_refined_v2'
OPTIONS = {
    'cost_limit_s': 180., 'numeric_color_budget_K': .2,
    'numeric_template_budget_e': .17, 'minimum_stable_comparisons': 2,
    'stages': [
        {'gtol': 1e-4, 'ftol': 1e-11, 'maxiter': 400, 'maxfun': 850},
        {'gtol': 1e-6, 'ftol': 1e-13, 'maxiter': 600, 'maxfun': 1250},
        {'gtol': 1e-8, 'ftol': 5e-15, 'maxiter': 800, 'maxfun': 1650},
        {'gtol': 1e-10, 'ftol': 1e-15, 'maxiter': 1000, 'maxfun': 2050},
    ],
}


def validated_options(options=None):
    opt = copy.deepcopy(OPTIONS)
    if set(options or {})-set(OPTIONS):
        raise ValueError('Unknown v3 numerical option')
    opt.update(copy.deepcopy(options or {}))
    if not np.isfinite(opt['cost_limit_s']) or not 0 < opt['cost_limit_s'] <= 180:
        raise ValueError('The total v3 refinement budget cannot exceed 180 seconds')
    if opt['minimum_stable_comparisons'] != 2 or not 3 <= len(opt['stages']) <= 4:
        raise ValueError('Three or four endpoints and two tighter comparisons are required')
    for name, maximum in [('numeric_color_budget_K', .2), ('numeric_template_budget_e', .17)]:
        if not np.isfinite(opt[name]) or not 0 < opt[name] <= maximum:
            raise ValueError('The registered output budget cannot be relaxed')
    previous = None
    for stage in opt['stages']:
        if set(stage) != {'gtol', 'ftol', 'maxiter', 'maxfun'}:
            raise ValueError('Incomplete numerical stage')
        if any(not isinstance(stage[k], int) or stage[k] < 1 for k in ('maxiter', 'maxfun')):
            raise ValueError('Stage iteration limits must be positive integers')
        if any(not np.isfinite(stage[k]) or stage[k] <= 0 for k in ('gtol', 'ftol')):
            raise ValueError('Stage tolerances must be positive and finite')
        if previous is not None and any(stage[k] >= previous[k] for k in ('gtol', 'ftol')):
            raise ValueError('Every numerical stage must strictly tighten both tolerances')
        previous = stage
    return opt


class V3Objective:
    """Original weighted residual plus original latent nearest-neighbor prior."""
    def __init__(self, p, raw_e, solver_config=None, deadline=None):
        if p.M.dtype != torch.float64:
            raise ValueError('v3 numerical refinement requires float64')
        self.p = p
        self.raw_e = raw_e
        self.W = m.fixed_weights(p, raw_e)
        self.cfg = copy.deepcopy(solver_config or m.selection()['solver_config'])
        self.strength = p.M.new_tensor(self.cfg['regularization'])
        self.indices = [0, 1] + [2+i for i, v in enumerate(p.active) if bool(v)]
        self.shape = (1, 5, *p.cfg['shape'])
        self.npix = int(np.prod(p.cfg['shape']))
        self.deadline = deadline
        self.evaluations = 0
        self.last = None
        self.history = []
        # Coordinate scaling is fixed from calibration and raw weights, never
        # from an initializer or truth. It does not alter the physical prior.
        anchor = p.M.new_zeros(self.shape)
        anchor[:, 0] = (p.cfg['reference_temperature_K']-p.T0)/p.Tscale
        anchor[:, 1] = np.log(.5)
        d, db = p.preconditioner_diagonal(anchor, p.M.new_zeros((1, 3)), self.W)
        block = d[:, self.indices].mean((-2, -1)).clamp_min(1e-10)
        scale = block.rsqrt().clamp(1e-6, 1e3)[..., None, None].expand(1, len(self.indices), *p.cfg['shape'])
        self.scale = torch.cat([scale.flatten(), db.rsqrt().clamp(1e-6, 1e3).flatten()]).cpu().numpy()
        limits = p.cfg['temperature_range_K']
        lows = [(limits[0]-p.T0)/p.Tscale, -4.] + [-20.]*(len(self.indices)-2)
        highs = [(limits[1]-p.T0)/p.Tscale, .3] + [20.]*(len(self.indices)-2)
        self.lower = np.r_[np.repeat(lows, self.npix), [-20.]*3]/self.scale
        self.upper = np.r_[np.repeat(highs, self.npix), [20.]*3]/self.scale

    def pack(self, state, beta):
        return torch.cat([state[:, self.indices].flatten(), beta.flatten()]).detach().cpu().numpy()

    def unpack(self, flat):
        value = torch.as_tensor(flat, device=self.p.M.device, dtype=self.p.M.dtype)
        state = self.p.M.new_zeros(self.shape)
        state[:, self.indices] = value[:-3].reshape(1, len(self.indices), *self.p.cfg['shape'])
        return state, value[-3:][None]

    @torch.no_grad()
    def evaluate(self, flat):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TimeoutError('Bounded v3 objective refinement reached its wall budget')
        state, beta = self.unpack(flat)
        prediction = self.p(state, beta)
        residual = prediction-self.raw_e
        penalty, reg_gradient = regularization(state, self.strength)
        value = .5*(self.W*residual.square()).sum()+penalty
        gradient, beta_gradient = self.p.vjp(state, beta, self.W*residual)
        gradient = self.pack(gradient+reg_gradient, beta_gradient)
        if not np.isfinite(float(value)) or not np.isfinite(gradient).all():
            raise ArithmeticError('Nonfinite v3 physical objective or gradient')
        self.evaluations += 1
        self.last = (float(value), gradient, state, beta)
        self.history.append({'evaluation': self.evaluations, 'objective': float(value),
                             'gradient_l2': float(np.linalg.norm(gradient))})
        return self.last

    def fun(self, scaled):
        value, gradient, _, _ = self.evaluate(scaled*self.scale)
        return value, gradient*self.scale


def output_difference(previous, current):
    return {
        'color_RMS_K': float((previous['temperature_color_K']-current['temperature_color_K']).square().mean().sqrt()),
        'template_RMS_e': float((previous['template_postblur_e_ref']-current['template_postblur_e_ref']).square().mean().sqrt()),
    }


def plateau_accepts(comparisons, opt=OPTIONS):
    needed = int(opt['minimum_stable_comparisons'])
    return len(comparisons) >= needed and all(
        row['both_optimizer_success'] and row['color_RMS_K'] <= opt['numeric_color_budget_K']
        and row['template_RMS_e'] <= opt['numeric_template_budget_e']
        for row in comparisons[-needed:])


@torch.no_grad()
def refined_infer(raw_dn, p, *, initial_state=None, initial_beta=None, options=None):
    """Raw/calibration only; cached initializer must come from this same raw.

    The caller is responsible for checking cached raw provenance. No cached
    initializer is produced internally by another complete inverse.
    """
    opt = validated_options(options)
    started = time.perf_counter()
    raw_e = p.raw_to_electrons(m.raw_tensor(raw_dn, p))
    if (initial_state is None) != (initial_beta is None):
        raise ValueError('Both cached state and beta are required')
    if initial_state is None:
        state, beta = p.initialize(raw_e)
        initializer = 'raw_demodulation_only'
    else:
        state = initial_state.to(device=p.M.device, dtype=p.M.dtype)
        beta = initial_beta.to(device=p.M.device, dtype=p.M.dtype)
        initializer = 'hash_verified_same_raw_frozen_v3'
    state, beta = p.constrain(state, beta)
    objective = V3Objective(p, raw_e, deadline=started+opt['cost_limit_s'])
    x = objective.pack(state, beta)/objective.scale
    objective.evaluate(x*objective.scale)
    endpoint = from_state(p, raw_dn, state, beta, {})
    stages, comparisons, errors = [], [], []
    previous = None
    for stage in opt['stages']:
        try:
            fit = minimize(objective.fun, x, method='L-BFGS-B', jac=True,
                           bounds=np.column_stack([objective.lower, objective.upper]),
                           options={**stage, 'maxls': 25, 'maxcor': 20})
            value, grad, state, beta = objective.evaluate(fit.x*objective.scale)
            x = fit.x.copy()
            endpoint = from_state(p, raw_dn, state, beta, {})
            scaled_gradient = grad*objective.scale
            projected_gradient = np.where(
                ((x <= objective.lower+1e-12) & (scaled_gradient > 0)) |
                ((x >= objective.upper-1e-12) & (scaled_gradient < 0)), 0., scaled_gradient)
            gradient_inf = float(np.max(np.abs(projected_gradient)))
            stage_kkt = gradient_inf <= stage['gtol']*(1+1e-6)
            record = {'configuration': stage, 'optimizer_success': bool(fit.success) and stage_kkt,
                      'optimizer_reported_success': bool(fit.success),
                      'projected_gradient_tolerance_passed': stage_kkt,
                      'message': str(fit.message), 'iterations': int(fit.nit), 'objective': value,
                      'physical_gradient_l2': float(np.linalg.norm(grad)),
                      'scaled_projected_gradient_inf': gradient_inf}
            if previous is not None:
                comparisons.append({**output_difference(previous, endpoint),
                                    'both_optimizer_success': stages[-1]['optimizer_success'] and record['optimizer_success']})
            stages.append(record)
            previous = endpoint
            if plateau_accepts(comparisons, opt):
                break
        except TimeoutError as error:
            errors.append(str(error))
            # Retain the latest finite evaluated iterate, with no certificate.
            _, _, state, beta = objective.last
            endpoint = from_state(p, raw_dn, state, beta, {})
            break
    finite = all(bool(torch.isfinite(endpoint[k]).all()) for k in (
        'temperature_color_K', 'template_postblur_e_ref', 'predicted_raw_e'))
    nonnegative = bool(endpoint['predicted_raw_e'].min() >= 0)
    certified = finite and nonnegative and not errors and plateau_accepts(comparisons, opt)
    info = {
        'method': VERSION, 'status': 'OUTPUT_PLATEAU_CERTIFIED' if certified else 'FINITE_NUMERICAL_UNCERTIFIED',
        'numerical_certified': certified, 'objective_version': 'unchanged_frozen_v3_weighted_raw_plus_latent_prior',
        'solver_version': VERSION, 'solver': 'float64 analytic-gradient bounded L-BFGS-B',
        'same_original_regularization': objective.cfg['regularization'],
        'initializer': initializer, 'CG_directions_used': False,
        'coordinate_anchor': {'T_K': p.cfg['reference_temperature_K'], 'amplitude': .5},
        'options': opt, 'stages': stages, 'output_comparisons': comparisons, 'errors': errors,
        'returned_endpoint': 'last_tightest_completed_stage' if not errors else 'latest_finite_evaluated_iterate',
        'finite_output': finite, 'nonnegative_predicted_measurement': nonnegative,
        'truth_used': False, 'wall_s': time.perf_counter()-started,
        'objective_evaluations': objective.evaluations,
        'objective_history_summary': {'first': objective.history[0], 'last': objective.history[-1]},
        'certification_scope': 'bounded numerical output stability; no global optimum or scientific accuracy claim',
    }
    endpoint.update(info=info, numerical_valid=certified)
    return endpoint
