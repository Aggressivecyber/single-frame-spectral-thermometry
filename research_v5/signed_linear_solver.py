"""Frozen-continuum, signed template/background quadratic inversion.

S is a latent field in electrons, not the old species-normalized state.  Beta
retains the three normalized coefficients used by SpectralPhysics.  Every raw
prediction contains exactly one object blur.  No decoded image appears in the
linear objective, and this module never receives temperature truth or labels.
"""
from __future__ import annotations

import math
from typing import Callable

import torch


DEFAULT_OPTIONS = {
    "alpha_g": 0.0,
    "lambda0_factor": 1e-6,
    "lambda_beta_factor": 1e-6,
    "rtol": 1e-4,
    "max_iterations": 200,
    "diagonal_mode": "exact_cyclic",
    "residual_refresh_interval": 25,
}
ALPHA_G_GRID = (0.0, 0.01, 0.1, 1.0)


def laplacian(field):
    """D^T D for periodic, unit-pixel forward differences in x and y."""
    return (4 * field - torch.roll(field, 1, -1) - torch.roll(field, -1, -1)
            - torch.roll(field, 1, -2) - torch.roll(field, -1, -2))


def _filter(image, otf):
    return torch.fft.ifft2(torch.fft.fft2(image) * otf).real


def fixed_raw_weights(physics, raw_e, supplied=None):
    """Return detached positive weights, fixed throughout a quadratic solve."""
    weights = ((raw_e.clamp_min(0) + physics.extra_variance).reciprocal()
               if supplied is None else torch.as_tensor(
                   supplied, dtype=raw_e.dtype, device=raw_e.device))
    if weights.ndim == 2:
        weights = weights[None]
    try:
        weights = torch.broadcast_to(weights, raw_e.shape)
    except RuntimeError as error:
        raise ValueError("weights must broadcast to B,H,W") from error
    if not bool(torch.isfinite(weights).all() and (weights > 0).all()):
        raise ValueError("Fixed raw weights must be finite and strictly positive")
    return weights.detach().clone()


class SignedLinearOperator:
    """Exact scalar implementation of op(M[:,2]*S) and op(q*B(beta)).

    Fusing the seven constant template responses before coding is algebraically
    identical to CodedOperator.forward.  It saves six redundant scalar blurs;
    it does not remove or repeat object blur.  Background columns are formed by
    the full existing CodedOperator once and then reused by linearity.
    """
    def __init__(self, physics):
        self.physics = physics
        if tuple(physics.M.shape) != (7, 3):
            raise ValueError("The frozen synthetic response must be 7 by 3")
        if not torch.equal(physics.active.bool(),
                           physics.active.new_tensor([0, 0, 1]).bool()):
            raise ValueError("v5 supports only the frozen active template index 2")
        if bool((physics.M[5:7, 2] != 0).any() or (physics.q[5:7] != 0).any()):
            raise ValueError("The 700/905 continuum channels must be template/background free")
        self.shape = tuple(physics.cfg["shape"])
        self.pixel_count = math.prod(self.shape)
        self.template_pattern = (physics.M[:, 2, None, None]
                                 * physics.op.patterns).sum(0)
        radiance = (physics.Bscale * physics.background_basis[:, None]
                    * physics.q[None, :, None, None])
        self.beta_images = physics.op(radiance).detach()
        self.stationary_normal_symbol = self._stationary_symbol()
        fy, fx = torch.meshgrid(
            torch.fft.fftfreq(self.shape[0], dtype=physics.M.dtype,
                             device=physics.M.device),
            torch.fft.fftfreq(self.shape[1], dtype=physics.M.dtype,
                             device=physics.M.device), indexing="ij")
        self.laplacian_symbol = (4 * torch.sin(math.pi * fx).square()
                                 + 4 * torch.sin(math.pi * fy).square())

    def _stationary_symbol(self):
        """Uniform-W normal symbol; carrier cross terms vanish after H.

        The declared geometry separates all DC/carrier/conjugate centers by
        more than twice the object bandwidth.  Thus H kills the off-diagonal
        carrier cross terms for constant W.  For varying W this symbol is only
        a preconditioner and a declared diagonal-scale approximation.
        """
        p = self.physics
        gain = p.M.new_tensor(p.cfg["channel_gain"])
        depth = p.M.new_tensor(p.cfg["modulation_depth"])
        response = p.M[:, 2]
        dc = 0.5 * (response * gain).sum()
        relay_square = p.op.relay_otf.square()
        power = dc.square() * relay_square
        for k, (kx, ky) in enumerate(p.op.bins):
            coefficient_square = (response[k] * gain[k] * depth[k] / 4).square()
            power = power + coefficient_square * (
                torch.roll(relay_square, (-ky, -kx), (-2, -1))
                + torch.roll(relay_square, (ky, kx), (-2, -1)))
        return p.op.object_otf.square() * power

    def apply_template(self, field):
        postblur = self.physics.op.object_blur(field)
        return _filter(self.template_pattern * postblur, self.physics.op.relay_otf)

    def adjoint_template(self, residual):
        back = _filter(residual, self.physics.op.relay_otf)
        return self.physics.op.object_blur(self.template_pattern * back)

    def apply_beta(self, beta):
        return torch.einsum("...i,ihw->...hw", beta, self.beta_images)

    def adjoint_beta(self, residual):
        return torch.einsum("...hw,ihw->...i", residual, self.beta_images)

    def template_diagonal(self, weights, mode):
        if mode == "exact_cyclic":
            return self._exact_cyclic_diagonal(weights)
        if mode == "stationary_weight_mean":
            scalar = weights.mean() * self.stationary_normal_symbol.mean()
            return scalar.expand(self.shape), {
                "method": "stationary_weight_mean",
                "exact_for_uniform_weights": True,
                "approximate_for_nonuniform_weights": True,
                "formula": "mean(W)*mean_f[H(f)^2*(dc^2*R(f)^2+sum_carriers |c_k|^2*(R(f+k)^2+R(f-k)^2))]",
                "cross_carrier_terms": "zero under declared nonoverlapping object bandwidth",
            }
        if mode != "exact_small":
            raise ValueError("diagonal_mode is exact_cyclic, stationary_weight_mean or exact_small")
        if self.pixel_count > 1024:
            raise ValueError("exact_small diagonal is restricted to <=1024 pixels")
        diagonal = weights.new_empty(self.pixel_count)
        for start in range(0, self.pixel_count, 64):
            count = min(64, self.pixel_count - start)
            basis = weights.new_zeros((count, self.pixel_count))
            rows = torch.arange(count, device=weights.device)
            basis[rows, start + rows] = 1
            columns = self.apply_template(basis.reshape(count, *self.shape))
            diagonal[start:start + count] = (columns.square() * weights).sum((-2, -1))
        return diagonal.reshape(self.shape), {
            "method": "exact_small_basis_columns",
            "exact_for_uniform_weights": True,
            "approximate_for_nonuniform_weights": False,
            "formula": "diag_j=sum_pixels W*(A_S e_j)^2",
        }

    def _exact_cyclic_diagonal(self, weights):
        """Exact nonuniform-W diagonal using the finite carrier-pair formula.

        For P(k)=sum_l c_l exp(i f_l k), A e_j(r) is
        sum_l c_l exp(i f_l j) G_l(r-j), with
        FFT(G_l)=R(f) H(f-f_l). Squared columns reduce to finite carrier-pair
        correlations with W, rather than 65536 separately formed columns.
        Tiny sampled-pattern roundoff is measured explicitly; no carrier cross
        term is dropped for varying W. The stationary symbol remains only the
        Fourier preconditioner and the separately declared noise proxy.
        """
        if not hasattr(self, '_diagonal_pair_cache'):
            p = self.physics
            h, w = self.shape
            frequencies = [(0, 0)]
            for response, (kx, ky) in zip(p.M[:, 2], p.op.bins):
                if float(response) != 0.:
                    frequencies.extend(((kx, ky), (-kx, -ky)))
            frequencies = list(dict.fromkeys((kx % w, ky % h) for kx, ky in frequencies))
            pattern_fft = torch.fft.fft2(self.template_pattern)
            projected_fft = torch.zeros_like(pattern_fft)
            terms = []
            for kx, ky in frequencies:
                coefficient = pattern_fft[ky, kx] / self.pixel_count
                projected_fft[ky, kx] = pattern_fft[ky, kx]
                kernel = coefficient * torch.fft.ifft2(
                    p.op.relay_otf * torch.roll(p.op.object_otf, (ky, kx), (-2, -1)))
                terms.append(((kx, ky), kernel))
            projection = torch.fft.ifft2(projected_fft).real
            relative_pattern_error = float((projection-self.template_pattern).abs().max()
                                           / self.template_pattern.abs().max().clamp_min(1e-30))
            limit = 2e-5 if p.M.dtype == torch.float32 else 1e-11
            if relative_pattern_error > limit:
                raise ValueError('Sampled template pattern has unregistered non-carrier content')
            kernels, phases = [], []
            for li, ((lx, ly), left) in enumerate(terms):
                for mi in range(li, len(terms)):
                    (mx, my), right = terms[mi]
                    product = left * right.conj()
                    # Correlation sum_t W[j+t] product[t]. FFT at negative
                    # frequency is conj(FFT(conj(product))), also for complex K.
                    kernels.append(torch.fft.fft2(product.conj()).conj()
                                   * (1 if li == mi else 2))
                    phase_fft = torch.zeros_like(pattern_fft)
                    phase_fft[(ly-my) % h, (lx-mx) % w] = self.pixel_count
                    phases.append(torch.fft.ifft2(phase_fft))
            self._diagonal_pair_cache = (torch.stack(kernels), torch.stack(phases),
                                         relative_pattern_error, len(terms))
        kernels, phases, pattern_error, term_count = self._diagonal_pair_cache
        correlations = torch.fft.ifft2(torch.fft.fft2(weights)[None] * kernels)
        diagonal = (correlations*phases).real.sum(0)
        if not bool(torch.isfinite(diagonal).all() and (diagonal > 0).all()):
            raise ValueError('Nonpositive/nonfinite exact cyclic normal diagonal')
        return diagonal, {
            'method': 'exact_cyclic_finite_carrier_pair_correlations',
            'exact_for_uniform_weights': True, 'approximate_for_nonuniform_weights': False,
            'formula': 'diag_j=sum_r W[r]*(A_S e_j)[r]^2; all finite carrier pairs retained',
            'finite_carrier_terms': term_count, 'carrier_pairs': len(kernels),
            'relative_sampled_pattern_projection_roundoff': pattern_error,
            'floating_point_scope': 'exact finite-carrier model up to recorded sampled-pattern roundoff',
        }


def pcg_true_residual(matvec: Callable, rhs, precondition: Callable, *,
                      residual_scale, rtol=1e-4, max_iterations=200,
                      refresh_interval=25, objective_zero=None):
    """PCG with explicitly recomputed normal-equation residual every iteration.

    Both the physical-coordinate and diagonally equilibrated true residual
    must satisfy rtol.  The second check prevents the three large-scale beta
    coordinates from hiding an unresolved S residual.  Neither scaling nor
    preconditioning changes the quadratic objective or its normal equations.
    """
    if not 0 < rtol <= 1e-4 or not 1 <= max_iterations <= 200:
        raise ValueError("Require 0<rtol<=1e-4 and 1<=max_iterations<=200")
    x = torch.zeros_like(rhs)
    rhs_norm = torch.linalg.vector_norm(rhs)
    scale_root = residual_scale.sqrt()
    equilibrated_norm = torch.linalg.vector_norm(rhs / scale_root)
    if float(rhs_norm) == 0:
        return x, {"iterations": 0, "relative_true_residual": 0.0,
                   "relative_residual": 0.0, "equilibrated_relative_true_residual": 0.0,
                   "true_residual_history": [], "equilibrated_true_residual_history": [],
                   "recursive_residual_history": [], "residual_refreshes": [],
                   "objective_history": ([] if objective_zero is None else
                                         [{"iteration": 0, "objective": objective_zero}]),
                   "stop": "ZERO_RHS", "converged": True}
    r = rhs.clone()
    z = precondition(r)
    direction = z.clone()
    rz = (r * z).sum()
    true_history, equilibrated_history, recursive_history, refreshes = [], [], [], []
    objective_history = ([] if objective_zero is None else
                         [{"iteration": 0, "objective": objective_zero}])
    stop = "MAX_ITERATIONS"
    for iteration in range(1, max_iterations + 1):
        if not bool(torch.isfinite(rz)) or float(rz) <= 0:
            stop = "PRECONDITIONER_BREAKDOWN"
            break
        product = matvec(direction)
        denominator = (direction * product).sum()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            stop = "NON_POSITIVE_CURVATURE"
            break
        step = rz / denominator
        candidate = x + step * direction
        if not bool(torch.isfinite(candidate).all()):
            stop = "NONFINITE_ITERATE"
            break
        x = candidate
        r = r - step * product
        r_true = rhs - matvec(x)
        relative = float(torch.linalg.vector_norm(r_true) / rhs_norm)
        equilibrated = float(torch.linalg.vector_norm(r_true / scale_root) / equilibrated_norm)
        recursive = float(torch.linalg.vector_norm(r) / rhs_norm)
        true_history.append(relative)
        equilibrated_history.append(equilibrated)
        recursive_history.append(recursive)
        if objective_zero is not None:
            # J(x)=J(0)+x^T N x/2-x^T rhs, with the same actual fixed-W N.
            # Reuse the explicitly recomputed N*x; no second measurement cost.
            objective = (objective_zero + 0.5 * float((x.double() * (rhs - r_true).double()).sum())
                         - float((x.double() * rhs.double()).sum()))
            objective_history.append({"iteration": iteration, "objective": objective})
        if not math.isfinite(relative) or not math.isfinite(equilibrated):
            stop = "NONFINITE_TRUE_RESIDUAL"
            break
        if relative <= rtol and equilibrated <= rtol:
            stop = "RELATIVE_TRUE_RESIDUAL"
            break
        refresh = ((refresh_interval > 0 and iteration % refresh_interval == 0)
                   or recursive <= rtol)
        if refresh:
            r = r_true
            z = precondition(r)
            direction = z.clone()
            rz = (r * z).sum()
            refreshes.append(iteration)
            continue
        z = precondition(r)
        next_rz = (r * z).sum()
        if not bool(torch.isfinite(next_rz)) or float(next_rz) <= 0:
            stop = "PRECONDITIONER_BREAKDOWN"
            break
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    final_residual = rhs - matvec(x)
    relative = float(torch.linalg.vector_norm(final_residual) / rhs_norm)
    equilibrated = float(torch.linalg.vector_norm(final_residual / scale_root) / equilibrated_norm)
    return x, {
        "iterations": len(true_history), "relative_true_residual": relative,
        "relative_residual": relative,
        "equilibrated_relative_true_residual": equilibrated,
        "true_residual_history": true_history,
        "equilibrated_true_residual_history": equilibrated_history,
        "recursive_residual_history": recursive_history,
        "residual_refreshes": refreshes, "stop": stop,
        "objective_history": objective_history,
        "converged": relative <= rtol and equilibrated <= rtol,
        "rhs_norm_physical": float(rhs_norm),
        "rhs_norm_equilibrated": float(equilibrated_norm),
        "rtol": rtol, "max_iterations": max_iterations,
    }


def _options(options):
    options = {} if options is None else dict(options)
    aliases = {"cg_relative_tolerance": "rtol", "cg_max_iterations": "max_iterations"}
    for old, new in aliases.items():
        if old in options:
            if new in options:
                raise ValueError(f"Specify only {old} or {new}")
            options[new] = options.pop(old)
    unknown = set(options) - set(DEFAULT_OPTIONS) - {"weights"}
    if unknown:
        raise ValueError(f"Unknown signed solver options: {sorted(unknown)}")
    config = {**DEFAULT_OPTIONS, **options}
    if float(config["alpha_g"]) not in ALPHA_G_GRID:
        raise ValueError(f"alpha_g must be in the preregistered grid {ALPHA_G_GRID}")
    for key in ("lambda0_factor", "lambda_beta_factor"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    return config


def _create_single_frame_system(physics, raw_frame, continuum_raw, W, linear, config, fixed_beta):
    """Shared normal system for inversion and finite Hessian diagnostics."""
    shape, n = linear.shape, linear.pixel_count
    target = raw_frame - continuum_raw
    if fixed_beta is not None:
        target = target - linear.apply_beta(fixed_beta)
    diag_s, diagonal_info = linear.template_diagonal(W, config["diagonal_mode"])
    median_s = diag_s.median()
    beta_flat = linear.beta_images.flatten(1)
    beta_gram = (beta_flat * W.flatten()[None]) @ beta_flat.T
    diag_beta = beta_gram.diagonal()
    median_beta = diag_beta.median()
    if not bool(torch.isfinite(median_s) and median_s > 0
                and torch.isfinite(median_beta) and median_beta > 0):
        raise ValueError("Nonpositive/nonfinite normal diagonal scale")
    lambda_g = float(config["alpha_g"]) * median_s
    lambda0 = float(config["lambda0_factor"]) * median_s
    lambda_beta = float(config["lambda_beta_factor"]) * median_beta

    def unpack(vector):
        field = vector[:n].reshape(shape)
        beta = None if fixed_beta is not None else vector[n:]
        return field, beta

    def pack(field, beta):
        return field.flatten() if beta is None else torch.cat([field.flatten(), beta])

    def apply_measurement(vector):
        field, beta = unpack(vector)
        image = linear.apply_template(field)
        return image if beta is None else image + linear.apply_beta(beta)

    def adjoint_measurement(image):
        return pack(linear.adjoint_template(image),
                    None if fixed_beta is not None else linear.adjoint_beta(image))

    def measurement_matvec(vector):
        return adjoint_measurement(W * apply_measurement(vector))

    def matvec(vector):
        field, beta = unpack(vector)
        return measurement_matvec(vector) + pack(
            lambda_g * laplacian(field) + lambda0 * field,
            None if beta is None else lambda_beta * beta)

    rhs = adjoint_measurement(W * target)
    denominator = (W.mean() * linear.stationary_normal_symbol
                   + lambda_g * linear.laplacian_symbol + lambda0)
    beta_regularized = beta_gram + lambda_beta * torch.eye(3, dtype=W.dtype, device=W.device)
    beta_cholesky = torch.linalg.cholesky(beta_regularized)

    def precondition(vector):
        field, beta = unpack(vector)
        s_part = torch.fft.ifft2(torch.fft.fft2(field) / denominator).real
        b_part = None if beta is None else torch.cholesky_solve(
            beta[:, None], beta_cholesky).squeeze(-1)
        return pack(s_part, b_part)

    scale_s = torch.ones_like(target) * (median_s + 4 * lambda_g + lambda0)
    residual_scale = pack(scale_s, None if fixed_beta is not None else diag_beta + lambda_beta)

    def objective(field, beta):
        residual = (raw_frame - continuum_raw - linear.apply_template(field)
                    - linear.apply_beta(beta))
        data = 0.5 * (W * residual.square()).sum()
        gradient_energy = ((field - torch.roll(field, 1, -1)).square()
                           + (field - torch.roll(field, 1, -2)).square()).sum()
        gradient_penalty = 0.5 * lambda_g * gradient_energy
        ridge_s = 0.5 * lambda0 * field.square().sum()
        ridge_beta = 0.5 * lambda_beta * beta.square().sum()
        return data + gradient_penalty + ridge_s + ridge_beta, {
            "data": float(data), "gradient_penalty": float(gradient_penalty),
            "ridge_S": float(ridge_s), "ridge_beta": float(ridge_beta),
            "gradient_energy_S_e2": float(gradient_energy),
        }

    return {
        "matvec": matvec, "measurement_matvec": measurement_matvec,
        "measurement_forward": apply_measurement, "measurement_adjoint": adjoint_measurement,
        "precondition": precondition, "rhs": rhs, "pack": pack, "unpack": unpack,
        "residual_scale": residual_scale, "objective": objective,
        "weights": W, "linear_operator": linear,
        "shape": shape, "n_S": n, "n_beta": 0 if fixed_beta is not None else 3,
        "dimension": len(rhs), "diag_S": diag_s, "diag_beta": diag_beta,
        "median_diag_S": median_s, "median_diag_beta": median_beta,
        "lambda_g": lambda_g, "lambda0": lambda0, "lambda_beta": lambda_beta,
        "diagonal_info": diagonal_info, "options": config,
        "continuum_raw_e": continuum_raw, "fixed_beta": fixed_beta,
        "raw_e": raw_frame,
    }


@torch.no_grad()
def create_linear_system(physics, raw_e, continuum_latent, options=None, *, fixed_beta=None):
    """Expose the identical single-frame normal system for labelled diagnostics.

    Returned callables use actual fixed raw weights and the same declared
    regularization as solve_signed_linear.  Hessian uncertainty probes can use
    measurement_matvec for the sandwich covariance without reconstructing an
    alternative operator.  This factory accepts no truth or input identity.
    """
    config = _options(options)
    raw = torch.as_tensor(raw_e, dtype=physics.M.dtype, device=physics.M.device).detach()
    if raw.ndim == 2:
        raw = raw[None]
    shape = tuple(physics.cfg["shape"])
    if tuple(raw.shape) != (1, *shape):
        raise ValueError("create_linear_system requires one frame, shape 1,H,W or H,W")
    continuum = torch.as_tensor(continuum_latent, dtype=raw.dtype, device=raw.device).detach().clone()
    if continuum.ndim == 3:
        continuum = continuum[None]
    if tuple(continuum.shape) != (1, 7, *shape):
        raise ValueError("continuum_latent must have shape 1,7,H,W or 7,H,W")
    if not bool(torch.isfinite(raw).all() and torch.isfinite(continuum).all()):
        raise ValueError("Inputs must be finite")
    beta = None
    if fixed_beta is not None:
        beta = torch.as_tensor(fixed_beta, dtype=raw.dtype, device=raw.device).detach().reshape(-1)
        if tuple(beta.shape) != (3,) or not bool(torch.isfinite(beta).all()):
            raise ValueError("fixed_beta must contain three finite coefficients")
        beta = beta.clone()
    W = fixed_raw_weights(physics, raw, config.get("weights"))[0]
    linear = SignedLinearOperator(physics)
    return _create_single_frame_system(physics, raw[0], physics.op(continuum)[0], W,
                                       linear, config, beta)


@torch.no_grad()
def solve_signed_linear(physics, raw_e, continuum_latent, options=None, *, fixed_beta=None):
    """Minimize the declared fixed-W convex quadratic, independently per frame.

    Passing fixed_beta restricts the same objective to a supplied background
    and is reserved for labelled diagnostic oracles.  Deployment passes None
    and jointly estimates S and beta.  No positivity constraint is applied.
    """
    config = _options(options)
    raw_e = torch.as_tensor(raw_e, dtype=physics.M.dtype, device=physics.M.device).detach()
    if raw_e.ndim == 2:
        raw_e = raw_e[None]
    shape = tuple(physics.cfg["shape"])
    if raw_e.ndim != 3 or tuple(raw_e.shape[-2:]) != shape:
        raise ValueError("raw_e must have shape B,H,W")
    continuum_latent = torch.as_tensor(continuum_latent, dtype=raw_e.dtype,
                                       device=raw_e.device).detach().clone()
    if tuple(continuum_latent.shape) != (len(raw_e), 7, *shape):
        raise ValueError("continuum_latent must have shape B,7,H,W")
    if not bool(torch.isfinite(raw_e).all() and torch.isfinite(continuum_latent).all()):
        raise ValueError("Inputs must be finite")
    weights = fixed_raw_weights(physics, raw_e, config.get("weights"))
    linear = SignedLinearOperator(physics)
    continuum_raw = physics.op(continuum_latent)
    if fixed_beta is not None:
        fixed_beta = torch.as_tensor(fixed_beta, dtype=raw_e.dtype, device=raw_e.device).detach()
        if fixed_beta.ndim == 1:
            fixed_beta = fixed_beta[None]
        if tuple(fixed_beta.shape) != (len(raw_e), 3) or not bool(torch.isfinite(fixed_beta).all()):
            raise ValueError("fixed_beta must have shape B,3 and finite values")
        fixed_beta = fixed_beta.clone()
    fields, betas, expected, reports = [], [], [], []
    for frame in range(len(raw_e)):
        W = weights[frame]
        bg_fixed = None if fixed_beta is None else fixed_beta[frame]
        system = _create_single_frame_system(physics, raw_e[frame], continuum_raw[frame], W,
                                              linear, config, bg_fixed)
        diag_s, diagonal_info = system["diag_S"], system["diagonal_info"]
        diag_beta = system["diag_beta"]
        median_s, median_beta = system["median_diag_S"], system["median_diag_beta"]
        lambda_g, lambda0, lambda_beta = system["lambda_g"], system["lambda0"], system["lambda_beta"]
        objective = system["objective"]
        zero_field = torch.zeros_like(W)
        zero_beta = bg_fixed if bg_fixed is not None else W.new_zeros(3)
        before, before_terms = objective(zero_field, zero_beta)
        solution, cg = pcg_true_residual(
            system["matvec"], system["rhs"], system["precondition"],
            residual_scale=system["residual_scale"],
            rtol=float(config["rtol"]), max_iterations=int(config["max_iterations"]),
            refresh_interval=int(config["residual_refresh_interval"]), objective_zero=float(before))
        field, estimated_beta = system["unpack"](solution)
        beta = bg_fixed if bg_fixed is not None else estimated_beta
        after, after_terms = objective(field, beta)
        acceptance_allowance = 32 * torch.finfo(W.dtype).eps * max(1.0, abs(float(before)))
        accepted = bool(torch.isfinite(after) and torch.isfinite(field).all()
                        and torch.isfinite(beta).all()
                        and float(after) <= float(before) + acceptance_allowance)
        if not accepted:
            field, beta = zero_field, zero_beta
            after, after_terms = before, before_terms
        prediction = (continuum_raw[frame] + linear.apply_template(field)
                      + linear.apply_beta(beta))
        reports.append({
            "frame": frame, "CG": cg, "accepted": accepted,
            "acceptance": "finite fixed-quadratic objective <= zero-variable objective (roundoff allowance only)",
            "acceptance_roundoff_allowance": acceptance_allowance,
            "objective_before": float(before), "objective_after": float(after),
            "objective_terms_before": before_terms, "objective_terms_after": after_terms,
            "objective_history": cg["objective_history"],
            "best_iteration_by_objective": min(cg["objective_history"], key=lambda h: h["objective"])["iteration"],
            "output_iteration": cg["iterations"] if accepted else 0,
            "best_iteration_role": "diagnostic hindsight on the fixed raw quadratic; returned output is final CG iterate",
            "alpha_g": float(config["alpha_g"]), "lambda_g": float(lambda_g),
            "lambda0": float(lambda0), "lambda_beta": float(lambda_beta),
            "lambda0_factor": float(config["lambda0_factor"]),
            "lambda_beta_factor": float(config["lambda_beta_factor"]),
            "median_diag_A_S_T_W_A_S": float(median_s),
            "diagonal_S_min": float(diag_s.min()), "diagonal_S_max": float(diag_s.max()),
            "diagonal_S": diagonal_info,
            "diag_A_beta_T_W_A_beta": [float(v) for v in diag_beta],
            "median_diag_A_beta_T_W_A_beta": float(median_beta),
            "beta_diagonal_method": "exact three coded background columns",
            "preconditioner": "uniform-W Fourier S normal plus exact regularized 3x3 beta Gram; block diagonal",
            "preconditioner_weight_approximation": "W replaced by its spatial mean only in S preconditioner/declared approximate diagonal scale",
            "normal_equations_use_actual_fixed_W": True, "hidden_damping": False,
            "fixed_beta_diagnostic_oracle": bg_fixed is not None,
            "expected_e_min": float(prediction.min()),
            "expected_e_max": float(prediction.max()),
            "expected_e_nonnegative": bool((prediction >= 0).all()),
            "no_template_positivity_constraint": True,
            "numeric_dtype": str(W.dtype), "S_units": "e", "beta_units": "normalized 3 coefficients; Bscale applied in operator",
            "gradient_grid": "periodic unit-pixel differences; lambda_g units e^-2 for the dimensionless weighted objective",
            "weight_min": float(W.min()), "weight_max": float(W.max()), "weight_mean": float(W.mean()),
        })
        fields.append(field)
        betas.append(beta)
        expected.append(prediction)
    template_latent = torch.stack(fields)
    return {
        "template_latent_e": template_latent,
        "template_postblur_e": physics.op.object_blur(template_latent),
        "beta": torch.stack(betas), "expected_e": torch.stack(expected),
        "weights": weights, "continuum_raw_e": continuum_raw,
        "info": {
            "frames": reports, "options": {k: v for k, v in config.items() if k != "weights"},
            "measurement_objective": "one fixed-W raw likelihood; decoded images are absent",
            "weight_rule": ("explicit fixed diagnostic weights" if config.get("weights") is not None
                            else "1/(clamp(observed raw_e,0)+read_noise_e^2+adc_step^2/12)"),
            "all_CG_converged": all(r["CG"]["converged"] for r in reports),
            "all_accepted": all(r["accepted"] for r in reports),
            "continuum_frozen": True,
        },
    }
