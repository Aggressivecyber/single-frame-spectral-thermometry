"""Bounded CPU float64 controls for v5 operators and the fixed quadratic solve.

These small synthetic numerical controls are not 256x256 scientific acceptance
or a hardware benchmark.  They use the frozen calibration with only image
shape/carrier bins rescaled for a declared 16x16 test.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time

from research_v5 import common

common.configure_runtime()

import torch

from research_v5.continuum_solver import estimate_continuum
from research_v5.signed_linear_solver import (
    SignedLinearOperator, create_linear_system, fixed_raw_weights, solve_signed_linear,
)


def _relative(first, second):
    scale = max(float(torch.linalg.vector_norm(first)), float(torch.linalg.vector_norm(second)), 1e-30)
    return float(torch.linalg.vector_norm(first - second)) / scale


def run_tests():
    started = time.perf_counter()
    torch.set_num_threads(2)
    config = copy.deepcopy(common.physics_config())
    original_shape = list(config["shape"])
    size = 16
    config["shape"] = [size, size]
    config["carrier_bins_xy"] = [[round(kx * size / original_shape[1]),
                                  round(ky * size / original_shape[0])]
                                 for kx, ky in config["carrier_bins_xy"]]
    physics = common.make_physics("cpu", torch.float64, config=config)
    rng = torch.Generator(device="cpu").manual_seed(510132)
    checks = []

    def check(name, value, limit, *, lower=False, detail=None):
        checks.append({"name": name, "value": float(value), "limit": float(limit),
                       "criterion": ">=" if lower else "<=",
                       "passed": float(value) >= limit if lower else float(value) <= limit,
                       "detail": detail})

    def random(*shape):
        return torch.randn(shape, dtype=torch.float64, generator=rng)

    bands, raw_direction = random(1, 7, size, size), random(1, size, size)
    left = (physics.op(bands) * raw_direction).sum()
    right = (bands * physics.op.adjoint(raw_direction)).sum()
    check("full_coded_forward_adjoint_inner_product", abs(float(left - right))
          / max(abs(float(left)), abs(float(right)), 1e-30), 2e-12)

    yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float64),
                            torch.arange(size, dtype=torch.float64), indexing="ij")
    state = torch.zeros((1, 5, size, size), dtype=torch.float64)
    state[:, 0] = (3052.5 - physics.T0) / physics.Tscale
    state[:, 1] = -0.2 + 0.025 * torch.cos(2 * torch.pi * xx / size)
    beta = torch.tensor([[0.7, 0.02, -0.01]], dtype=torch.float64)
    dz, db = random(1, 5, size, size), random(1, 3)
    jvp = physics.jvp(state, beta, dz, db)
    gz, gb = physics.vjp(state, beta, raw_direction)
    left, right = (jvp * raw_direction).sum(), (dz * gz).sum() + (db * gb).sum()
    check("spectral_JVP_VJP_inner_product", abs(float(left - right))
          / max(abs(float(left)), abs(float(right)), 1e-30), 2e-12)
    epsilon = 1e-7
    finite_difference = (physics(state + epsilon * dz, beta + epsilon * db)
                         - physics(state - epsilon * dz, beta - epsilon * db)) / (2 * epsilon)
    check("spectral_JVP_centered_finite_difference", _relative(jvp, finite_difference), 2e-7)

    linear = SignedLinearOperator(physics)
    field = random(size, size)
    exact_template = physics.op(physics.M[:, 2, None, None] * field[None])
    fused_template = linear.apply_template(field)
    check("fused_template_matches_full_operator_once_blur", _relative(exact_template, fused_template), 2e-12)
    beta_direction = random(3)
    exact_background = physics.op(physics.q[:, None, None] * physics.Bscale
                                  * torch.einsum("i,ihw->hw", beta_direction,
                                                 physics.background_basis)[None])
    check("background_columns_match_full_operator_once_blur",
          _relative(exact_background, linear.apply_beta(beta_direction)), 2e-12)
    probe = random(size, size)
    left = (linear.apply_template(field) * probe).sum()
    right = (field * linear.adjoint_template(probe)).sum()
    check("signed_template_adjoint_inner_product", abs(float(left - right))
          / max(abs(float(left)), abs(float(right)), 1e-30), 2e-12)
    left = (linear.apply_beta(beta_direction) * probe).sum()
    right = (beta_direction * linear.adjoint_beta(probe)).sum()
    check("three_background_coefficients_adjoint_inner_product", abs(float(left - right))
          / max(abs(float(left)), abs(float(right)), 1e-30), 2e-12)

    impulse_fields = torch.eye(size * size, dtype=torch.float64).reshape(-1, size, size)
    template_columns = linear.apply_template(impulse_fields).flatten(1).T
    beta_columns = linear.beta_images.flatten(1).T
    design = torch.cat([template_columns, beta_columns], dim=1)
    exact_uniform_diagonal = template_columns.square().sum(0)
    analytic_uniform_diagonal = linear.stationary_normal_symbol.mean().expand_as(exact_uniform_diagonal)
    check("uniform_W_analytic_diagonal_matches_all_dense_columns",
          _relative(exact_uniform_diagonal, analytic_uniform_diagonal), 2e-12)
    exact_uniform_normal = linear.adjoint_template(linear.apply_template(field))
    analytic_uniform_normal = torch.fft.ifft2(torch.fft.fft2(field)
                                             * linear.stationary_normal_symbol).real
    check("uniform_W_Fourier_normal_symbol_matches_exact_normal",
          _relative(exact_uniform_normal, analytic_uniform_normal), 2e-12)

    nonuniform_weights = .5 + torch.rand((size, size), dtype=torch.float64, generator=rng)
    exact_nonuniform_diagonal = (template_columns.square()*nonuniform_weights.flatten()[:, None]).sum(0)
    cyclic_diagonal, cyclic_info = linear.template_diagonal(nonuniform_weights, 'exact_cyclic')
    check('nonuniform_W_exact_cyclic_diagonal_matches_all_dense_columns',
          _relative(cyclic_diagonal.flatten(), exact_nonuniform_diagonal), 2e-11,
          detail='all carrier cross terms retained; formal regularizer uses this actual-W median')
    check('exact_cyclic_diagonal_nonuniform_not_stationary_approximation',
          float(cyclic_info['approximate_for_nonuniform_weights']), 0.)

    continuum = physics.components(state, beta)[0]
    S = 45 + 12 * torch.cos(2 * torch.pi * yy / size)
    raw = physics.op(continuum) + linear.apply_template(S)[None] + linear.apply_beta(beta)
    raw_before, continuum_before = raw.clone(), continuum.clone()
    dense_results = []
    for alpha in (0.0, 0.1):
        options = {"alpha_g": alpha, "diagonal_mode": "exact_small", "rtol": 1e-10,
                   "max_iterations": 200}
        system = create_linear_system(physics, raw, continuum, options)
        W = system["weights"].flatten()
        dense_normal = design.T @ (W[:, None] * design)
        n = size * size
        L = (4 * impulse_fields - torch.roll(impulse_fields, 1, -1)
             - torch.roll(impulse_fields, -1, -1) - torch.roll(impulse_fields, 1, -2)
             - torch.roll(impulse_fields, -1, -2)).flatten(1).T
        dense_normal[:n, :n] += system["lambda_g"] * L
        ridge = torch.cat([torch.ones(n, dtype=torch.float64) * system["lambda0"],
                           torch.ones(3, dtype=torch.float64) * system["lambda_beta"]])
        dense_normal += torch.diag(ridge)
        vector = random(n + 3)
        check(f"alpha_{alpha}_matrix_free_normal_matches_dense",
              _relative(system["matvec"](vector), dense_normal @ vector), 2e-12)
        exact_diagonal = template_columns.square().T @ W
        check(f"alpha_{alpha}_exact_small_diagonal_matches_dense",
              _relative(system["diag_S"].flatten(), exact_diagonal), 2e-12)
        root_scale = system["residual_scale"].sqrt()
        equilibrated = dense_normal / (root_scale[:, None] * root_scale[None])
        reference = torch.linalg.solve(equilibrated, system["rhs"] / root_scale) / root_scale
        result = solve_signed_linear(physics, raw, continuum, options)
        solution = system["pack"](result["template_latent_e"][0], result["beta"][0])
        report = result["info"]["frames"][0]
        true_residual = system["rhs"] - system["matvec"](solution)
        measured_relative = float(torch.linalg.vector_norm(true_residual)
                                  / torch.linalg.vector_norm(system["rhs"]))
        check(f"alpha_{alpha}_CG_explicit_true_residual", measured_relative, 1e-10)
        check(f"alpha_{alpha}_CG_report_matches_recomputed_residual",
              abs(report["CG"]["relative_true_residual"] - measured_relative), 2e-12)
        check(f"alpha_{alpha}_PCG_dense_equilibrated_solution",
              _relative(solution, reference), 2e-5)
        reference_field = system["unpack"](reference)[0]
        check(f"alpha_{alpha}_PCG_dense_postblur_template",
              _relative(result["template_postblur_e"][0], physics.op.object_blur(reference_field)), 2e-8)
        check(f"alpha_{alpha}_quadratic_objective_nonincrease",
              report["objective_after"] - report["objective_before"], 1e-9)
        check(f"alpha_{alpha}_CG_iterations", report["CG"]["iterations"], 200)
        check(f"alpha_{alpha}_CG_converged", float(report["CG"]["converged"]), 1, lower=True)
        if report["objective_history"]:
            check(f"alpha_{alpha}_last_quadratic_history_matches_objective",
                  abs(report["objective_history"][-1]["objective"] - report["objective_after"])
                  / max(1.0, abs(report["objective_after"])), 1e-9)
        dense_results.append({"alpha_g": alpha, "frame_report": report,
                              "dense_reference": "float64 symmetrically equilibrated torch.linalg.solve",
                              "dense_dimension": n + 3})
    check("raw_input_unchanged", float(torch.equal(raw, raw_before)), 1, lower=True)
    check("frozen_continuum_input_unchanged", float(torch.equal(continuum, continuum_before)), 1, lower=True)

    base = physics.op(continuum) + linear.apply_beta(beta)
    diagnostic_weights = fixed_raw_weights(physics, base)
    signed_outputs = []
    for sign in (1, -1):
        signed_raw = base + sign * linear.apply_template(S)[None]
        result = solve_signed_linear(physics, signed_raw, continuum,
                                      {"weights": diagnostic_weights, "rtol": 1e-10}, fixed_beta=beta)
        signed_outputs.append(result["template_postblur_e"])
        check(f"fixed_beta_sign_{sign}_CG_true_residual",
              result["info"]["frames"][0]["CG"]["relative_true_residual"], 1e-10)
        check(f"fixed_beta_sign_{sign}_projection_sign",
              float((result["template_postblur_e"] * physics.op.object_blur(sign * S)[None]).sum()),
              0, lower=True)
    check("fixed_W_signed_antisymmetry", _relative(signed_outputs[0], -signed_outputs[1]), 2e-10)

    red_results = []
    for sign in (1, -1):
        signed_raw = base + sign * linear.apply_template(S)[None]
        warm = estimate_continuum(physics, signed_raw, {"warm_iterations": 2})
        red_results.append(warm)
        check(f"continuum_sign_{sign}_inactive_template_exact_zero",
              float(warm["state"][:, 2:].abs().max()), 0)
        check(f"continuum_sign_{sign}_postblur_exactly_once",
              _relative(warm["continuum_postblur"], physics.op.object_blur(warm["continuum_latent"])), 2e-12)
    check("two_band_continuum_invariance_to_signed_template",
          float((red_results[0]["temperature_color_K"] - red_results[1]["temperature_color_K"]).abs().max()),
          1e-5, detail="numerical carrier separation control; not a noisy scientific gate")

    from research_v5.reconstruct import reconstruct
    raw_dn = raw / physics.adc_step + config["black_level_DN"]
    calibration = {"physics_config": copy.deepcopy(config)}
    selection = {"dtype": "float64", "continuum_options": {"warm_iterations": 1},
                 "linear_options": {"max_iterations": 1}}
    calibration_before, selection_before = copy.deepcopy(calibration), copy.deepcopy(selection)
    deployed = reconstruct(raw_dn, calibration, selection)
    converged = deployed["info"]["linear"]["frames"][0]["CG"]["converged"]
    check("deployment_nonconvergence_masks_valid_pixels",
          float((~deployed["valid"]).all()) if not converged else 1.0, 1, lower=True)
    check("deployment_abstain_is_complement_valid",
          float(torch.equal(deployed["abstain"], ~deployed["valid"])), 1, lower=True)
    check("deployment_calibration_not_mutated", float(calibration == calibration_before), 1, lower=True)
    check("deployment_selection_not_mutated", float(selection == selection_before), 1, lower=True)

    # Production uses float32 unless selected otherwise; verify that the true
    # residual criterion and per-frame batch handling survive the scale change.
    physics32 = common.make_physics("cpu", torch.float32, config=config)
    linear32 = SignedLinearOperator(physics32)
    continuum32 = physics32.components(state.float(), beta.float())[0]
    base32 = physics32.op(continuum32) + linear32.apply_beta(beta.float())
    signed32 = linear32.apply_template(S.float())[None]
    batch_raw32 = torch.cat([base32 + signed32, base32 - signed32])
    batch_continuum32 = continuum32.expand(2, -1, -1, -1).clone()
    batch_result = solve_signed_linear(physics32, batch_raw32, batch_continuum32)
    check("default_float32_two_frames_count", len(batch_result["info"]["frames"]), 2, lower=True)
    for frame in range(2):
        report32 = batch_result["info"]["frames"][frame]
        check(f"default_float32_frame_{frame}_true_residual",
              report32["CG"]["relative_true_residual"], 1e-4)
        check(f"default_float32_frame_{frame}_equilibrated_true_residual",
              report32["CG"]["equilibrated_relative_true_residual"], 1e-4)
        check(f"default_float32_frame_{frame}_accepted", float(report32["accepted"]), 1, lower=True)
        single_result = solve_signed_linear(physics32, batch_raw32[frame:frame + 1],
                                             batch_continuum32[frame:frame + 1])
        check(f"float32_frame_{frame}_batch_matches_single_postblur",
              _relative(batch_result["template_postblur_e"][frame],
                        single_result["template_postblur_e"][0]), 2e-5)
        check(f"float32_frame_{frame}_batch_matches_single_beta",
              _relative(batch_result["beta"][frame], single_result["beta"][0]), 2e-5)

    return {
        "kind": "bounded CPU float64 numerical controls",
        "scientific_256x256_acceptance": False, "device": "cpu", "shape": [size, size],
        "old_physics_config_sha256": common.sha(common.ROOT / "reports/training_v4/run_001/backbone_selection.json"),
        "test_only_changes": {"shape": [size, size], "carrier_bins_xy": config["carrier_bins_xy"]},
        "seed": 510132, "python": sys.version, "torch_version": torch.__version__,
        "checks": checks, "all_passed": all(item["passed"] for item in checks),
        "dense_controls": dense_results, "wall_s": time.perf_counter() - started,
        "default_float32_frames": batch_result["info"]["frames"],
        "source_sha256": {common.relative(common.CODE / name): common.sha(common.CODE / name)
                          for name in ("continuum_solver.py", "signed_linear_solver.py", "reconstruct.py", "tests_adjoint.py")},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    stamp = common.now().replace(":", "").replace(".", "").replace("+", "_")
    output = args.output or common.REPORTS.parent / "core_tests" / f"cpu_float64_{stamp}.json"
    if Path(output).exists():
        raise FileExistsError("Numerical test receipts must use a new path")
    report = run_tests()
    common.write_json(output, report)
    failed = [item["name"] for item in report["checks"] if not item["passed"]]
    print(f"{len(report['checks'])} checks; passed={report['all_passed']}; wall_s={report['wall_s']:.3f}; {output}")
    if failed:
        print("FAILED: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
