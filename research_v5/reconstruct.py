"""Raw-DN deployment path: estimate continuum, freeze it, solve signed S/beta."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from research_v5 import common

common.configure_runtime()  # E/D temp/cache paths before torch or legacy imports.

import torch

from research_v5.continuum_solver import estimate_continuum
from research_v5.signed_linear_solver import solve_signed_linear


def _calibration_config(calibration):
    if not isinstance(calibration, dict):
        raise TypeError("calibration must be an externally fixed configuration dictionary")
    if "physics_config" in calibration:
        if set(calibration) != {"physics_config"}:
            raise ValueError("calibration wrapper may contain only physics_config")
        config = calibration["physics_config"]
    else:
        config = calibration
    if not isinstance(config, dict):
        raise TypeError("physics_config must be a dictionary")
    forbidden = {"truth", "condition", "clean_identity", "case_id", "input_id", "labels"}
    if set(config) & forbidden:
        raise ValueError("Deployment calibration must not contain truth or input identity")
    return copy.deepcopy(config)


@torch.no_grad()
def reconstruct(raw_dn, calibration, selection):
    """Infer using only raw DN and external calibration/selection parameters.

    Selection keys are continuum_options, linear_options, device and dtype.
    No diagnostic fixed_beta is accepted here.  A failed signed solve marks
    that frame invalid/abstain while preserving its numerical output and logs.
    Uncertainty and empirical detection thresholds are computed by the
    separately registered uncertainty pipeline, not invented by this function.
    """
    config = _calibration_config(calibration)
    if not isinstance(selection, dict):
        raise TypeError("selection must be an external fixed parameter dictionary")
    unknown = set(selection) - {"continuum_options", "linear_options", "device", "dtype"}
    if unknown:
        raise ValueError(f"Unknown deployment selection keys: {sorted(unknown)}")
    device = str(selection.get("device", "cpu"))
    dtype_name = str(selection.get("dtype", "float32"))
    if dtype_name not in ("float32", "float64"):
        raise ValueError("Deployment dtype is float32 or float64")
    physics = common.make_physics(device, getattr(torch, dtype_name), config=config)
    raw_e = common.raw_electrons(physics, raw_dn)
    continuum = estimate_continuum(physics, raw_e, selection.get("continuum_options"))
    # An explicit detached copy makes the boundary auditable: S/beta cannot
    # alter T/loga, and the solver receives no state or temperature truth.
    frozen_continuum = continuum["continuum_latent"].detach().clone()
    linear = solve_signed_linear(physics, raw_e, frozen_continuum,
                                  selection.get("linear_options"), fixed_beta=None)
    solved = raw_e.new_tensor([
        bool(frame["accepted"] and frame["CG"]["converged"]
             and frame["expected_e_nonnegative"])
        for frame in linear["info"]["frames"]], dtype=torch.bool)
    valid = continuum["valid"] & solved[:, None, None]
    continuum_info = continuum.pop("info")
    linear_info = linear.pop("info")
    return {
        **continuum, **linear, "valid": valid, "abstain": ~valid,
        "info": {
            "continuum": continuum_info, "linear": linear_info,
            "continuum_frozen": True,
            "deployment_inputs": ["raw_dn", "external calibration", "external selection"],
            "truth_or_identity_access": False, "diagnostic_fixed_beta_used": False,
            "validity_rule": "postblur color validity and finite accepted converged signed solve with nonnegative raw expectation",
            "temperature_uncertainty_status": "requires registered Monte Carlo; absent here",
            "detection_status": "no null-calibrated threshold supplied by this estimator",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="NPY or NPZ with raw_dn only")
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new outputs/research_v5/*.npz on E/D")
    parser.add_argument("--report", type=Path, help="new JSON receipt; defaults beside output")
    args = parser.parse_args(argv)
    import numpy as np
    if args.raw.suffix.lower() == ".npz":
        raw = common.arrays(args.raw, raw_only=True)["raw_dn"]
    elif args.raw.suffix.lower() == ".npy":
        raw = np.load(args.raw, allow_pickle=False)
    else:
        raise ValueError("Deployment input must be .npy or raw_dn-only .npz")
    calibration = common.read_json(args.calibration)
    selection = common.read_json(args.selection)
    output = common.ed_path(args.output)
    report = common.ed_path(args.report or output.with_suffix(".json"))
    if output.exists() or report.exists():
        raise FileExistsError("Use new output and report paths")
    result = reconstruct(raw, calibration, selection)
    arrays = {name: value.detach().cpu().numpy() for name, value in result.items()
              if isinstance(value, torch.Tensor)}
    output_hash = common.save_npz(output, **arrays)
    receipt = {
        "output": common.relative(output), "output_sha256": output_hash,
        "raw_sha256": common.sha(args.raw),
        "calibration_sha256": common.sha(args.calibration),
        "selection_sha256": common.sha(args.selection),
        "tensor_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "info": result["info"], "continuum_history": result["history"],
        "valid_fraction": float(result["valid"].double().mean()),
    }
    common.write_json(report, receipt)
    print(f"Saved {output}; receipt {report}; valid_fraction={receipt['valid_fraction']:.6f}")


if __name__ == "__main__":
    main()
