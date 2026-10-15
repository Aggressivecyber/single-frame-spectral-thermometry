"""Read-only causal evidence audit. No rendering, inversion, or sealed payload access.

Run with the recorded environment's Python -B. Only a small JSON result is written
beside this script. NumPy is the only nonstandard dependency.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/closeout_001"
HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def rows(name):
    with (REPORT / name).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def aggregate(items):
    count = sum(row["metrics"]["domain_pixels"] for row in items)
    return {
        "rows": len(items),
        "independent_mothers": len({row["mother_id"] for row in items}),
        "color_RMSE_K": math.sqrt(sum(row["metrics"]["temperature_SSE_K2"] for row in items) / count),
        "template_RMSE_e": math.sqrt(sum(row["metrics"]["template_SSE"] for row in items) / count),
        "color_bias_K": sum(row["metrics"]["temperature_error_sum_K"] for row in items) / count,
        "template_bias_e": sum(row["metrics"]["template_error_sum"] for row in items) / count,
    }


def main():
    science = rows("per_case.jsonl")
    development = rows("development_results.jsonl")
    joint = [row for row in development if row["method"].startswith("joint_")]
    thresholds = read("DETECTOR_THRESHOLDS.json")["methods"]
    cfg = json.loads((ROOT / "reports/training_v4/run_001/backbone_selection.json").read_text())["physics_config"]
    sigma = cfg["relay_psf_sigma_px"]
    radius = max(1, math.ceil(5 * sigma))
    axis = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    transfers = []
    for kx, ky in cfg["carrier_bins_xy"]:
        fx, fy = kx / cfg["shape"][1], ky / cfg["shape"][0]
        discrete = float(np.sum(kernel * np.cos(2 * np.pi * (fx * xx + fy * yy))))
        continuous = float(np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2)))
        transfers.append({"carrier": [kx, ky], "core_analytic": continuous,
                          "independent_sampled": discrete, "ratio": discrete / continuous})
    ratio_factor = transfers[-2]["ratio"] / transfers[-1]["ratio"]
    temperatures = np.arange(1200., 4501.)
    spectral = []
    for k in (5, 6):
        center, width = cfg["centers_nm"][k], cfg["widths_nm"][k]
        lam = np.linspace(center - width / 2, center + width / 2, 321) * 1e-9
        photons = 2 * 299792458. / lam[None, :] ** 4 / np.expm1(.01438776877 / (temperatures[:, None] * lam[None, :]))
        spectral.append(np.trapz(photons * cfg["quantum_efficiencies"][k], lam, axis=1))
    color_ratios = spectral[0] / spectral[1]
    measured = np.interp(3100., temperatures, color_ratios) * ratio_factor
    apparent = float(np.interp(measured, color_ratios, temperatures))
    nominal = {}
    for method in ("v3_physics", "old_B4", "B1_lambda100"):
        subset = [r for r in science if r["family"] == "independent" and r["condition"] == "nominal" and r["method"] == method]
        blank = [r for r in subset if r["coefficient_e"] == 0]
        nominal[method] = aggregate(subset)
        nominal[method]["blank_denominator"] = len(blank)
        nominal[method]["blank_over_calibrated_threshold"] = sum(r["metrics"]["matched_dictionary_frame_score"] > thresholds[method]["threshold"] for r in blank)
    linearization = [json.loads(p.read_text()) for p in sorted((REPORT / "linearization").glob("*.json"))]
    uq = read("UNCERTAINTY_RESULT.json")["operational_interval_evaluation"]
    evaluation = [r for r in uq["rows"] if r["role"] == "interval_evaluation"]
    registration = read("SCIENCE_REGISTRATION.json")
    unchanged = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                 for name, digest in registration["source_sha256"].items()}
    inputs = ["SCIENCE_RESULT.json", "SCIENCE_REGISTRATION.json", "per_case.jsonl",
              "development_results.jsonl", "DETECTOR_THRESHOLDS.json", "UNCERTAINTY_RESULT.json"]
    result = {
        "scope": "Existing result aggregation and analytic transfer check; no new full inversions or sealed payload reads",
        "source_files_checked": len(unchanged), "registered_source_unchanged": all(unchanged.values()),
        "input_sha256": {name: hashlib.sha256((REPORT / name).read_bytes()).hexdigest() for name in inputs},
        "science_case_method_rows": len(science), "science_unique_frames": len({r["case_id"] for r in science}),
        "joint": {
            "candidate_count": len({r["method"] for r in joint}), "initializer_outputs": len(joint),
            "unique_raw_inputs": len({r["raw_sha256"] for r in joint}),
            "numerically_certified": sum(r["production_numerical_certified"] for r in joint),
            "base_stops": dict(Counter(r["info"]["base_outer"]["message"] for r in joint)),
            "tighter_stops": dict(Counter(r["info"]["tighter_outer"]["message"] for r in joint)),
            "color_difference_over_0p2K": sum(r["info"]["color_outer_difference_RMS_K"] > .2 for r in joint),
            "template_difference_over_0p17e": sum(r["info"]["template_outer_difference_RMS_e"] > .17 for r in joint),
            "maximum_inner_difference_e": max(r["info"]["template_inner_difference_RMS_e"] for r in joint),
            "time_range_s": [min(r["info"]["wall_s"] for r in joint), max(r["info"]["wall_s"] for r in joint)]},
        "relay": {"sigma_px": sigma, "carrier_transfers": transfers,
                  "700_over_905_response_ratio_factor": ratio_factor,
                  "flat_continuum_true_K": 3100., "apparent_color_K": apparent,
                  "color_bias_K": apparent - 3100.,
                  "scope": "Constant-field carrier algebra; exact sampled kernel versus analytic Gaussian; no full inverse"},
        "independent_nominal": nominal,
        "local_linearization": {"cases": len(linearization), "statuses": dict(Counter(r["status"] for r in linearization)),
                                "largest_step_discrepancy": max(max(r["step_size_relative_discrepancies"]) for r in linearization)},
        "interval_evaluation": {"mothers": len(evaluation), "statistics": uq["statistics"],
                                "pooled_coverage": uq["pooled_evaluation_marginal_coverage"],
                                "mean_width": np.mean([r["mean_interval_width"] for r in evaluation], axis=0).tolist()},
    }
    assert result["registered_source_unchanged"]
    (HERE / "ROOT_DIAGNOSTIC_RECEIPT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"written": str(HERE / "ROOT_DIAGNOSTIC_RECEIPT.json"), "joint": result["joint"],
                      "relay_bias_K": result["relay"]["color_bias_K"], "source_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
