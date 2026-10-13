"""Evidence-led v5 reports and strictly computed stage gates.

Small fixture receipts, source registration, numerical execution, and scientific
acceptance remain separate.  This module is stdlib-only until a plot is requested.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Mapping

from .common import (CODE, DATA, REPORTS, ROOT, canonical_hash, ed_path, jsonl,
                     now, read_json, relative, sha, write_json)


LEVELS = ("O0", "O1", "O2", "O3")
PHASES = ("register", "oracle", "identifiability", "linear-solver",
          "development-compare", "train-denoiser", "candidate-review", "final-once")


def write_text(path, value, *, replace=False):
    """Atomic E/D report write, followed by a byte-preserving reopen check."""
    path = ed_path(path)
    if not path.is_relative_to(REPORTS.parent.resolve()):
        raise ValueError("Text reports must stay in reports/research_v5")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError("Completed evidence is immutable: " + str(path))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    if path.read_text(encoding="utf-8") != value:
        raise RuntimeError("Text reopen differs: " + str(path))
    return sha(path)


def write_csv(path, rows):
    """Keep every primitive diagnostic and serialize nested histories as JSON."""
    flattened = []
    for row in rows:
        value = {key: item for key, item in row.items() if key not in ("metrics", "CG")}
        value.update({"metric_" + key: item for key, item in (row.get("metrics") or {}).items()})
        value.update({"CG_" + key: item for key, item in (row.get("CG") or {}).items()})
        flattened.append({key: json.dumps(item, ensure_ascii=False, allow_nan=False)
                          if isinstance(item, (dict, list, tuple)) else item
                          for key, item in value.items()})
    fields = sorted({key for row in flattened for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flattened)
    return write_text(path, stream.getvalue())


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    values = [value for value in values if _finite(value)]
    return sum(values) / len(values) if values else None


def pool_oracle_rows(rows):
    """Pool sums and support counts, never average frame-relative L2 ratios."""
    metrics = [row.get("metrics") for row in rows]
    scored = [m for m in metrics if m is not None and _finite(m.get("template_SSE"))
              and _finite(m.get("truth_energy")) and _finite(m.get("pixels"))]
    complete = len(scored) == len(rows) and bool(rows)
    pixels = sum(m["pixels"] for m in scored)
    sse = sum(m["template_SSE"] for m in scored)
    energy = sum(m["truth_energy"] for m in scored)
    support = sum(m.get("truth_support_pixels", 0) for m in scored)
    recalls_complete = all(_finite(m.get("sign_recall")) for m in scored
                           if m.get("truth_support_pixels", 0) > 0)
    projection_complete = all(_finite(m.get("signed_projection")) for m in scored
                              if m["truth_energy"] > 0)
    recall = (sum(m["sign_recall"] * m["truth_support_pixels"] for m in scored
                  if m.get("truth_support_pixels", 0) > 0) / support
              if support and recalls_complete else None)
    projection = (sum(m["signed_projection"] * m["truth_energy"] for m in scored
                      if m["truth_energy"] > 0) / energy
                  if energy and projection_complete else None)
    return {"frames": len(rows), "scored_frames": len(scored), "metrics_complete": complete,
            "pixels": pixels, "template_SSE_e2": sse if complete else None,
            "truth_energy_e2": energy if complete else None,
            "pooled_MSE_e2": sse / pixels if complete and pixels else None,
            "pooled_RMSE_e": math.sqrt(sse / pixels) if complete and pixels else None,
            "pooled_L2": math.sqrt(sse / energy) if complete and energy else None,
            "pooled_signed_projection": projection if complete else None,
            "support_pixels": support,
            "pooled_support_sign_recall": recall if complete else None,
            "CG_converged_frames": sum((row.get("CG") or {}).get("converged") is True for row in rows),
            "maximum_true_CG_residual": max((row.get("CG") or {}).get("relative_true_residual")
                                            for row in rows
                                            if _finite((row.get("CG") or {}).get("relative_true_residual")))
                                           if any(_finite((r.get("CG") or {}).get("relative_true_residual")) for r in rows) else None,
            "mean_latency_s": _mean([row.get("latency_s") for row in rows]),
            "mean_raw_weighted_MSE": _mean([m.get("raw_weighted_MSE") for m in scored]),
            "mean_valid_coverage": _mean([m.get("valid_coverage") for m in scored])}


def oracle_summary(rows, plan, protocol, *, baseline_rows=(), manifest=None):
    """Require the full registered case/level Cartesian product before any gate."""
    ids = [case["case_id"] for case in plan]
    expected = {(case, level) for case in ids for level in LEVELS}
    actual = [(row.get("case_id"), row.get("level")) for row in rows]
    accounting_complete = len(ids) == 180 and len(set(ids)) == 180 and len(actual) == 720 and set(actual) == expected
    complete = accounting_complete and all(row.get('status') != 'NOT_EXECUTED_BUDGET' for row in rows)
    paired_hash_checks = []
    for case in ids:
        case_rows = [row for row in rows if row.get("case_id") == case]
        measured = [row for row in case_rows if row.get("status") == "RETURNED"]
        for key in ("raw_file_sha256", "label_sha256", "source_descriptor_sha256", "source_sha256"):
            values = [row.get(key) for row in case_rows]
            paired_hash_checks.append(len(values) == 4 and len({canonical_hash(value) for value in values}) == 1
                                      and all(value is not None for value in values)
                                      if key != "raw_file_sha256" or measured else True)
        for key in ("raw_dn_sha256", "raw_e_sha256"):
            if measured:
                paired_hash_checks.append(len({row.get(key) for row in measured}) == 1
                                          and all(row.get(key) is not None for row in measured))
        weights = [(r.get("metrics") or {}).get("raw_weight_sha256") for r in measured]
        if weights:
            paired_hash_checks.append(len(set(weights)) == 1 and all(weights))
    same_raw_audit = complete and all(paired_hash_checks)
    settings = protocol["oracle_noiseless_gate"]
    required_ids = {case["case_id"] for case in plan
                    if case["input_kind"] == settings["input_kind"]
                    and case["coefficient_e"] in settings["coefficients_e"]}
    selected = [row for row in rows if row.get("level") == "O0" and row.get("case_id") in required_ids]
    pooled = pool_oracle_rows(selected)
    cg_tol = protocol["CG"]["true_rtol"]
    converged = bool(selected) and all((row.get("CG") or {}).get("converged") is True
                                     and _finite((row.get("CG") or {}).get("relative_true_residual"))
                                     and row["CG"]["relative_true_residual"] <= cg_tol
                                     and (row.get("CG") or {}).get("accepted") is True for row in selected)
    checks = {
        "complete_180_case_720_level_accounting": complete,
        "same_raw_source_label_and_weight_hashes": same_raw_audit,
        "exact_24_continuous_normal_inputs": len(required_ids) == 24 and len(selected) == 24,
        "all_selected_metrics_available": pooled["metrics_complete"],
        "pooled_RMSE_e": _finite(pooled["pooled_RMSE_e"]) and pooled["pooled_RMSE_e"] <= settings["pooled_RMSE_e_max"],
        "pooled_L2": _finite(pooled["pooled_L2"]) and pooled["pooled_L2"] <= settings["pooled_L2_max"],
        "pooled_support_sign_recall": _finite(pooled["pooled_support_sign_recall"])
                                     and pooled["pooled_support_sign_recall"] >= settings["support_sign_recall_min"],
        "all_CG_converged_at_true_residual_tolerance": converged,
    }
    o0_pass = all(checks.values())
    screen = protocol["oracle_crosstalk_screen"]
    crosstalk = {}
    for level in LEVELS[1:]:
        other = pool_oracle_rows([row for row in rows if row.get("level") == level
                                 and row.get("case_id") in required_ids])
        mse0, mse = pooled["pooled_MSE_e2"], other["pooled_MSE_e2"]
        worse = (_finite(mse0) and _finite(mse) and mse > screen["pooled_MSE_factor_over_O0"] * mse0
                 + screen["absolute_MSE_allowance_e2"])
        collapsed = (not other["metrics_complete"] or not _finite(other["pooled_RMSE_e"])
                     or other["pooled_RMSE_e"] > screen["collapsed_noiseless_normal_RMSE_e"])
        crosstalk[level] = {"metrics": other, "worse_than_O0": worse, "strict_collapse": collapsed,
                            "screen_threshold_MSE_e2": screen["pooled_MSE_factor_over_O0"] * mse0
                                                      + screen["absolute_MSE_allowance_e2"] if _finite(mse0) else None}
    groups = []
    for level in LEVELS:
        for kind in ("continuous", "quantization", "noise"):
            for coefficient in (0., 45., -45., 180., -180.):
                values = [row for row in rows if row.get("level") == level and row.get("input_kind") == kind
                          and row.get("coefficient_e") == coefficient]
                groups.append({"level": level, "input_kind": kind, "coefficient_e": coefficient,
                               **pool_oracle_rows(values)})
    baseline_ids = {case["case_id"] for case in plan if case["input_kind"] == "noise"
                    or case["input_kind"] == "continuous" and abs(case["coefficient_e"]) == 180.}
    baseline_names = protocol["baselines"]
    baseline_pairs = [(row.get("case_id"), row.get("method")) for row in baseline_rows]
    baseline_complete = (len(baseline_pairs) == len(baseline_ids) * len(baseline_names)
                         and set(baseline_pairs) == {(case, method) for case in baseline_ids for method in baseline_names}
                         and all(row.get('status') != 'NOT_EXECUTED_BUDGET' for row in baseline_rows))
    return {"schema": "v5-oracle-ladder-evidence-001", "time": now(), "complete": complete,
            "full_planned_record_accounting": accounting_complete,
            "case_counts": {"planned_inputs": len(plan), "distinct_mothers": len({c["mother_id"] for c in plan}),
                            "oracle_rows": len(rows), "oracle_returned_rows": sum(r.get("status") == "RETURNED" for r in rows),
                            "oracle_evaluated_inputs": len({r['case_id'] for r in rows if r.get('status') == 'RETURNED'}),
                            "oracle_not_executed_rows": sum(r.get('status') == 'NOT_EXECUTED_BUDGET' for r in rows),
                            "baseline_planned_inputs": len(baseline_ids), "baseline_rows": len(baseline_rows)},
            "manifest_generated_raw_rows": (manifest or {}).get("generated_raw_rows"),
            "manifest_unavailable_rows": (manifest or {}).get("unavailable_rows"),
            "same_raw_label_source_and_weights_audit": same_raw_audit,
            "o0_noiseless_gate": {"passed": o0_pass, "criteria": settings, "metrics": pooled,
                                  "checks": checks, "failed_checks": [key for key, value in checks.items() if not value]},
            "crosstalk": {"criteria": screen, "levels": crosstalk,
                          "scope": "registered descriptive screen, not universal causal identification"},
            "ml_stop_required": not o0_pass or any(crosstalk[level]["strict_collapse"] for level in ("O1", "O2")),
            "group_metrics": groups, "baseline_subset_accounting_complete": baseline_complete,
            "baseline_subset_scope": "60 noisy + 24 normal continuous inputs; other 96 cases excluded by registration",
            "fixture_only": False, "scientific_final_acceptance": False,
            "final_or_confirmation_payloads_opened": False}


def _display(value):
    return "unavailable" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def write_oracle_reports(summary, rows, baseline_rows):
    write_json(REPORTS / "ORACLE_SUMMARY.json", summary)
    write_csv(REPORTS / "ORACLE_LADDER.csv", rows)
    write_csv(REPORTS / "ORACLE_BASELINES_84.csv", baseline_rows)
    gate = summary["o0_noiseless_gate"]
    lines = ["# v5 oracle ladder", "", f"Accounting complete: **{summary['complete']}**. "
             f"O0 continuous ±180 e gate: **{gate['passed']}**. ML stop required: **{summary['ml_stop_required']}**.", "",
             "180 registered inputs from 12 opened development mothers; four diagnostic levels share each raw frame. "
             "O0/O1/O2 use explicitly labelled diagnostic truth; O3 inference receives only raw and fixed calibration.", "",
             "| Level | Input | Coefficient e | Scored/frames | Pooled RMSE e | Pooled L2 | Signed projection | Sign recall | CG converged |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in summary["group_metrics"]:
        lines.append("| " + " | ".join(_display(group.get(key)) for key in
                     ("level", "input_kind", "coefficient_e")) + " | "
                     + f"{group['scored_frames']}/{group['frames']} | "
                     + " | ".join(_display(group.get(key)) for key in
                       ("pooled_RMSE_e", "pooled_L2", "pooled_signed_projection", "pooled_support_sign_recall", "CG_converged_frames")) + " |")
    lines += ["", "Failed gate checks: " + (", ".join(gate["failed_checks"]) or "none"), "",
              "Baseline scope: " + summary["baseline_subset_scope"] + ". "
              f"Subset accounting complete: {summary['baseline_subset_accounting_complete']}.", "",
              "ORACLE_LADDER.jsonl retains every source/raw/label hash, actual true CG residual, histories, latency, "
              "postblur errors, signed projection, support recall, raw reprojection, and unavailable/error record. "
              "Unconverged solves and unavailable cases remain in the denominator of execution accounting. "
              "Fixed-60 e detection is a diagnostic; independent null-calibrated detection is assessed later.", "",
              "These are finite synthetic diagnostics; negative coefficients are template deviations, not negative concentrations. "
              "No sealed final or signed-confirmation array was opened.", ""]
    write_text(REPORTS / "ORACLE_LADDER.md", "\n".join(lines))
    _oracle_plot(summary)


def _oracle_plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = ["#264c88", "#d88721", "#36a477"]
    for kind, color in zip(("continuous", "quantization", "noise"), colors):
        for index, level in enumerate(LEVELS):
            groups = [g for g in summary["group_metrics"] if g["level"] == level
                      and g["input_kind"] == kind and abs(g["coefficient_e"]) == 180.]
            mse = _mean([g["pooled_MSE_e2"] for g in groups]) if len(groups) == 2 and all(g['metrics_complete'] for g in groups) else None
            axes[0].plot([index], [math.sqrt(mse) if mse is not None else float("nan")],
                         marker="o", color=color, label=kind if index == 0 else None)
            projection = _mean([g["pooled_signed_projection"] for g in groups]) if mse is not None else float('nan')
            axes[1].plot([index], [projection],
                         marker="o", color=color, label=kind if index == 0 else None)
    for axis in axes:
        axis.set_xticks(range(4), LEVELS)
        axis.grid(alpha=.2)
    axes[0].set(ylabel="Normal +/- pooled RMSE (e)", title="Oracle ladder error")
    axes[1].set(ylabel="Signed projection", title="Retained normal signal")
    axes[0].legend()
    path = ed_path(REPORTS / "ORACLE_LADDER.png")
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def information_summary(rows, optical, plan):
    required = {case["case_id"] for case in plan if case["input_kind"] == "noise"
                and abs(case["coefficient_e"]) == 180.}
    actual = [row.get("case_id") for row in rows]
    complete = len(required) == 24 and len(actual) == 24 and set(actual) == required and all(r.get('status') == 'RETURNED' for r in rows)
    returned = [row for row in rows if row.get("status") == "RETURNED"]
    valid_checks = (len(returned) == 24 and all(frame["adjoint_all_passed"]
                    and frame["float64_finite_difference"]["all_checked_passed"]
                    for row in returned for frame in row["fisher"]["frames"]))
    mode_rows = []
    for row in returned:
        for frame in row["fisher"]["frames"]:
            for nuisance, stats in frame["nuisance_projected_S"].items():
                mode_rows.append({"case_id": row["case_id"], "mother_id": row["mother_id"],
                    "coefficient_e": row["coefficient_e"], "nuisance": nuisance,
                    "rank": stats["rank"], "columns": stats["columns"],
                    "minimum_singular_value": stats["minimum_singular_value"],
                    "condition_number": stats["condition_number"],
                    "condition_number_is_infinite": stats["condition_number_is_infinite"],
                    "sigma_e": stats["finite_modal_linearized_sigma_e"],
                    "single_mode_detection_SNR": stats["single_mode_detection_SNR"],
                    "joint_modal_coefficient_SNR": stats["joint_modal_coefficient_SNR"],
                    "retained_information_fraction_per_mode": stats["retained_information_fraction_per_mode"]})
    return {"schema": "v5-stage1-information-evidence-001", "time": now(), "complete": complete,
            "planned_linearizations": 24, "recorded_linearizations": len(rows),
            "returned_linearizations": len(returned), "numerical_checks_passed": valid_checks,
            "optical_all_passed": optical.get("all_passed") is True,
            "finite_modes_only": True, "not_full_image_exact_CRLB": True,
            "same_oracle_noise_inputs": True, "mode_statistics": mode_rows,
            "fixture_only": False, "scientific_final_acceptance": False}


def write_information_reports(summary, rows, optical):
    write_json(REPORTS / "INFORMATION_SUMMARY.json", summary)
    write_json(REPORTS / "FISHER_IDENTIFIABILITY.json", {"schema": "v5-fisher-all-24-001", "rows": rows})
    write_json(REPORTS / "OPTICAL_NULLSPACE_TESTS.json", optical)
    lines = ["# Finite-mode Fisher identifiability", "",
             f"Linearizations returned: {summary['returned_linearizations']}/24. "
             f"Complete case accounting: {summary['complete']}. Adjoint/float64 finite-difference checks: {summary['numerical_checks_passed']}.", "",
             "Each of the 12 diagnostic mothers contributes its +180 e and -180 e noisy frame from the oracle data. "
             "The original observed raw fixes W at the diagnostic true state. T, loga, S and beta columns use "
             "physical units and representative DC, Fourier and bandlimited ROI modes. Each nuisance span is "
             "projected independently and in combinations.", "",
             "| Case | Nuisance | S rank/modes | Smallest singular value | Condition |",
             "|---|---|---:|---:|---:|"]
    for row in summary["mode_statistics"]:
        if row["nuisance"] in ("none", "T_only", "loga_only", "beta_only", "T_loga_beta"):
            condition = "infinite" if row["condition_number_is_infinite"] else _display(row["condition_number"])
            lines.append(f"| {row['case_id']} | {row['nuisance']} | {row['rank']}/{row['columns']} | "
                         f"{_display(row['minimum_singular_value'])} | {condition} |")
    lines += ["", "Numeric JSON retains weighted/unweighted adjoint checks, finite-difference reference shifts, "
              "singular values, ranks, condition numbers, retained information, single-mode and joint-modal SNR, "
              "and finite-modal sigma in electrons. A null direction has unbounded variance; it receives no "
              "finite pseudoinverse confidence interval.", "",
              "Scope: local finite representative modes under fixed synthetic calibration and variance. "
              "Finite nuisance spaces can overstate information relative to unrestricted fields. These are "
              "not exact full-image 256×256 CRLBs, physical-instrument detection limits, or empirical uncertainty coverage.", ""]
    write_text(REPORTS / "FISHER_IDENTIFIABILITY.md", "\n".join(lines))
    high = optical.get("high_frequency", {})
    write_text(REPORTS / "OPTICAL_NULLSPACE_TESTS.md", "\n".join([
        "# Optical nullspace and carrier tests", "", f"All tests passed: **{optical.get('all_passed', False)}**.", "",
        f"Sampled exact-zero OTF bins: {_display(optical.get('sampled_zero_OTF_bins'))}. "
        f"High-frequency object transfer norm: {_display(high.get('object_output_to_input_norm'))}; "
        f"complete raw transfer norm: {_display(high.get('complete_raw_output_to_input_norm'))}; "
        f"encoding without object blur norm: {_display(high.get('encode_without_object_blur_output_to_input_norm'))}.", "",
        f"Forward/adjoint relative error: {_display(optical.get('adjoint', {}).get('relative_error'))}. "
        "Numeric JSON retains the low-frequency transmission and cross-carrier isolation for all seven bands.", "",
        "Carrier modulation after the object OTF cannot recover pre-OTF zero-space information. "
        "Primary reconstruction targets remain postblur continuum color and signed template.", ""]))
    _information_plot(summary, rows)


def _information_plot(summary, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [row for row in summary["mode_statistics"] if row["nuisance"] == "T_loga_beta"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for index, row in enumerate(selected):
        snr = row["single_mode_detection_SNR"].get("45.0", [])
        axes[0].scatter([index] * len(snr), snr, s=10, alpha=.6)
        retained = row["retained_information_fraction_per_mode"]
        axes[1].scatter([index] * len(retained), retained, s=10, alpha=.6)
    axes[0].axhline(3, color="#b12929", linestyle="--", label="Registered modal SNR screen = 3")
    axes[0].set(ylabel="45 e single-mode SNR", xlabel="Diagnostic normal noise linearization",
                title="After finite T/loga/beta projection")
    axes[1].set(ylabel="Retained information fraction", xlabel="Diagnostic normal noise linearization",
                title="Finite-mode nuisance loss", ylim=(0, 1.05))
    for axis in axes:
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    path = ed_path(REPORTS / "FISHER_IDENTIFIABILITY.png")
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_decision_1(protocol):
    oracle = _optional("ORACLE_SUMMARY.json")
    information = _optional("INFORMATION_SUMMARY.json")
    o0 = oracle.get("o0_noiseless_gate", {}).get("passed") is True
    info_pass = information.get("numerical_checks_passed") is True and information.get("optical_all_passed") is True
    complete = oracle.get("complete") is True and information.get("complete") is True
    reasons = []
    if not complete:
        reasons.append("full oracle/information evidence is incomplete")
    if not o0:
        reasons.append("registered O0 continuous +/-180 e numerical recovery gate failed or unavailable")
    if not info_pass:
        reasons.append("finite-mode derivative or optical checks failed or unavailable")
    for level in ("O1", "O2"):
        if oracle.get("crosstalk", {}).get("levels", {}).get(level, {}).get("strict_collapse") is True:
            reasons.append(level + " continuum/background cross-talk collapsed at the registered screen")
    result = {"schema": "v5-decision-1-001", "time": now(), "stage1_evidence_complete": complete,
              "o0_noiseless_passed": o0, "information_numerical_checks_passed": info_pass,
              "physical_stage2_eligible": complete and o0 and info_pass,
              "ml_stop_required": bool(reasons), "reasons": reasons,
              "cross_talk_causal_claim": "descriptive within the fixed finite synthetic diagnostic",
              "training_authorized_by_this_receipt": False,
              "sealed_final_or_confirmation_opened": False,
              "evidence_sha256": {name: sha(REPORTS / name) for name in
                                  ("ORACLE_SUMMARY.json", "INFORMATION_SUMMARY.json") if (REPORTS / name).exists()}}
    write_json(REPORTS / "DECISION_1.json", result, replace=True)
    gate = oracle.get("o0_noiseless_gate", {})
    text = ["# v5 stage 1 decision", "", f"Stage 1 evidence complete: **{complete}**. "
            f"Physical stage 2 eligible: **{result['physical_stage2_eligible']}**. ML stop required: **{result['ml_stop_required']}**.", "",
            "O0 normal continuous pooled results: " + ", ".join(key + "=" + _display(gate.get("metrics", {}).get(key))
             for key in ("pooled_RMSE_e", "pooled_L2", "pooled_support_sign_recall", "maximum_true_CG_residual")) + ".", ""]
    for level, screen in oracle.get("crosstalk", {}).get("levels", {}).items():
        text.append(f"{level}: pooled normal continuous RMSE={_display(screen['metrics'].get('pooled_RMSE_e'))} e; "
                    f"worse-than-O0 screen={screen['worse_than_O0']}; strict collapse={screen['strict_collapse']}.")
    text += ["", "Stop/restriction reasons: " + ("; ".join(reasons) or "none from stage 1; development eligibility still required for training"), "",
             "If O0 fails, stop ML and inspect numerical inversion, forward/adjoint and optical information. "
             "If O0 passes and O1/O2 collapse, quantify continuum calibration or structured background identifiability. "
             "If only noisy weak modes have low observable SNR, report a detection limit and abstention; weak L2<1 "
             "is not required below that limit. Actual modal results are in FISHER_IDENTIFIABILITY.json.", "",
             "This decision does not open final/confirmation data, establish final scientific acceptance, "
             "or support true gas-temperature/species-concentration/10 kHz experimental claims.", ""]
    write_text(REPORTS / "DECISION_1.md", "\n".join(text), replace=True)
    return result


def _optional(name):
    path = REPORTS / name
    return read_json(path) if path.exists() else {}


def write_train_report(summary):
    """One completed/negative/skipped neural-control report, bound to its JSON."""
    source = REPORTS / 'TRAIN_SUMMARY.json'
    if not source.is_file():
        raise FileNotFoundError('Training report requires a real saved summary')
    assert read_json(source) == summary, 'Training prose must describe the saved result'
    digest = sha(source)
    path = REPORTS / 'TRAIN_V5_REPORT.md'
    if path.exists():
        assert digest in path.read_text(encoding='utf-8'), 'Saved training report describes another summary'
        return {'path': relative(path), 'sha256': sha(path)}
    lines = ['# Single bounded neural control', '',
             f"Status: **{summary.get('status')}**. Eligible: **{summary.get('eligible', False)}**. "
             f"Accepted: **{summary.get('accepted', False)}**. Actual optimizer updates: **{summary.get('actual_optimizer_updates', 0)}**.", '',
             'Exactly one preregistered width-8 SignedBoundedGainDenoiser multiplies the physical signed field by '
             '`1 + 0.3*tanh(g_theta(z))`. Zero field stays zero; gain is positive and sign is preserved. '
             'The last head starts at zero. Continuum parameters are frozen, beta is handled by the physical solve, '
             'and truth/condition/case identity are excluded from deployment features.', '',
             'One Adam learning rate 3e-4, seed 53131, at most 1000 real updates; snapshots at 0/200/500/1000. '
             'Fresh 36 training mothers cover all 6×6 temperature/template geometry pairs. '
             'Checkpoint acceptance requires both normal signs, detectable weak improvement and unchanged independent null threshold.', '',
             '| Update | Accepted | Weights changed | Detectable weak mother MSE e² |',
             '|---:|---|---|---:|']
    for checkpoint in summary.get('checkpoints', []):
        lines.append('| ' + ' | '.join(_display(checkpoint.get(key)) for key in
                     ('updates', 'accepted', 'weights_changed_from_initial',
                      'selection_objective_detectable_weak_mother_MSE_e2')) + ' |')
    lines += ['', 'Skip reason: ' + str(summary.get('skip_reason') or 'none') + '.', '',
              'Failures retained: ' + json.dumps(summary.get('failures', []), ensure_ascii=False, allow_nan=False) + '.', '',
              'Selected checkpoint: ' + json.dumps(summary.get('selected_checkpoint'), ensure_ascii=False, allow_nan=False) + '.', '',
              'An unsuccessful or inapplicable neural control keeps the eligible physics plus uncertainty route. '
              'Checkpoint execution is not final scientific acceptance, and final/confirmation data are never used for selection.', '',
              'Saved summary SHA256: ' + digest + '.', '']
    write_text(path, '\n'.join(lines))
    return {'path': relative(path), 'sha256': sha(path)}


def final_evaluation_receipt(summary, protocol):
    """Verify completed reserved evaluation without reading any array payload."""
    failed = {'completed': False, 'scientific_acceptance_passed': False,
              'numerical_execution_passed': False, 'reason': 'no hash-bound completed evaluation receipt'}
    if summary.get('status') not in ('FINAL_SCIENTIFIC_ACCEPTANCE_PASSED', 'FINAL_FAILED_NO_RETUNING'):
        return failed
    receipt_path = REPORTS / 'FINAL_ONCE_RECEIPT.json'
    summary_path = REPORTS / 'FINAL_ONCE_SUMMARY.json'
    journal_path = REPORTS / 'final_once/records.jsonl'
    plans_path = CODE / 'family_plans.json'
    if not all(path.is_file() for path in (receipt_path, summary_path, journal_path, plans_path)):
        return failed
    try:
        receipt = read_json(receipt_path)
        counts = protocol.get('final_once', {}).get('case_counts', {})
        expected_count = counts.get('total')
        if (not isinstance(expected_count, int) or expected_count != 800
                or receipt.get('status') != 'FINAL_ONCE_COMPLETED_NO_RETUNING'
                or receipt.get('summary_sha256') != sha(summary_path)
                or receipt.get('summary_path') != relative(summary_path)
                or receipt.get('case_counts') != counts
                or summary.get('case_counts') != counts
                or receipt.get('evaluated_candidate_cases') != expected_count
                or summary.get('evaluated_candidate_cases') != expected_count
                or receipt.get('records_sha256') != summary.get('records_sha256')
                or receipt.get('records_sha256') != sha(journal_path)
                or sha(plans_path) != protocol.get('family_plans_sha256')):
            return failed
        plans = read_json(plans_path)['plans']
        required_ids = {row['case_id'] for name in ('signed_confirmation', 'final_once') for row in plans[name]}
        candidate_rows = []
        with journal_path.open(encoding='utf-8-sig') as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get('method') == summary.get('candidate_kind'):
                        candidate_rows.append(row.get('case_id'))
        if (len(required_ids) != expected_count or len(candidate_rows) != expected_count
                or set(candidate_rows) != required_ids):
            return failed
        numerical = summary.get('numerical_execution_passed') is True and receipt.get('numerical_execution_passed') is True
        scientific = (numerical and summary.get('scientific_acceptance_passed') is True
                      and receipt.get('scientific_acceptance_passed') is True
                      and bool(summary.get('gates')) and all(summary['gates'].values()))
        return {'completed': True, 'scientific_acceptance_passed': scientific,
                'numerical_execution_passed': numerical, 'reason': None,
                'receipt_sha256': sha(receipt_path), 'summary_sha256': sha(summary_path),
                'records_sha256': sha(journal_path), 'verified_candidate_cases': expected_count}
    except (KeyError, TypeError, ValueError, OSError) as error:
        return {**failed, 'reason': 'completed evaluation receipt could not be verified: ' + str(error)}


def write_scientific_decision(protocol=None):
    """Refresh the single-page audit from receipts that actually exist."""
    protocol = protocol or (read_json(CODE / "protocol.json") if (CODE / "protocol.json").exists() else {})
    oracle = _optional("ORACLE_SUMMARY.json")
    info = _optional("INFORMATION_SUMMARY.json")
    stage1 = _optional("DECISION_1.json")
    dev = _optional("development/development_gate.json")
    train = _optional("TRAIN_SUMMARY.json")
    candidate = _optional("candidate_review/candidate_receipt.json")
    final = _optional("FINAL_ONCE_SUMMARY.json")
    started = (REPORTS / "FINAL_ONCE_STARTED.json").exists()
    gate = oracle.get("o0_noiseless_gate", {})
    m = gate.get("metrics", {})
    final_check = final_evaluation_receipt(final, protocol)
    final_state = ("completed: " + str(final.get("status", "unknown")) if final_check['completed']
                   else 'attempt failed without completed cohort receipt: ' + str(final.get('status'))
                   if started and final else "started, no completed evaluation receipt" if started
                   else "sealed evaluation not opened")
    final_accepted = final_check['scientific_acceptance_passed']
    color = ("Nominal postblur color RMSE=" + _display(dev.get("nominal_pooled", {}).get("temperature_RMSE_K"))
             + " K; acceptance=" + _display(dev.get("gates", {}).get("nominal_color"))
             + "; full-domain valid coverage=" + _display(dev.get("nominal_pooled", {}).get("valid_fraction"))
             if dev else "Development nominal color-temperature validation has not been completed.")
    if final_check['completed']:
        color = ('Independent sealed nominal postblur color RMSE='
                 + _display(final.get('candidate_pooled', {}).get('temperature_RMSE_K'))
                 + ' K; valid coverage=' + _display(final.get('candidate_pooled', {}).get('valid_fraction'))
                 + '; final acceptance=' + str(final_accepted) + '.')
    signal = (f"O0 normal continuous pooled RMSE {_display(m.get('pooled_RMSE_e'))} e, "
              f"L2 {_display(m.get('pooled_L2'))}, support sign recall {_display(m.get('pooled_support_sign_recall'))}; "
              f"registered gate passed={gate.get('passed', False)}." if oracle
              else "The complete 180-input oracle ladder has not been completed.")
    crosstalk = oracle.get("crosstalk", {}).get("levels", {})
    cause = ("; ".join(level + ": worse=" + str(value.get("worse_than_O0"))
                       + ", collapse=" + str(value.get("strict_collapse")) for level, value in crosstalk.items())
             if crosstalk else "Continuum/background/combined diagnostic attribution is pending.")
    neural = ("Bounded neural control: " + str(train.get("status", "recorded")) + "; accepted="
              + str(train.get("accepted", False)) + "; actual optimizer updates="
              + str(train.get("actual_optimizer_updates", 0)) + "; " + str(train.get("skip_reason") or "see checkpoint evidence")
              if train else "No completed bounded neural-control receipt; stage 1 and physical development gates are required.")
    uncertainty = ("Nominal parent-mean MC coverage=" + _display(dev.get("MCcoverage"))
                   + "; disjoint-null frame FP=" + _display(dev.get("null_heldout_frame_FP"))
                   + "; physical eligibility=" + str(dev.get("physics_eligible", False)) + "."
                   if dev else "MC nominal coverage and independent null detection/rejection are not yet established.")
    if final_check['completed']:
        uncertainty = ('Final null frame FP=' + _display(final.get('final_null', {}).get('frame_FP'))
                       + '; fixed 12-case MC coverage='
                       + _display(final.get('fixed_final_MC', {}).get('signed_interval_full_domain_mother_coverage_CI95', {}).get('estimate'))
                       + '; below-limit weak cases='
                       + str(final.get('weak_detectability', {}).get('below_frozen_limit', {}).get('cases', 'unavailable')) + '.')
    reasons = stage1.get("reasons", [])
    if dev and dev.get('physics_eligible') is False:
        reasons.append('Real physical development acceptance failed: ' + ', '.join(dev.get('failed_gates', [])))
    if train.get("skip_reason"):
        reasons.append(str(train["skip_reason"]))
    if candidate and candidate.get('eligible') is False:
        reasons.append('Final candidate ineligible: ' + str(candidate.get('reason', candidate.get('status'))))
    if final.get("failures"):
        reasons.append("Final receipt contains failed acceptance checks; retuning remains prohibited.")
    lines = ["# V5 scientific decision", "", f"Audit updated {now()}. Final scientific acceptance: **{final_accepted}**. "
             f"Final state: **{final_state}**.", "",
             "| Scientific question | Evidence-bounded answer |", "|---|---|",
             "| 1. Is continuum color temperature retained? | " + color + " |",
             "| 2. Are ±S observable under this optical budget? | " + signal + " |",
             "| 3. Is the bottleneck T/a, beta, or their combination? | " + cause + " |",
             "| 4. Is a CNN needed? | " + neural + " |",
             "| 5. What can one frame estimate or reject? | " + uncertainty + " |",
             "| 6. Do current results support experimental temperature/species claims? | No. Fixed synthetic calibration, "
             "postblur continuum color and synthetic signed template deviation do not establish true gas temperature, "
             "concentration, or 10 kHz instrument performance. |", "",
             "**Execution and scope.** " + ("Source protocol is frozen. " if protocol else "Source registration is incomplete. ")
             + f"Oracle case/level accounting={oracle.get('complete', False)} "
             + f"({oracle.get('case_counts', {}).get('oracle_rows', 0)}/720); information="
             + f"{info.get('returned_linearizations', 0)}/24. Fisher scope is local finite DC/Fourier/ROI modes, "
             "not an exact full-image CRLB. O0/O1/O2 remain diagnostic; deployment uses raw DN and external calibration only.", "",
             "**Stopped, skipped, or incomplete stages.** " + ("; ".join(reasons) if reasons else
             "No completed failure receipt establishes an additional stop; missing later receipts remain incomplete, not accepted.")
             + " The candidate/statistics freeze alone and FINAL_ONCE_STARTED.json alone do not establish completed confirmation. "
             "Independent final acceptance requires the completed FINAL_ONCE_RECEIPT.json bound to summary and all 800 candidate case records.", "",
             "**Detection boundary.** Below identifiable signal SNR, report undetectable/low-confidence and the independently "
             "evaluated simulated null false-positive rate. Do not force every weak frame below L2=1. "
             "Calibration/emissivity mismatch, finite-probe Hessian estimates, eight-repeat MC noise and distribution scope "
             "remain explicit limits. No retuning of a failed sealed evaluation is allowed.", ""]
    path = REPORTS / "V5_SCIENTIFIC_DECISION.md"
    write_text(path, "\n".join(lines), replace=True)
    write_completion_audit(protocol)
    return path


def write_completion_audit(protocol=None):
    """Account for every task stage by actual evidence or an applicable stop.

    A stopped research route can be fully audited without pretending later
    experiments ran. Missing earlier evidence, execution budgets, and interrupted
    sealed evaluation are incomplete, not scientifically approved skips.
    """
    protocol = protocol or {}
    decision = _optional('DECISION_1.json')
    oracle = _optional('ORACLE_SUMMARY.json')
    information = _optional('INFORMATION_SUMMARY.json')
    development = _optional('development/development_gate.json')
    training = _optional('TRAIN_SUMMARY.json')
    final = _optional('FINAL_ONCE_SUMMARY.json')
    final_check = final_evaluation_receipt(final, protocol)
    candidate = _optional('candidate_review/candidate_receipt.json')
    phase_stop = (decision.get('stage1_evidence_complete') is True
                  and decision.get('physical_stage2_eligible') is False)
    ml_stop = decision.get('stage1_evidence_complete') is True and decision.get('ml_stop_required') is True
    dev_stop = bool(development) and development.get('physics_eligible') is False
    required = {
        'register': [CODE / 'protocol.json', CODE / 'diagnostic_plan.json', CODE / 'family_plans.json',
                     REPORTS / 'REGISTRATION_RECEIPT.json', REPORTS / 'SOURCE_REGISTRATION_CHECKS.json',
                     REPORTS / 'baseline_readonly_inventory.json', REPORTS / 'old_array_readonly_inventory.json'],
        'oracle': [REPORTS / name for name in ('ORACLE_LADDER.csv', 'ORACLE_LADDER.jsonl',
                   'ORACLE_SUMMARY.json', 'ORACLE_BASELINES_84.csv', 'ORACLE_BASELINES_84.jsonl', 'ORACLE_LADDER.png')],
        'identifiability': [REPORTS / name for name in ('FISHER_IDENTIFIABILITY.md', 'FISHER_IDENTIFIABILITY.json',
                    'FISHER_IDENTIFIABILITY.png', 'OPTICAL_NULLSPACE_TESTS.md', 'OPTICAL_NULLSPACE_TESTS.json',
                    'INFORMATION_SUMMARY.json', 'DECISION_1.md', 'DECISION_1.json')],
        'linear-solver': [REPORTS / 'LINEAR_SOLVER_CHECKS.json'],
        'development-compare': [REPORTS / 'development' / name for name in ('PHYSICS_ONLY_V5_REPORT.md',
                    'development_gate.json', 'selection.json', 'null_threshold.json', 'mc_coverage.json',
                    'hessian_diagnostics.json', 'paired_statistics.json', 'records.jsonl')],
        'train-denoiser': [REPORTS / 'TRAIN_SUMMARY.json', REPORTS / 'TRAIN_V5_REPORT.md'],
        'candidate-review': [REPORTS / 'candidate_review/candidate_receipt.json', REPORTS / 'final_protocol.json'],
        'final-once': [REPORTS / 'FINAL_ONCE_SUMMARY.json', REPORTS / 'FINAL_ONCE_RECEIPT.json',
                       REPORTS / 'final_once/records.jsonl'],
    }
    events_path = REPORTS / 'PHASE_EVENTS.jsonl'
    events = jsonl(events_path) if events_path.exists() else []
    stages = []
    for phase, artifacts in required.items():
        items = [{'path': relative(path), 'exists': path.is_file(),
                  'sha256': sha(path) if path.is_file() else None} for path in artifacts]
        all_present = all(item['exists'] for item in items)
        phase_events = [event for event in events if event.get('phase') == phase]
        numerical_pass = next((event.get('numerical_execution_passed') for event in reversed(phase_events)
                               if event.get('status') != 'STARTED'), None)
        completed = all_present
        if phase == 'register':
            completed = all_present and protocol.get('source_QA_passed_before_freeze') is True
        elif phase == 'oracle':
            completed = all_present and oracle.get('complete') is True and oracle.get('case_counts', {}).get('oracle_rows') == 720
            # A limited comparator budget is an explicit incomplete comparison,
            # even when all four diagnostic oracle levels finished.
            completed = completed and oracle.get('baseline_subset_accounting_complete') is True
        elif phase == 'identifiability':
            completed = all_present and information.get('complete') is True and information.get('returned_linearizations') == 24
        elif phase == 'final-once':
            completed = all_present and final_check['completed']
        skip = None
        if not completed and phase in ('linear-solver', 'development-compare', 'train-denoiser', 'candidate-review', 'final-once'):
            if phase_stop:
                skip = 'registered stage 1 stop: ' + '; '.join(decision.get('reasons', []))
            elif phase == 'train-denoiser' and ml_stop:
                skip = 'registered stage 1 ML stop: ' + '; '.join(decision.get('reasons', []))
            elif phase in ('train-denoiser', 'candidate-review', 'final-once') and dev_stop:
                skip = 'real physical development gates failed: ' + ', '.join(development.get('failed_gates', []))
            elif phase == 'train-denoiser' and training.get('skip_reason'):
                skip = 'bounded neural control skipped: ' + str(training['skip_reason'])
        # A STARTED sealed run cannot be explained away as an unopened skip.
        if phase == 'final-once' and (REPORTS / 'FINAL_ONCE_STARTED.json').exists() and not final:
            completed, skip = False, None
        stages.append({'phase': phase, 'status': 'COMPLETED_EVIDENCE' if completed else
                       'SKIPPED_REGISTERED_STOP' if skip else 'INCOMPLETE',
                       'actual_numerical_execution_passed': numerical_pass, 'skip_reason': skip,
                       'artifacts': items, 'scientific_acceptance_passed':
                       final_check['scientific_acceptance_passed'] if phase == 'final-once' else None})
    frozen_sources_unchanged = bool(protocol) and all((ROOT / name).is_file() and sha(ROOT / name) == digest
                                      for name, digest in protocol.get('source_sha256', {}).items())
    sealed_started = (REPORTS / 'FINAL_ONCE_STARTED.json').exists()
    sealed_generated = any((DATA / name / 'data_manifest.json').exists() for name in ('signed_confirmation', 'final_once'))
    result = {'schema': 'v5-task-stage-completion-audit-001', 'time': now(),
              'task_spec_sha256': sha(CODE / 'TASK_SPEC.md'), 'stages': stages,
              'source_frozen_and_unchanged': frozen_sources_unchanged,
              'research_route_fully_audited': frozen_sources_unchanged and all(stage['status'] != 'INCOMPLETE' for stage in stages),
              'scientific_final_acceptance_passed': final_check['scientific_acceptance_passed'],
              'sealed_evaluation_started': sealed_started, 'sealed_arrays_generated': sealed_generated,
              'sealed_evaluation_completed': final_check['completed'],
              'final_completion_verification': final_check,
              'six_scientific_questions_report': relative(REPORTS / 'V5_SCIENTIFIC_DECISION.md'),
              'negative_results_preserved': True,
              'limits': ['CPU implementation QA does not establish 256x256 scientific acceptance',
                         'an inapplicable later phase may be stopped by its preregistered gate',
                         'started/partial sealed evaluation is incomplete and cannot be rerun',
                         'synthetic calibration/finite modes do not establish experimental gas temperature or concentration']}
    write_json(REPORTS / 'COMPLETION_AUDIT.json', result, replace=True)
    return result


def contract_checks():
    """Pure CPU/stdlib controls for exact accounting and strict gate behavior."""
    plan = [{"case_id": f"m{mother}_{coefficient}_{kind}", "mother_id": f"m{mother}",
             "coefficient_e": coefficient, "input_kind": kind}
            for mother in range(12) for coefficient in (0., 45., -45., 180., -180.)
            for kind in ("continuous", "quantization", "noise")]
    protocol = {"oracle_noiseless_gate": {"input_kind": "continuous", "coefficients_e": [180., -180.],
                 "pooled_RMSE_e_max": .5, "pooled_L2_max": .02, "support_sign_recall_min": .99},
                "CG": {"true_rtol": 1e-4}, "oracle_crosstalk_screen": {
                    "pooled_MSE_factor_over_O0": 1.25, "absolute_MSE_allowance_e2": 1.,
                    "collapsed_noiseless_normal_RMSE_e": 5.},
                "baselines": ["B1_tikhonov", "v3_physics", "old_B4", "v4.4_final_continuum_fixed"]}
    rows = []
    for case in plan:
        for level in LEVELS:
            rows.append({**case, "level": level, "status": "RETURNED", "raw_file_sha256": "raw-file",
                         "label_sha256": "label", "source_descriptor_sha256": "descriptor", "source_sha256": {"test": "source"},
                         "raw_dn_sha256": "raw-array", "raw_e_sha256": "raw-e", "latency_s": .1,
                         "metrics": {"template_SSE": 1., "truth_energy": 10000., "pixels": 100,
                                     "truth_support_pixels": 50, "sign_recall": 1., "signed_projection": 1.,
                                     "raw_weight_sha256": "weights"},
                         "CG": {"converged": True, "accepted": True, "relative_true_residual": 1e-6}})
    checks = []
    summary = oracle_summary(rows, plan, protocol)
    checks.append({"name": "all_180_720_24_gate", "passed": summary["o0_noiseless_gate"]["passed"]})
    checks.append({"name": "missing_case_cannot_pass", "passed": not oracle_summary(rows[:-4], plan, protocol)["o0_noiseless_gate"]["passed"]})
    copied = json.loads(json.dumps(rows))
    copied[-1]['status'] = 'NOT_EXECUTED_BUDGET'
    checks.append({'name': 'planned_budget_placeholders_cannot_claim_full_execution',
                   'passed': not oracle_summary(copied, plan, protocol)['complete']})
    copied = json.loads(json.dumps(rows))
    selected = next(r for r in copied if r["level"] == "O0" and r["input_kind"] == "continuous" and r["coefficient_e"] == 180.)
    selected["CG"]["relative_true_residual"] = .01
    checks.append({"name": "converged_flag_without_true_residual_cannot_pass",
                   "passed": not oracle_summary(copied, plan, protocol)["o0_noiseless_gate"]["passed"]})
    copied = json.loads(json.dumps(rows))
    copied[0]["raw_dn_sha256"] = "wrong-raw"
    checks.append({"name": "different_raw_cannot_pass", "passed": not oracle_summary(copied, plan, protocol)["o0_noiseless_gate"]["passed"]})
    copied = json.loads(json.dumps(rows))
    for r in copied:
        if r["level"] == "O1":
            r["metrics"]["template_SSE"] = 3600.
    checks.append({"name": "O1_collapse_blocks_ML", "passed": oracle_summary(copied, plan, protocol)["ml_stop_required"]})
    pooled = pool_oracle_rows([{"metrics": {"pixels": 1, "template_SSE": 1., "truth_energy": 100.,
                                               "truth_support_pixels": 1, "sign_recall": 1., "signed_projection": 1.}},
                              {"metrics": {"pixels": 9, "template_SSE": 9., "truth_energy": 100.,
                                               "truth_support_pixels": 9, "sign_recall": 0., "signed_projection": 0.}}])
    checks.append({"name": "pooled_errors_use_counts_and_energy", "passed": abs(pooled["pooled_RMSE_e"] - 1.) < 1e-12
                   and abs(pooled["pooled_L2"] - math.sqrt(.05)) < 1e-12
                   and abs(pooled["pooled_support_sign_recall"] - .1) < 1e-12})
    for status in ('NOT_ELIGIBLE', 'FINAL_EXECUTION_FAILED_NO_RETRY_OR_RETUNING', 'FINAL_ONCE_STARTED'):
        value = final_evaluation_receipt({'status': status, 'scientific_acceptance_passed': True}, {})
        checks.append({'name': status + '_is_not_completed_final_evaluation',
                       'passed': value['completed'] is False and value['scientific_acceptance_passed'] is False})
    assert all(check["passed"] for check in checks), checks
    return checks
