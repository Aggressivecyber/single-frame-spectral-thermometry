"""Fixed-domain pooled estimates and paired mother-cluster uncertainty.

Pixels contribute to field losses, never to the count of independent scenes.
All stochastic intervals resample whole mother scenes with their repeats and
signed conditions kept together. Missing predictions invalidate an aggregate;
they are not removed to improve an error estimate.
"""
from __future__ import annotations

import math
from collections import defaultdict


BOOTSTRAP_PLAN = {"unit": "mother_scene", "draws": 2000, "seed": 53130,
                  "confidence_level": 0.95, "method": "percentile",
                  "keep_all_noise_repeats_and_signed_conditions_together": True}


def statistics_plan():
    return {**BOOTSTRAP_PLAN,
            "primary_targets": ["template_postblur_e", "continuum_postblur_color_temperature_K"],
            "error_domain": "full fixed truth evaluation_domain, including rejected pixels",
            "selection_population": "fresh development mothers; normal +/- noisy raw only",
            "primary_comparison": "paired mother mean template MSE and pooled RMSE by sign",
            "final_tuning": False, "pixel_independence_assumed": False}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metrics(record):
    return record.get("metrics") or {}


def _mse(record, field="template"):
    m = _metrics(record)
    key = "template_SSE" if field == "template" else "temperature_SSE_K2"
    n = m.get("domain_pixels", 0)
    return m[key]/n if n > 0 and _finite(m.get(key)) else None


def filter_records(records, *, method=None, group=None, input_kind=None, sign=None,
                   partition_role=None, alpha_g=None):
    return [r for r in records if (method is None or r.get("method") == method)
            and (group is None or r.get("group") == group)
            and (input_kind is None or r.get("input_kind") == input_kind)
            and (sign is None or r.get("sign") == sign)
            and (partition_role is None or r.get("partition_role") == partition_role)
            and (alpha_g is None or r.get("alpha_g") == alpha_g)]


def aggregate_metrics(records):
    """Pool full-domain SSE, preserve failures and report execution separately."""
    rows = list(records)
    if not rows:
        return {"schema": "v5-fixed-domain-aggregate-001", "complete": False,
                "reason": "empty population", "cases": 0, "mothers": 0}
    metrics = [_metrics(r) for r in rows]
    domains = [m.get("domain_pixels", 0) for m in metrics]
    complete = all(n > 0 and _finite(m.get("template_SSE"))
                   and _finite(m.get("temperature_SSE_K2"))
                   for n, m in zip(domains, metrics))
    result = {"schema": "v5-fixed-domain-aggregate-001", "complete": complete,
              "cases": len(rows), "mothers": len({r["mother_id"] for r in rows}),
              "domain_pixels": sum(domains),
              "numerical_success_cases": sum(bool(r.get("numerical_execution_passed")) for r in rows),
              "failed_or_abstained_cases": [r["case_id"] for r in rows
                                            if not r.get("numerical_execution_passed")],
              "pixel_independence_assumed": False}
    n = sum(domains)
    valid_known = all(_finite(m.get("accepted_pixels")) for m in metrics)
    result["valid_fraction"] = (sum(m["accepted_pixels"] for m in metrics)/n
                                if n and valid_known else None)
    if complete:
        sse = sum(m["template_SSE"] for m in metrics)
        tsse = sum(m["temperature_SSE_K2"] for m in metrics)
        energy = sum(m.get("template_truth_energy", 0) for m in metrics)
        sbias = sum(m.get("template_error_sum", 0) for m in metrics)/n
        tbias = sum(m.get("temperature_error_sum_K", 0) for m in metrics)/n
        result.update(template_SSE=sse, temperature_SSE_K2=tsse,
                      template_truth_energy=energy, template_MSE_e2=sse/n,
                      template_RMSE_e=math.sqrt(sse/n), temperature_MSE_K2=tsse/n,
                      temperature_RMSE_K=math.sqrt(tsse/n),
                      template_bias_e=sbias, temperature_bias_K=tbias,
                      template_spatial_error_variance_e2=max(0., sse/n-sbias*sbias),
                      bias_variance_scope="pooled spatial errors; MC repeat variance reported separately",
                      template_relative_L2=math.sqrt(sse/energy) if energy > 0 else None)
        projections = [m.get("signed_projection_numerator") for m in metrics]
        result["signed_projection"] = (sum(p for p in projections if _finite(p))/energy
                                        if energy > 0 and all(_finite(p) for p, m in
                                        zip(projections, metrics) if m.get("template_truth_energy", 0) > 0) else None)
    else:
        result.update(template_SSE=None, temperature_SSE_K2=None,
                      template_MSE_e2=None, template_RMSE_e=None,
                      temperature_RMSE_K=None, template_relative_L2=None,
                      template_bias_e=None, temperature_bias_K=None)
    diag = [r.get("diagnostics", {}) for r in rows]
    latencies = [d.get("seconds") for d in diag]
    known = [v for v in latencies if _finite(v)]
    result["latency"] = {"known_cases": len(known),
                         "mean_seconds": sum(known)/len(known) if known else None,
                         "total_seconds": sum(known) if known else None,
                         "shared_continuum_time_included": True}
    cg = [d.get("CG") for d in diag if d.get("CG") is not None]
    residuals = [c.get("relative_true_residual") for c in cg]
    equilibrated = [c.get("equilibrated_relative_true_residual") for c in cg]
    result["CG"] = {"reported_cases": len(cg), "converged_cases": sum(bool(c.get("converged")) for c in cg),
                    "all_converged": bool(cg) and len(cg) == len(rows) and all(c.get("converged") for c in cg),
                    "max_true_relative_residual": max(residuals) if residuals and all(_finite(v) for v in residuals) else None,
                    "max_equilibrated_true_relative_residual": max(equilibrated) if equilibrated and all(_finite(v) for v in equilibrated) else None}
    gradients = [d.get("gradient_energy_S_e2") for d in diag]
    gradients = [v for v in gradients if _finite(v)]
    result["mean_latent_S_gradient_energy_e2"] = sum(gradients)/len(gradients) if gradients else None
    return result


def mother_bootstrap(values_by_mother, *, draws=2000, seed=53130, confidence_level=.95):
    """Percentile CI of an equally weighted mother mean, with fixed RNG."""
    import numpy as np
    if draws != 2000 or seed != 53130 or confidence_level != .95:
        raise ValueError("The scientific bootstrap plan is frozen at 2000/53130/95%")
    items = sorted(values_by_mother.items())
    result = {**BOOTSTRAP_PLAN, "mothers": len(items), "mother_ids": [x[0] for x in items]}
    if not items or any(not _finite(x[1]) for x in items):
        return {**result, "complete": False, "estimate": None, "CI95": None,
                "reason": "empty population or missing mother metric"}
    values = np.asarray([x[1] for x in items], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    lower, upper = np.quantile(sample, [.025, .975])
    return {**result, "complete": True, "estimate": float(values.mean()),
            "CI95": [float(lower), float(upper)],
            "mother_values": {mid: float(value) for mid, value in items},
            "single_mother_CI_is_degenerate": len(items) == 1}


def mother_metric(records, *, field="template"):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["mother_id"]].append(_mse(row, field))
    return {mid: sum(values)/len(values) if all(_finite(v) for v in values) else None
            for mid, values in grouped.items()}


def paired_statistics(records, method, reference, *, group="normal", input_kind="noise",
                      sign=None, alpha_g=None):
    """Pair exactly the same raw cases, then resample their parent means."""
    selected = filter_records(records, method=method, group=group, input_kind=input_kind,
                              sign=sign, alpha_g=alpha_g)
    control = filter_records(records, method=reference, group=group, input_kind=input_kind, sign=sign)
    candidate = {r["case_id"]: r for r in selected}
    ref = {r["case_id"]: r for r in control}
    if len(candidate) != len(selected) or len(ref) != len(control):
        raise ValueError("Paired statistics requires one record per case/method/config")
    same = set(candidate) == set(ref) and bool(candidate)
    rows, grouped = [], defaultdict(list)
    for cid in sorted(set(candidate) | set(ref)):
        left, right = candidate.get(cid), ref.get(cid)
        l, r = _mse(left or {}), _mse(right or {})
        mother = (left or right)["mother_id"]
        if left and right and (left["mother_id"] != right["mother_id"] or
                              left.get("raw_sha256") != right.get("raw_sha256")):
            raise ValueError("Paired raw/mother hashes differ: " + cid)
        delta = l-r if _finite(l) and _finite(r) else None
        rows.append({"case_id": cid, "mother_id": mother, "candidate_MSE_e2": l,
                     "reference_MSE_e2": r, "paired_MSE_difference_e2": delta})
        grouped[mother].append((l, r, delta))
    mothers = {mid: {"candidate_MSE_e2": sum(v[0] for v in vals)/len(vals),
                     "reference_MSE_e2": sum(v[1] for v in vals)/len(vals),
                     "paired_MSE_difference_e2": sum(v[2] for v in vals)/len(vals)}
               if all(all(_finite(x) for x in v) for v in vals) else
               {"candidate_MSE_e2": None, "reference_MSE_e2": None, "paired_MSE_difference_e2": None}
               for mid, vals in grouped.items()}
    delta_ci = mother_bootstrap({mid: v["paired_MSE_difference_e2"] for mid, v in mothers.items()})
    ratio_ci = None
    estimate_ratio = None
    if same and mothers and delta_ci["complete"]:
        import numpy as np
        values = np.asarray([[mothers[mid]["candidate_MSE_e2"], mothers[mid]["reference_MSE_e2"]]
                             for mid in sorted(mothers)], dtype=np.float64)
        means = values.mean(0)
        if means[1] > 0:
            estimate_ratio = math.sqrt(means[0]/means[1])
            idx = np.random.default_rng(53130).integers(0, len(values), size=(2000, len(values)))
            means_b = values[idx].mean(1)
            if bool((means_b[:, 1] > 0).all()):
                ratio_ci = [float(x) for x in np.quantile(np.sqrt(means_b[:, 0]/means_b[:, 1]), [.025, .975])]
    return {"schema": "v5-paired-mother-statistics-001", "method": method, "reference": reference,
            "population": {"group": group, "input_kind": input_kind, "sign": sign},
            "complete": same and delta_ci["complete"], "same_raw_cases": same,
            "case_pairs": rows, "mother_pairs": mothers,
            "paired_MSE_difference_bootstrap": delta_ci,
            "paired_parent_RMSE_ratio": estimate_ratio, "paired_parent_RMSE_ratio_CI95": ratio_ci,
            "candidate_pooled": aggregate_metrics(selected), "reference_pooled": aggregate_metrics(control),
            "plan": statistics_plan()}


# Explicit small aliases used by phase consumers.
aggregate = aggregate_metrics
paired_compare = paired_statistics
