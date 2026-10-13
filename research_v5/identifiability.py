"""Finite-mode information and optical diagnostics; never a deployed estimator.

The Fisher columns use physical 1 K, 1 log-amplitude, and 1 electron
coefficients, with spatial modes of unit RMS. Beta keeps the physics' normalized
coordinates. Nuisance spans and S ranks use fixed, reported thresholds. These
small subspaces do not give the exact CRLB of an H x W parameter field.
"""
import copy
import math

import runtime  # Set E-drive cache/temp policy before importing torch.
import torch


FISHER_DEFAULTS = {
    "seed": 53117,
    "frequencies_xy": ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2)),
    "roi_boxes_xy": ((0.25, 0.75, 0.25, 0.75), (0.1, 0.45, 0.15, 0.55)),
    "mode_rank_rtol": 1e-10,
    "rank_rtol": 1e-8,
    "rank_atol": 1e-10,
    "amplitudes_e": (45.0, 180.0),
    "fd_epsilon": 1e-4,
    "fd_tolerance": 1e-5,
    "fd_max_directions": 8,
}
NULLSPACE_DEFAULTS = {"seed": 53118, "maximum_low_frequency_probes": 3}


def _options(defaults, supplied):
    supplied = {} if supplied is None else dict(supplied)
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError("Unknown diagnostic options: " + ", ".join(sorted(unknown)))
    return dict(defaults, **supplied)


def _number(value):
    result = float(value)
    return result if math.isfinite(result) else None


def _relative(a, b):
    return float(torch.linalg.vector_norm(a - b) /
                 torch.linalg.vector_norm(b).clamp_min(torch.finfo(b.dtype).tiny))


def _mode_bank(physics, options):
    """Fixed-order orthogonalization; ROI inputs are binary-cutoff bandlimited.

    Filtering a mode with a *binary* support mask is parameterization, not an
    extra object OTF. The physical OTF remains applied exactly once in p.jvp.
    """
    h, w = physics.cfg["shape"]
    dtype, device = physics.M.dtype, physics.M.device
    yy, xx = torch.meshgrid(torch.arange(h, dtype=dtype, device=device),
                            torch.arange(w, dtype=dtype, device=device), indexing="ij")
    support = (physics.op.object_otf > 0).to(dtype)
    proposed = [("DC", torch.ones_like(xx), {"kind": "constant"})]
    omitted = []
    for kx, ky in options["frequencies_xy"]:
        if int(kx) != kx or int(ky) != ky:
            raise ValueError("Mode frequencies must be integer FFT bins.")
        radius = math.hypot(kx / w, ky / h)
        if radius <= 0 or radius >= physics.cfg["object_cutoff_cpp"]:
            omitted.append({"kind": "frequency", "bins_xy": [kx, ky],
                            "reason": "outside nonzero object OTF support"})
            continue
        phase = 2 * math.pi * (kx * xx / w + ky * yy / h)
        for name, field in (("cos", phase.cos()), ("sin", phase.sin())):
            proposed.append((f"{name}_{kx}_{ky}", field,
                             {"kind": name, "bins_xy": [kx, ky], "frequency_cpp": radius}))
    for i, box in enumerate(options["roi_boxes_xy"]):
        x0, x1, y0, y1 = box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("ROI bounds must be ordered fractions of the field.")
        mask = ((xx >= x0 * w) & (xx < x1 * w) &
                (yy >= y0 * h) & (yy < y1 * h)).to(dtype)
        field = torch.fft.ifft2(torch.fft.fft2(mask) * support).real
        proposed.append((f"ROI_{i}", field, {"kind": "bandlimited_ROI", "box_xy": list(box)}))
    modes, descriptions = [], []
    for name, candidate, description in proposed:
        original_rms = candidate.square().mean().sqrt()
        field = candidate.clone()
        # Reorthogonalization avoids accepting numerical duplicate ROI modes.
        for _ in range(2):
            for accepted in modes:
                field = field - (field * accepted).mean() * accepted
        residual_rms = field.square().mean().sqrt()
        if float(residual_rms) <= max(float(original_rms), 1.0) * options["mode_rank_rtol"]:
            omitted.append({"name": name, "reason": "dependent after fixed-order orthogonalization"})
            continue
        field = field / residual_rms
        modes.append(field)
        descriptions.append(dict(description, name=name, rms=1.0,
                                 orthogonalized_to_previous=True,
                                 input_rms=float(original_rms),
                                 residual_input_rms=float(residual_rms)))
    return modes, descriptions, omitted


def _matrix_statistics(matrix, options, reference_largest=None, include_gram=False):
    singular = torch.linalg.svdvals(matrix)
    largest = float(singular[0]) if singular.numel() else 0.0
    reference = largest if reference_largest is None else reference_largest
    threshold = max(options["rank_atol"], options["rank_rtol"] * reference)
    rank = int((singular > threshold).sum())
    full_rank = rank == matrix.shape[1]
    minimum = float(singular[-1]) if singular.numel() else 0.0
    effective_minimum = float(singular[rank - 1]) if rank else None
    result = {
        "columns": matrix.shape[1], "rank": rank, "rank_threshold": threshold,
        "rank_reference_largest": reference,
        "singular_values": singular.detach().cpu().tolist(),
        "largest_singular_value": largest, "minimum_singular_value": minimum,
        "smallest_identifiable_singular_value": effective_minimum,
        "full_column_rank": full_rank,
        "condition_number": largest / minimum if full_rank and minimum > 0 else None,
        "condition_number_is_infinite": not full_rank,
        "identifiable_subspace_condition_number": largest / effective_minimum if rank else None,
    }
    if include_gram:
        result["gram"] = (matrix.T @ matrix).detach().cpu().tolist()
        if full_rank:
            _, s, vh = torch.linalg.svd(matrix, full_matrices=False)
            covariance = (vh.T / s.square()) @ vh
            result["finite_modal_linearized_sigma_e"] = covariance.diagonal().clamp_min(0).sqrt().cpu().tolist()
        else:
            result["finite_modal_linearized_sigma_e"] = None
            result["variance_limitation"] = "null directions have unbounded variance; no finite pseudoinverse error bar reported"
    return result


def _project_out(matrix, nuisance, options):
    if nuisance.shape[1] == 0:
        return matrix.clone(), {"columns": 0, "rank": 0}
    norms = torch.linalg.vector_norm(nuisance, dim=0)
    keep = norms > options["rank_atol"]
    normalized = nuisance[:, keep] / norms[keep]
    if normalized.shape[1] == 0:
        return matrix.clone(), {"columns": nuisance.shape[1], "rank": 0}
    u, s, _ = torch.linalg.svd(normalized, full_matrices=False)
    threshold = max(options["rank_atol"], options["rank_rtol"] * float(s[0]))
    rank = int((s > threshold).sum())
    span = u[:, :rank]
    residual = matrix - span @ (span.T @ matrix)
    return residual, {"columns": nuisance.shape[1], "nonzero_columns": int(keep.sum()),
                      "rank": rank, "rank_threshold": threshold,
                      "column_normalization": "unit weighted norm before nuisance SVD",
                      "singular_values_normalized": s.detach().cpu().tolist()}


def _directions(state, beta, modes, physics):
    result = []
    for group, channel, scale in (("T", 0, physics.Tscale), ("loga", 1, 1.0),
                                  ("S", 4, physics.Sscale)):
        for i, mode in enumerate(modes):
            direction = torch.zeros_like(state)
            direction[:, channel] = mode / scale
            result.append((group, i, direction, torch.zeros_like(beta)))
    for i in range(beta.shape[1]):
        db = torch.zeros_like(beta)
        db[:, i] = 1.0
        result.append(("beta", i, torch.zeros_like(state), db))
    return result


def _finite_difference_reference(physics, state, beta, directions, options):
    # Module.to(dtype=float64) would discard phases in complex decoder buffers.
    # Preserve complex values, including buffers unused by this FD forward test.
    p64 = copy.deepcopy(physics)._apply(
        lambda tensor: tensor.to(dtype=torch.complex128 if tensor.is_complex() else torch.float64)
        if tensor.is_complex() or tensor.is_floating_point() else tensor)
    z64, b64 = state.detach().double().clone(), beta.detach().double().clone()
    temperatures = p64.T0 + p64.Tscale * z64[:, 0]
    lut_step = p64.cfg["temperature_lut_step_K"]
    grid = (temperatures - p64.temperatures[0]) / lut_step
    near_knot = (grid - grid.round()).abs() < 1e-3
    # At LUT knots a central finite difference is not the selected right slope.
    # Move only the FD validation reference; Fisher remains at the supplied state.
    replacement = (temperatures + 0.25 * lut_step).clamp(
        float(p64.temperatures[0]) + 0.25 * lut_step,
        float(p64.temperatures[-1]) - 0.25 * lut_step)
    moved = torch.where(near_knot, replacement, temperatures)
    z64[:, 0] = (moved - p64.T0) / p64.Tscale
    selected = []
    for group in ("T", "loga", "S", "beta"):
        selected.extend([d for d in directions if d[0] == group][:2])
    rows = []
    for group, mode_index, direction, db in selected[:options["fd_max_directions"]]:
        dz64, db64 = direction.double(), db.double()
        eps = options["fd_epsilon"]
        if eps <= 0:
            raise ValueError("FD epsilon must be positive.")
        if group == "T":
            displacement = (p64.Tscale * dz64[:, 0]).abs()
            fraction = ((moved - p64.temperatures[0]) / lut_step).remainder(1)
            margin = torch.minimum(fraction, 1 - fraction) * lut_step
            active = displacement > 0
            if bool(active.any()):
                eps = min(eps, 0.25 * float((margin[active] / displacement[active]).min()))
        if eps <= 1e-12:
            rows.append({"group": group, "mode_index": mode_index, "checked": False,
                         "reason": "no finite central step within a single LUT segment"})
            continue
        actual = p64.jvp(z64, b64, dz64, db64)
        fd = (p64(z64 + eps * dz64, b64 + eps * db64) -
              p64(z64 - eps * dz64, b64 - eps * db64)) / (2 * eps)
        error = _relative(fd, actual)
        rows.append({"group": group, "mode_index": mode_index, "checked": True,
                     "epsilon_in_reported_parameter_units": eps, "relative_error": error,
                     "passed": error <= options["fd_tolerance"]})
    return {"dtype": "float64", "LUT_source": "existing physics buffers converted to float64",
            "shifted_temperature_pixels": int(near_knot.sum()),
            "maximum_reference_temperature_shift_K": float((moved - temperatures).abs().max()),
            "reference_shift_only_affects_derivative_test": True,
            "Fisher_uses_unmodified_supplied_state": True,
            "tolerance": options["fd_tolerance"], "checks": rows,
            "all_checked_passed": all(row.get("passed", False) for row in rows)}


@torch.no_grad()
def fisher_identifiability(physics, state, beta, raw_e, options=None):
    """Return JSON-safe local, finite-mode Fisher diagnostics on each frame.

    ``state`` is a diagnostic linearization point, possibly truth in an oracle
    experiment. It must never be passed from truth to deployed reconstruction.
    The supplied raw fixes W; weights are not re-estimated from a candidate fit.
    """
    opt = _options(FISHER_DEFAULTS, options)
    if raw_e.ndim == 4 and raw_e.shape[1] == 1:
        raw_e = raw_e[:, 0]
    shape = tuple(physics.cfg["shape"])
    if state.ndim != 4 or state.shape[1] != 5 or tuple(state.shape[-2:]) != shape:
        raise ValueError("state must have shape B x 5 x H x W.")
    if tuple(beta.shape) != (len(state), 3) or tuple(raw_e.shape) != (len(state), *shape):
        raise ValueError("beta/raw shapes do not match state.")
    if physics.active.tolist() != [0.0, 0.0, 1.0]:
        raise ValueError("v5 information path requires only species index 2 active.")
    if not all(bool(torch.isfinite(x).all()) for x in (state, beta, raw_e)):
        raise ValueError("All diagnostic inputs must be finite.")
    if opt["rank_rtol"] <= 0 or opt["rank_atol"] <= 0 or opt["mode_rank_rtol"] <= 0:
        raise ValueError("Rank thresholds must be positive and fixed before evaluation.")
    modes, descriptions, omitted = _mode_bank(physics, opt)
    frames = []
    adjoint_tolerance = 3e-5 if state.dtype == torch.float32 else 3e-11
    for frame in range(len(state)):
        z, b, y = state[frame:frame+1], beta[frame:frame+1], raw_e[frame:frame+1]
        weights = 1 / (y.clamp_min(0) + physics.extra_variance)
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("Fixed raw variance must be strictly positive and finite.")
        generator = torch.Generator(device=z.device).manual_seed(int(opt["seed"]) + frame)
        residual = torch.randn(y.shape, generator=generator, dtype=z.dtype, device=z.device)
        directions = _directions(z, b, modes, physics)
        columns = {name: [] for name in ("T", "loga", "S", "beta")}
        checks = []
        for group, index, dz, db in directions:
            jv = physics.jvp(z, b, dz, db)
            columns[group].append((weights.sqrt() * jv).flatten().double())
            check = {"group": group, "mode_index": index}
            for name, r in (("unweighted", residual), ("weighted", weights.sqrt() * residual)):
                gz, gb = physics.vjp(z, b, r)
                lhs, rhs = (jv * r).sum(), (gz * dz).sum() + (gb * db).sum()
                denominator = (lhs.abs() + rhs.abs()).clamp_min(torch.finfo(z.dtype).tiny)
                relative = float((lhs-rhs).abs() / denominator)
                check[name] = {"lhs": float(lhs), "rhs": float(rhs), "relative_error": relative,
                               "passed": relative <= adjoint_tolerance}
            checks.append(check)
        matrices = {name: torch.stack(value, dim=1) for name, value in columns.items()}
        species = matrices["S"]
        raw_species_stats = _matrix_statistics(species, opt, include_gram=True)
        projections = {}
        groups = {"none": (), "T_only": ("T",), "loga_only": ("loga",),
                  "beta_only": ("beta",), "T_loga": ("T", "loga"),
                  "T_beta": ("T", "beta"), "loga_beta": ("loga", "beta"),
                  "T_loga_beta": ("T", "loga", "beta")}
        for name, nuisance_groups in groups.items():
            nuisance = (torch.cat([matrices[group] for group in nuisance_groups], dim=1)
                        if nuisance_groups else species[:, :0])
            projected, nuisance_stats = _project_out(species, nuisance, opt)
            statistics = _matrix_statistics(projected, opt,
                                           raw_species_stats["largest_singular_value"], True)
            column_norms = torch.linalg.vector_norm(projected, dim=0)
            original_norms = torch.linalg.vector_norm(species, dim=0)
            statistics.update({"nuisance_groups": list(nuisance_groups),
                               "nuisance": nuisance_stats,
                               "retained_information_fraction_per_mode":
                               (column_norms.square() / original_norms.square().clamp_min(1e-30)).cpu().tolist(),
                               "single_mode_detection_SNR": {
                                   str(amplitude): (abs(amplitude) * column_norms).cpu().tolist()
                                   for amplitude in opt["amplitudes_e"]}})
            sigma = statistics["finite_modal_linearized_sigma_e"]
            statistics["joint_modal_coefficient_SNR"] = ({
                str(amplitude): [abs(amplitude) / value if value > 0 else None for value in sigma]
                for amplitude in opt["amplitudes_e"]} if sigma is not None else None)
            projections[name] = statistics
        physical_temperature = physics.T0 + physics.Tscale * z[:, 0]
        at_domain_edge = ((physical_temperature <= physics.temperatures[0]) |
                          (physical_temperature >= physics.temperatures[-1]))
        frames.append({"frame_index": frame,
                       "linearization_temperature_range_K": [float(physical_temperature.min()), float(physical_temperature.max())],
                       "temperature_pixels_at_or_outside_LUT_domain": int(at_domain_edge.sum()),
                       "LUT_domain_caveat": "analytic temperature Jacobian is the selected LUT slope; clamped domain edges are not smooth two-sided points",
                       "weight_range": [float(weights.min()), float(weights.max())],
                       "negative_raw_pixels_clamped_for_weight": int((y < 0).sum()),
                       "adjoint_tolerance": adjoint_tolerance, "adjoint_checks": checks,
                       "adjoint_all_passed": all(c[k]["passed"] for c in checks for k in ("unweighted", "weighted")),
                       "float64_finite_difference": _finite_difference_reference(physics, z, b, directions, opt),
                       "raw_S_information": raw_species_stats,
                       "nuisance_projected_S": projections})
    return {"schema": "v5-finite-modal-fisher-001", "options": opt,
            "input_role": "diagnostic linearization point; may be oracle truth, never deployment input",
            "weight_definition": "1 / (raw_e.clamp_min(0) + read_noise_e^2 + adc_step^2/12), fixed",
            "parameter_units": {"T": "K", "loga": "log amplitude", "S": "electrons in M_active*S",
                                "beta": "normalized physics background coefficients"},
            "spatial_amplitude_definition": "unit RMS modes; coefficient is RMS amplitude before object blur",
            "modes": descriptions, "omitted_modes": omitted, "frames": frames,
            "limitations": ["finite representative mode subspace, not full-image exact CRLB",
                            "finite nuisance modes may overstate information relative to an unrestricted full nuisance field",
                            "local linearization and fixed variance; calibration/model error excluded",
                            "single-mode SNR fits one S mode; joint modal SNR fits all reported S modes",
                            "nuisance orthogonal projection removes measurement components, not optical deconvolution",
                            "rank-deficient covariance is not replaced with a finite pseudoinverse interval"]}


@torch.no_grad()
def optical_nullspace_tests(physics, options=None):
    """Test actual OTF zero modes, transfer, carrier isolation, and adjoint."""
    opt = _options(NULLSPACE_DEFAULTS, options)
    op = physics.op
    h, w = physics.cfg["shape"]
    dtype, device = physics.M.dtype, physics.M.device
    tolerance = 3e-5 if dtype == torch.float32 else 3e-11
    yy, xx = torch.meshgrid(torch.arange(h, dtype=dtype, device=device),
                            torch.arange(w, dtype=dtype, device=device), indexing="ij")
    fy, fx = torch.meshgrid(torch.fft.fftfreq(h, device=device, dtype=dtype),
                            torch.fft.fftfreq(w, device=device, dtype=dtype), indexing="ij")
    radius = (fx.square() + fy.square()).sqrt()
    outside = (op.object_otf == 0) & (radius > physics.cfg["object_cutoff_cpp"])
    indices = torch.nonzero(outside, as_tuple=False)
    if not len(indices):
        raise ValueError("Configuration has no sampled exact object OTF zero modes.")
    # Choose the first frequency beyond cutoff on x if possible; otherwise a
    # deterministic zero-OTF bin from the real sampled operator.
    xzero = torch.nonzero(outside[0], as_tuple=False).flatten()
    ky_index, kx_index = (0, int(xzero[0])) if len(xzero) else map(int, indices[0])
    phase = 2 * math.pi * (kx_index * xx / w + ky_index * yy / h)
    high_mode = phase.cos()
    high_bands = torch.zeros((1, 7, h, w), dtype=dtype, device=device)
    high_bands[:, 2] = high_mode
    high_input_norm = torch.linalg.vector_norm(high_bands)
    high_object_norm = torch.linalg.vector_norm(op.object_blur(high_bands))
    high_raw_norm = torch.linalg.vector_norm(op(high_bands))
    high_without_blur = torch.linalg.vector_norm(op.encode_images(high_bands))
    probes = [("DC", 0, 0)]
    for kx, ky in ((1, 0), (0, 1), (1, 1), (2, 0)):
        if math.hypot(kx/w, ky/h) < physics.cfg["object_cutoff_cpp"]:
            probes.append((f"cos_{kx}_{ky}", kx, ky))
    probes = probes[:max(1, int(opt["maximum_low_frequency_probes"]))]
    low_checks = []
    for name, kx, ky in probes:
        mode = (2 * math.pi * (kx * xx/w + ky * yy/h)).cos()
        channel_rows = []
        for channel in range(7):
            bands = torch.zeros_like(high_bands)
            bands[:, channel] = mode
            target = op.object_blur(bands)
            decoded = op.decode(op(bands))
            transfer = float(torch.linalg.vector_norm(target) / torch.linalg.vector_norm(bands))
            target_norm = torch.linalg.vector_norm(target).clamp_min(torch.finfo(dtype).tiny)
            other = decoded.clone()
            other[:, channel] = 0
            error = _relative(decoded, target)
            leakage = float(torch.linalg.vector_norm(other) / target_norm)
            channel_rows.append({"channel": channel, "relative_decode_error": error,
                                 "other_channel_leakage_ratio": leakage,
                                 "measured_object_transmission": transfer,
                                 "expected_object_OTF": float(op.object_otf[ky % h, kx % w]),
                                 "passed": error <= tolerance and leakage <= tolerance})
        low_checks.append({"mode": name, "bins_xy": [kx, ky], "channels": channel_rows})
    generator = torch.Generator(device=device).manual_seed(int(opt["seed"]))
    x = torch.randn(high_bands.shape, generator=generator, device=device, dtype=dtype)
    r = torch.randn((1, h, w), generator=generator, device=device, dtype=dtype)
    lhs, rhs = (op(x)*r).sum(), (x*op.adjoint(r)).sum()
    adjoint_error = float((lhs-rhs).abs() / (lhs.abs()+rhs.abs()).clamp_min(torch.finfo(dtype).tiny))
    high_ratio = float(high_raw_norm/high_input_norm)
    return {"schema": "v5-optical-nullspace-001", "dtype": str(dtype), "options": opt,
            "tolerance": tolerance, "object_cutoff_cpp": physics.cfg["object_cutoff_cpp"],
            "sampled_zero_OTF_bins": len(indices),
            "high_frequency": {"bins_xy": [kx_index, ky_index],
                               "object_OTF": float(op.object_otf[ky_index, kx_index]),
                               "object_output_to_input_norm": float(high_object_norm/high_input_norm),
                               "complete_raw_output_to_input_norm": high_ratio,
                               "encode_without_object_blur_output_to_input_norm": float(high_without_blur/high_input_norm),
                               "passed": high_ratio <= tolerance and float(high_object_norm/high_input_norm) <= tolerance},
            "low_frequency_transfer_and_carrier_isolation": low_checks,
            "adjoint": {"lhs": float(lhs), "rhs": float(rhs), "relative_error": adjoint_error,
                        "passed": adjoint_error <= tolerance},
            "all_passed": high_ratio <= tolerance and adjoint_error <= tolerance and
                          all(row["passed"] for probe in low_checks for row in probe["channels"]),
            "interpretation": "carrier modulation after object OTF cannot restore pre-OTF zero-space information; decoded target is post-object-blur radiance"}
