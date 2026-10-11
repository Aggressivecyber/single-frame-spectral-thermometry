"""B0--B4 read only raw DN. Diagnostic electron/oracle paths have separate names."""
import time
import numpy as np
import torch
from physics import filt
from solver import data_step
from .common import MeasurementInput

METHODS = ("B0_constant_train", "B0_mean_field_train", "Z0_zero_template",
           "B1_demod_fit", "B2_six_stage_no_CNN", "B3_physics_iterated", "B4_published", "B1_tikhonov")
RECONSTRUCTING = METHODS[3:]


def state_output(model, state, beta, raw_e):
    p = model.physics
    c, s, b, _ = p.components(state, beta)
    bands = p.op.object_blur(c+s+b)
    cbands = p.op.object_blur(c)
    temp, valid = p.color_temperature(cbands)
    expected = p.op.encode_images(bands)
    valid = valid & torch.isfinite(expected) & (expected >= 0) & (bands.amin(1) >= -1e-3)
    species = p.op.object_blur(state[:, 2:]*p.Sscale*p.active[None, :, None, None])
    return {"temperature_color_K": temp, "species_signed_e": species,
            "valid": valid, "expected_e": expected,
            "state": state, "background_beta": beta}


def demod_fit(physics, raw_e, regularization=0., known_ratio=None):
    decoded = physics.op.decode(raw_e)
    if regularization:
        h, w = decoded.shape[-2:]
        fy, fx = torch.meshgrid(torch.fft.fftfreq(h, device=raw_e.device),
                                torch.fft.fftfreq(w, device=raw_e.device), indexing="ij")
        # Quadratic periodic neighbour penalty on postblur band fields.
        transfer = (1+regularization*(4*torch.sin(torch.pi*fx).square()+4*torch.sin(torch.pi*fy).square())).reciprocal()
        decoded = filt(decoded, transfer)
    color_input = decoded
    if known_ratio is not None:
        color_input = decoded.clone()
        color_input[:, 5] /= known_ratio
    T, valid = physics.color_temperature(color_input)
    C, _ = physics.continuum(T)
    norm = C.square().sum(1, keepdim=True).sqrt().clamp_min(1)
    # Local fit is already in the post-object-blur domain. No extra object blur.
    design = torch.stack([C/norm,
                          (physics.M[:, 2]*physics.Sscale)[None, :, None, None].expand_as(C),
                          (physics.q*physics.Bscale)[None, :, None, None].expand_as(C)], dim=-1)
    design = design.permute(0, 2, 3, 1, 4)
    gram = design.transpose(-2, -1) @ design
    gram += torch.diag_embed(gram.diagonal(dim1=-2, dim2=-1).clamp_min(1)*1e-5)
    rhs = design.transpose(-2, -1) @ decoded.permute(0, 2, 3, 1).unsqueeze(-1)
    coefficients = torch.linalg.solve(gram, rhs).squeeze(-1)
    bands = (design*coefficients.unsqueeze(-2)).sum(-1).permute(0, 3, 1, 2)
    expected = physics.op.encode_images(bands)
    species = torch.zeros((len(raw_e), 3, *T.shape[-2:]), device=raw_e.device, dtype=raw_e.dtype)
    species[:, 2] = coefficients[..., 1]*physics.Sscale
    valid = valid & torch.isfinite(expected) & (expected >= 0) & (bands.amin(1) >= -1e-3)
    return {"temperature_color_K": T, "species_signed_e": species, "valid": valid,
            "expected_e": expected, "decoded_bands_e": decoded}


def _snapshot(model, state, beta, raw_e, label):
    result = state_output(model, state, beta, raw_e)
    return {"label": label, "temperature_K": result["temperature_color_K"][0].cpu().numpy(),
            "template_e": result["species_signed_e"][0, 2].cpu().numpy(),
            "valid": result["valid"][0].cpu().numpy(),
            "standardized_objective": float(((result["expected_e"]-raw_e).square() /
                                              (result["expected_e"].clamp_min(0)+model.physics.extra_variance)).mean())}


def traced_six_stage(model, raw_e, use_prior=True):
    state, beta = model.physics.initialize(raw_e)
    trace = [_snapshot(model, state, beta, raw_e, "initialization")]
    for stage in range(model.cfg["steps"]):
        state, beta = data_step(model.physics, raw_e, state, beta)
        trace.append(_snapshot(model, state, beta, raw_e, f"stage{stage+1}_data1"))
        if use_prior:
            shared = model.trunk(model.features(raw_e, state, beta, stage))
            correction = torch.cat([model.continuum_head(shared), model.emission_head(shared)], 1).tanh()*model.correction_bounds
            state, beta = model.physics.constrain(state+correction, beta)
            trace.append(_snapshot(model, state, beta, raw_e, f"stage{stage+1}_CNN"))
        state, beta = data_step(model.physics, raw_e, state, beta)
        trace.append(_snapshot(model, state, beta, raw_e, f"stage{stage+1}_data2"))
    return state_output(model, state, beta, raw_e), trace


def iterated_physics(model, raw_e, *, max_steps=60, cg_steps=12, tolerance=1e-6,
                     deadline=None, save_trace=False):
    state, beta = model.physics.initialize(raw_e)
    curve, trace = [], []
    if save_trace:
        trace.append(_snapshot(model, state, beta, raw_e, "initialization"))
    previous, stop = None, "MAX_ITERATIONS"
    for iteration in range(1, max_steps+1):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Pure-physics per-frame budget exhausted")
        state, beta = data_step(model.physics, raw_e, state, beta, cg_steps=cg_steps)
        out = state_output(model, state, beta, raw_e)
        objective = float(((out["expected_e"]-raw_e).square() /
                           (out["expected_e"].clamp_min(0)+model.physics.extra_variance)).mean())
        curve.append({"iteration": iteration, "standardized_objective": objective})
        if save_trace:
            trace.append(_snapshot(model, state, beta, raw_e, f"iteration{iteration}_data"))
        if iteration >= 12 and previous is not None and abs(previous-objective) <= tolerance*max(abs(previous), 1e-12):
            stop = "OBJECTIVE_STAGNATION"
            break
        previous = objective
    return out, {"iterations": len(curve), "stop_reason": stop, "curve": curve,
                 "CG_steps_spectral": cg_steps, "stationarity_certified": False,
                 "global_optimum_certified": False}, trace


class Reconstructor:
    def __init__(self, model, name, training_statistics=None, *, regularization=0.,
                 max_steps=60, per_frame_wall_s=60., known_ratio=None):
        self.model, self.name = model, name
        self.training_statistics = training_statistics
        self.regularization = regularization
        self.max_steps, self.per_frame_wall_s = max_steps, per_frame_wall_s
        self.known_ratio = known_ratio

    def reconstruct(self, inputs, *, save_trace=False):
        if type(inputs) is not MeasurementInput:
            raise TypeError("Only MeasurementInput is accepted, never a frozen truth record")
        return self._infer(inputs.raw_dn, "raw_DN", save_trace)

    def diagnose_continuous_electrons(self, electrons, *, save_trace=False):
        """Oracle only: caller must label this path; never the deployment interface."""
        return self._infer(electrons, "oracle_continuous_e", save_trace)

    def _infer(self, array, input_kind, save_trace):
        model, p = self.model, self.model.physics
        device = p.M.device
        raw = torch.as_tensor(np.array(array, copy=True), dtype=p.M.dtype, device=device)[None]
        if not torch.isfinite(raw).all() or tuple(raw.shape[-2:]) != tuple(model.cfg["shape"]):
            raise ValueError("Invalid raw array")
        if input_kind == "raw_DN":
            if raw.min() < 0 or raw.max() > 2**model.cfg["adc_bits"]-1:
                raise ValueError("Out-of-bounds raw DN")
            if torch.any(raw >= 2**model.cfg["adc_bits"]-1):
                return {"status": "SATURATION_REJECTED", "metrics": None}
            raw_e = p.raw_to_electrons(raw)
        else:
            raw_e = raw
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        trace, convergence = [], None
        with torch.no_grad():
            if self.name.startswith("B0_") or self.name == "Z0_zero_template":
                if self.name == "B0_constant_train":
                    temperature = np.full(array.shape, self.training_statistics["constant_K"])
                elif self.name == "B0_mean_field_train":
                    temperature = self.training_statistics["mean_field_K"].copy()
                else:
                    # This comparator assesses template regression only. Temperature N/A.
                    temperature = np.full(array.shape, np.nan)
                prediction = {"temperature_K": temperature, "template_e": np.zeros(array.shape),
                              "valid": np.ones(array.shape, bool),
                              "temperature_applicable": self.name != "Z0_zero_template"}
                out = None
            elif self.name in ("B1_demod_fit", "B1_tikhonov", "B1_known_700_905_ratio"):
                out = demod_fit(p, raw_e, self.regularization, self.known_ratio)
            elif self.name == "B2_six_stage_no_CNN":
                if save_trace:
                    out, trace = traced_six_stage(model, raw_e, False)
                else:
                    out = model.infer_electrons(raw_e, use_prior=False)
            elif self.name in ("B4_published", "B4_known_true_calibration"):
                if save_trace:
                    out, trace = traced_six_stage(model, raw_e, True)
                    # Verify the diagnostic replica against the actual published forward.
                    actual = model.infer_electrons(raw_e, use_prior=True)
                    if not torch.allclose(actual["species_signed_e"], out["species_signed_e"], atol=1e-5, rtol=1e-6):
                        raise RuntimeError("Tracing changed published output")
                else:
                    out = model.infer_electrons(raw_e, use_prior=True)
            elif self.name == "B3_physics_iterated":
                out, convergence, trace = iterated_physics(model, raw_e, max_steps=self.max_steps,
                                    deadline=start+self.per_frame_wall_s, save_trace=save_trace)
            else:
                raise ValueError("Unknown method "+self.name)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        result = {"status": "INFERRED", "seconds": seconds, "input_kind": input_kind,
                  "peak_GPU_allocated_MiB": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0.,
                  "convergence": convergence, "trace": trace}
        if out is not None:
            prediction = {"template_e": out["species_signed_e"][0, 2].cpu().numpy(),
                          "temperature_K": out["temperature_color_K"][0].cpu().numpy(),
                          "valid": out["valid"][0].cpu().numpy()}
            residual = out["expected_e"]-raw_e
            variance = out["expected_e"].clamp_min(0)+p.extra_variance
            result.update({"raw_reprojection_RMSE_e": float(residual.square().mean().sqrt()),
                           "raw_reprojection_RMSE_DN": float(residual.square().mean().sqrt()/p.adc_step),
                           "raw_standardized_residual_RMS": float((residual.square()/variance).mean().sqrt())})
        else:
            result.update({"raw_reprojection_RMSE_e": None, "raw_reprojection_RMSE_DN": None,
                           "raw_standardized_residual_RMS": None})
        result["prediction"] = prediction
        return result
