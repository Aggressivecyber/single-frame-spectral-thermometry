"""Read-only adapters for exactly four already frozen comparison methods.

This module selects no hyperparameters and changes no checkpoint. A single raw
DN frame is converted once. The v3 inverse is shared with the v4.4 post-CNN;
sharing its computed state does not add another measurement to any objective.
"""
import copy
import json
from pathlib import Path
import time

import runtime
import torch


BASELINE_NAMES = ("B1_tikhonov", "v3_physics", "old_B4", "v4.4_final_continuum_fixed")
BACKBONE_PATH = runtime.ROOT / "reports/training_v4/run_001/backbone_selection.json"
OLD_CHECKPOINT = runtime.ROOT / "checkpoints/selected_checkpoint.pt"
V44_CHECKPOINT = runtime.ROOT / "training_v4_4/checkpoints/fixed_continuum/update_0500.pt"
V44_CHECKPOINT_SHA256 = "0bc8ee79dcc82447c40662373c42bb8d158ffbc33b85f457d771c2dc0043e90b"


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _canonical(out, physics, info, raw_e):
    template = out.get("template_e")
    if template is None:
        template = out["species_signed_e"][:, 2]
    result = {"template_postblur_e": template.detach(),
              "temperature_color_K": out["temperature_color_K"].detach(),
              "valid": out["valid"].detach().bool(), "expected_e": out["expected_e"].detach(),
              "raw_e": raw_e.detach(), "raw_residual_e": out["expected_e"].detach()-raw_e,
              "info": info}
    if "state" in out:
        result["state"] = out["state"].detach()
        result["template_latent_e"] = out["state"][:, 4].detach()*physics.Sscale
    beta = out.get("beta", out.get("background_beta"))
    if beta is not None:
        result["beta"] = beta.detach()
    if "continuum_images_e" in out:
        result["continuum_postblur"] = out["continuum_images_e"].detach()
    for name in ("template_postblur_e", "temperature_color_K", "valid", "expected_e"):
        if tuple(result[name].shape) != tuple(raw_e.shape):
            raise ValueError("Baseline returned an invalid shape for " + name)
    if not all(bool(torch.isfinite(result[name]).all()) for name in
               ("template_postblur_e", "temperature_color_K", "expected_e")):
        raise ValueError("Baseline produced non-finite output")
    return result


def _abstained(physics, raw_e, name, reason, info=None):
    invalid = torch.full_like(raw_e, float("nan"))
    return {"template_postblur_e": invalid.clone(), "temperature_color_K": invalid.clone(),
            "expected_e": invalid.clone(), "valid": torch.zeros_like(raw_e, dtype=torch.bool),
            "raw_e": raw_e.detach(),
            "info": {"method": name, "status": "ABSTAINED", "reason": reason, **(info or {})}}


class BaselineSet:
    """Load frozen artifacts for later root-owned formal execution.

    ``cfg`` must equal the frozen physics. ``selection_metadata`` may be the
    already read backbone JSON, or its path; its consumed config digest and
    traditional selection are checked. Construction reads both weight files;
    bounded implementation checks use the private component-injection helper.
    """
    def __init__(self, device="cpu", cfg=None, selection_metadata=None):
        from infer import load_model as load_old_model
        from evaluation_v2.common import canonical_hash
        from training_v4_4.infer import (load_model as load_v44_model,
                                       FROZEN_BACKBONE_CONFIG_SHA256)
        from evaluation_v2.methods import demod_fit
        from training_v4.scalar_engine import invert_single
        from training_v4.model import physical_output

        if selection_metadata is None:
            selection_metadata = BACKBONE_PATH
        if isinstance(selection_metadata, (str, Path)):
            selection_path = Path(selection_metadata).resolve()
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection_source = {"path": str(selection_path), "file_sha256": runtime.sha256(selection_path)}
        else:
            selection = copy.deepcopy(selection_metadata)
            selection_source = {"kind": "passed external frozen selection mapping"}
        consumed = {key: selection[key] for key in ("method", "solver_config", "physics_config")}
        if canonical_hash(consumed) != FROZEN_BACKBONE_CONFIG_SHA256:
            raise ValueError("Passed backbone differs from the frozen v3/v4.4 configuration")
        if selection["method"] != "old_init__joint_update_v3" or selection.get("traditional_selected") != 0.1:
            raise ValueError("Only frozen old_init__joint_update_v3 and B1 regularization 0.1 are admitted")
        if cfg is not None and canonical_hash(cfg) != canonical_hash(selection["physics_config"]):
            raise ValueError("Baseline physics cannot differ from the frozen selection physics")
        before_hash = runtime.sha256(V44_CHECKPOINT)
        if before_hash != V44_CHECKPOINT_SHA256:
            raise ValueError("v4.4 update_0500 checkpoint SHA256 changed")
        old_model = load_old_model(OLD_CHECKPOINT, device=device)
        v44_model, saved_v44 = load_v44_model(V44_CHECKPOINT, consumed, device=device)
        if runtime.sha256(V44_CHECKPOINT) != before_hash:
            raise ValueError("v4.4 checkpoint changed during loading")
        if canonical_hash(old_model.cfg) != canonical_hash(selection["physics_config"]):
            raise ValueError("Published B4 checkpoint uses different frozen physics")
        provenance = {"selection": selection_source,
                      "consumed_selection_sha256": canonical_hash(consumed),
                      "physics_config_sha256": canonical_hash(selection["physics_config"]),
                      "solver_config_sha256": canonical_hash(selection["solver_config"]),
                      "old_B4_checkpoint": str(OLD_CHECKPOINT),
                      "old_B4_checkpoint_sha256": runtime.sha256(OLD_CHECKPOINT),
                      "v4.4_checkpoint": str(V44_CHECKPOINT), "v4.4_checkpoint_sha256": before_hash,
                      "v4.4_optimizer_updates": saved_v44.get("optimizer_updates"),
                      "truth_or_conditions_loaded": False, "weights_or_config_modified": False,
                      "checkpoint_or_hyperparameter_selection_in_this_adapter": False}
        self._configure(device, selection, old_model, v44_model,
                        demod_fit, invert_single, physical_output, provenance)

    def _configure(self, device, selection, old_model, v44_model,
                   demod_fit, invert_single, physical_output, provenance):
        self.device = torch.device(device)
        self.selection = copy.deepcopy(selection)
        self.old_model, self.v44_model = old_model, v44_model
        self.physics = v44_model.physics
        self._demod_fit, self._invert_single, self._physical_output = demod_fit, invert_single, physical_output
        self.provenance = provenance

    @classmethod
    def _from_components_for_contract_test(cls, device, selection, old_model, v44_model,
                                           demod_fit, invert_single, physical_output):
        """Private CPU mock seam; no checkpoint reads or formal baseline claims."""
        if torch.device(device).type != "cpu":
            raise ValueError("Mock component injection is restricted to CPU contract tests")
        obj = cls.__new__(cls)
        obj._configure(device, selection, old_model, v44_model, demod_fit,
                       invert_single, physical_output,
                       {"scope": "CPU component mock, no weights loaded", "formal_baseline": False})
        return obj

    @torch.no_grad()
    def evaluate(self, raw_dn):
        raw = torch.as_tensor(raw_dn, device=self.device, dtype=self.physics.M.dtype)
        if raw.ndim == 2:
            raw = raw[None]
        if raw.ndim == 4 and raw.shape[1] == 1:
            raw = raw[:, 0]
        if tuple(raw.shape) != (1, *self.physics.cfg["shape"]) or not bool(torch.isfinite(raw).all()):
            raise ValueError("BaselineSet requires one finite B=1 raw DN frame")
        maximum = 2**self.physics.cfg["adc_bits"]-1
        if bool(((raw < 0) | (raw > maximum)).any()):
            raise ValueError("Raw DN is outside the frozen ADC range")
        raw_e = self.physics.raw_to_electrons(raw)
        if bool((raw >= maximum).any()):
            return {name: _abstained(self.physics, raw_e, name, "SATURATION_REJECTED")
                    for name in BASELINE_NAMES}
        outputs = {}

        def run(name, action, metadata=None):
            _sync(self.device)
            started = time.perf_counter()
            try:
                value = action()
                _sync(self.device)
                info = {"method": name, "status": "SUCCESS", "seconds": time.perf_counter()-started,
                        "provenance": self.provenance, **(metadata or {})}
                outputs[name] = _canonical(value, self.physics, info, raw_e)
            except Exception as error:
                _sync(self.device)
                outputs[name] = _abstained(self.physics, raw_e, name, str(error),
                    {"error_type": type(error).__name__, "seconds": time.perf_counter()-started,
                     "provenance": self.provenance, **(metadata or {})})

        run("B1_tikhonov", lambda: self._demod_fit(self.physics, raw_e, 0.1),
            {"regularization": 0.1, "target_domain": "postblur local band fit; no extra object blur"})
        shared = {}

        def v3_action():
            state, beta, history, engine_seconds = self._invert_single(
                self.physics, raw_e, self.selection["method"], self.selection["solver_config"])
            shared.update(state=state, beta=beta, history=history, engine_seconds=engine_seconds)
            return self._physical_output(self.physics, state, beta, raw_e)

        run("v3_physics", v3_action,
            {"backbone_method": self.selection["method"], "optimizer_updates": 0,
             "solver_budget_metadata_enforced_by_scalar_engine": False})
        if shared:
            outputs["v3_physics"]["info"].update(inverse_history=shared["history"],
                                                engine_seconds=shared["engine_seconds"])
        run("old_B4", lambda: self.old_model.infer_electrons(raw_e, use_prior=True),
            {"method_implementation": "original ParameterUnfoldedNet.infer_electrons(use_prior=True)",
             "CNN_used": True, "continuum_head_active": True})
        if not shared or outputs["v3_physics"]["info"]["status"] != "SUCCESS":
            outputs["v4.4_final_continuum_fixed"] = _abstained(
                self.physics, raw_e, "v4.4_final_continuum_fixed", "shared frozen v3 inversion failed",
                {"provenance": self.provenance})
        else:
            def v44_action():
                value = self.v44_model.forward_electrons(raw_e, shared["state"], shared["beta"])
                if not torch.equal(value["state"][:, :2], shared["state"][:, :2]):
                    raise ValueError("Frozen v4.4 changed its final continuum state")
                return value
            run("v4.4_final_continuum_fixed", v44_action,
                {"CNN_used": True, "shared_inverse_source": "v3_physics on the same raw",
                 "final_continuum_fixed": True, "internal_continuum_proposal_still_active": True,
                 "seconds_scope": "post-CNN only; include shared v3 inverse seconds for full route"})
            outputs["v4.4_final_continuum_fixed"]["info"]["shared_inverse_seconds"] = outputs["v3_physics"]["info"]["seconds"]
        return outputs
