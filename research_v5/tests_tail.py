"""Bounded CPU controls for the neural boundary and irreversible final guards.

These use 16x16 tensors and synthetic JSON fixtures only. They never generate,
open or train on any formal mother family or any reserved evaluation payload.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
import unittest
import uuid
from unittest import mock
from research_v5 import common

common.configure_runtime()

import numpy as np
import torch

from research_v5 import final, train
from research_v5.signed_gain_denoiser import (
    SignedBoundedGainDenoiser, build_gain_features, apply_denoiser, load_denoiser, FEATURE_NAMES,
)
from research_v5.statistics import statistics_plan


def _fake_plans():
    mothers = {
        "signed_confirmation": [{"mother_id": f"fixture_signed_{i:02d}"} for i in range(36)],
        "final_once": [{"mother_id": f"fixture_final_{i:02d}"} for i in range(64)],
    }
    plans = {}
    for family, entries in mothers.items():
        rows = []
        coefficients = (45., -45., 180., -180.) if family == "signed_confirmation" else (0., 45., 180., 400.)
        for mother in entries:
            for coefficient in coefficients:
                for repeat in (0, 1):
                    group = "blank" if coefficient == 0 else "weak" if abs(coefficient) == 45 else "strong" if coefficient == 400 else "normal"
                    rows.append({"case_id": f"{mother['mother_id']}_{coefficient:g}_{repeat}".replace("-", "m"),
                        "mother_id": mother["mother_id"], "coefficient_e": coefficient,
                        "partition_role": family, "condition": group, "group": group,
                        "input_kind": "noise", "noise_index": repeat})
        plans[family] = rows
    return {"mothers": mothers, "plans": plans,
            "plan_hashes": {key: common.canonical_hash(rows) for key, rows in plans.items()}}


def _validation_records():
    records = []
    for sign in ("positive", "negative"):
        for group in ("normal", "weak"):
            case = sign + "_" + group
            records.append({"case_id": case, "mother_id": "m1", "group": group, "sign": sign,
                "numerical_execution_passed": True, "metrics": {"template_SSE": 100., "domain_pixels": 100,
                    "valid_fraction": 1., "z_threshold": 3., "null_frame_FP": None}})
    for i in range(8):
        records.append({"case_id": f"null_{i}", "mother_id": f"n{i}", "group": "blank", "sign": "zero",
            "numerical_execution_passed": True, "metrics": {"template_SSE": 100., "domain_pixels": 100,
                "valid_fraction": 1., "z_threshold": 3., "null_frame_FP": False, "null_pixel_FP": 0.}})
    neural = copy.deepcopy(records)
    for r in neural:
        if r["group"] == "weak":
            r["metrics"]["template_SSE"] *= .90
    weak = {sign: {"detectable_case_ids": [sign + "_weak"]} for sign in ("positive", "negative")}
    return neural, records, weak


class TailControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(53131)
        cls.run_name = "cpu_tail_" + uuid.uuid4().hex
        cls.report_root = common.REPORTS.parent / "implementation_checks" / "tail" / cls.run_name
        cls.data_root = common.DATA.parent / "implementation_checks" / "tail" / cls.run_name
        cls.report_root.mkdir(parents=True)
        cls.data_root.mkdir(parents=True)
        cfg = copy.deepcopy(common.physics_config())
        cfg["shape"] = [16, 16]
        cfg["carrier_bins_xy"] = [[round(x*16/256), round(y*16/256)] for x, y in cfg["carrier_bins_xy"]]
        cls.physics = common.make_physics("cpu", torch.float64, config=cfg)
        yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
        cls.s = (xx.double()-7)[None].expand(1, 16, 16).clone()*12
        cls.s[:, 0] = 0
        state = torch.zeros((1, 5, 16, 16), dtype=torch.float64)
        state[:, 0] = (3050-cls.physics.T0)/cls.physics.Tscale
        state[:, 1] = -.2
        beta = torch.tensor([[.7, .01, -.01]], dtype=torch.float64)
        continuum, species, background, _ = cls.physics.components(state, beta)
        expected = cls.physics.op(continuum + species + background)
        c = cls.physics.op.object_blur(continuum)
        temperature, valid = cls.physics.color_temperature(c)
        cls.raw = expected/cls.physics.adc_step + cls.physics.cfg["black_level_DN"]
        cls.sigma = torch.ones_like(cls.s)*7
        cls.physical = {"template_postblur_e": cls.s.clone(), "template_latent_e": cls.s.clone(),
            "continuum_latent": continuum.clone(), "continuum_postblur": c.clone(), "beta": beta.clone(),
            "temperature_color_K": temperature, "valid": valid,
            "expected_e": expected, "info": {"continuum_frozen": True}}
        cls.features = build_gain_features(cls.raw, cls.physics, template_postblur_e=cls.s,
            expected_e=expected, continuum_postblur=c, sigma_S=cls.sigma)

    def test_zero_head_is_exact_identity(self):
        model = SignedBoundedGainDenoiser()
        self.assertTrue(torch.equal(model(self.s, self.features), self.s))
        self.assertTrue(torch.count_nonzero(model.head.weight) == 0)
        self.assertTrue(torch.count_nonzero(model.head.bias) == 0)
        self.assertEqual(self.features.shape, (1, 6, 16, 16))

    def test_sign_and_zero_preserved_at_both_gain_bounds(self):
        model = SignedBoundedGainDenoiser()
        for bias, expected in ((30., 1.3), (-30., .7)):
            with torch.no_grad():
                model.head.bias.fill_(bias)
            prediction = model(self.s, self.features)
            self.assertTrue(torch.equal(prediction[self.s == 0], self.s[self.s == 0]))
            self.assertTrue(torch.equal(torch.sign(prediction), torch.sign(self.s)))
            ratio = prediction[self.s != 0]/self.s[self.s != 0]
            self.assertLess(float((ratio-expected).abs().max().detach()), 1e-7)

    def test_no_continuum_feedback_or_gradients(self):
        model = SignedBoundedGainDenoiser()
        physical = {k: v.detach().clone().requires_grad_(True) if isinstance(v, torch.Tensor)
                    and v.is_floating_point() else copy.deepcopy(v) for k, v in self.physical.items()}
        protected = {k: physical[k].detach().clone() for k in
                     ("continuum_latent", "continuum_postblur", "temperature_color_K", "beta", "expected_e")}
        result = apply_denoiser(model, self.raw, self.physics, physical, self.sigma)
        ((result["template_postblur_e"]-.8*self.s).square().mean()).backward()
        for key, before in protected.items():
            self.assertTrue(torch.equal(result[key], before))
            self.assertIsNone(physical[key].grad)
            self.assertFalse(result[key].requires_grad)
        self.assertFalse(result["info"]["denoiser"]["expected_e_is_modified_template_raw_refit"])
        self.assertEqual(len([p for p in model.parameters()]), 6)

    def test_truth_identity_and_invalid_sigma_refused(self):
        model = SignedBoundedGainDenoiser()
        contaminated = dict(self.physical, case_id="should_never_enter_deployment")
        with self.assertRaises(ValueError):
            apply_denoiser(model, self.raw, self.physics, contaminated, self.sigma)
        with self.assertRaises(TypeError):
            build_gain_features(self.raw, self.physics, template_postblur_e=self.s,
                expected_e=self.physical["expected_e"], continuum_postblur=self.physical["continuum_postblur"],
                sigma_S=self.sigma, truth=self.s)
        with self.assertRaises(ValueError):
            build_gain_features(self.raw, self.physics, template_postblur_e=self.s,
                expected_e=self.physical["expected_e"], continuum_postblur=self.physical["continuum_postblur"],
                sigma_S=torch.zeros_like(self.sigma))

    def test_real_raw_proxy_features_do_not_depend_on_supervision(self):
        core = {"continuum_options": {}, "linear_options": {"alpha_g": .1},
                "device": "cpu", "dtype": "float64"}
        truth = {"template_postblur_e": self.s[0].cpu().numpy(),
                 "evaluation_domain": self.physical["valid"][0].cpu().numpy()}
        first = train._prepare_case(self.physics, self.raw, truth, core, self.physical)
        changed_truth = dict(truth, template_postblur_e=-truth["template_postblur_e"])
        second = train._prepare_case(self.physics, self.raw, changed_truth, core, self.physical)
        self.assertTrue(np.array_equal(first["features"], second["features"]))
        self.assertTrue(np.array_equal(first["sigma_S"], second["sigma_S"]))
        self.assertTrue(np.array_equal(first["target_S"], -second["target_S"]))
        self.assertTrue(np.isfinite(first["sigma_S"]).all())
        self.assertTrue((first["sigma_S"] > 0).all())

    def test_training_factorial_and_parent_separation_enforced(self):
        mothers = [{"mother_id": f"train_{t}_{s}", "seed": t*6+s,
                    "temperature_geometry": f"t{t}", "template_geometry": f"s{s}"}
                   for t in range(6) for s in range(6)]
        development = [{"mother_id": f"dev_{i}", "seed": 100+i} for i in range(36)]
        rows = [{"mother_id": m["mother_id"], "coefficient_e": coefficient,
                 "noise_index": repeat, "partition_role": "training", "input_kind": "noise"}
                for m in mothers for coefficient in (0., 45., -45., 180., -180.) for repeat in (0, 1)]
        plans = {"mothers": {"training": mothers, "development": development}, "plans": {"training": rows}}
        self.assertEqual(len(train.validate_training_partition(plans)), 360)
        development[0]["seed"] = 0
        with self.assertRaises(ValueError):
            train.validate_training_partition(plans)
        development[0]["seed"] = 100
        rows[1]["noise_index"] = 0
        with self.assertRaises(ValueError):
            train.validate_training_partition(plans)

    def test_only_registered_architecture_allowed(self):
        for options in ({"width": 16}, {"gain_bound": .5}):
            with self.assertRaises(ValueError):
                SignedBoundedGainDenoiser(**options)

    def test_actual_cpu_optimizer_update_receipt_and_reopened_weights(self):
        torch.manual_seed(53131)
        model = SignedBoundedGainDenoiser()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        initial_hash = common.canonical_hash({k: common.canonical_hash(v.detach().cpu().tolist())
                                             for k, v in model.state_dict().items()})
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(self.s, self.features)-.8*self.s).square().mean()
            loss.backward()
            optimizer.step()
        receipt = train._save_checkpoint(self.data_root / "tiny_update3.pt", model, optimizer, 3,
                                         initial_hash, "fixture_selection", "fixture_protocol")
        self.assertEqual(receipt["optimizer_state_steps"], [3])
        self.assertTrue(receipt["weights_changed_from_initial"])
        reopened, payload = load_denoiser(self.data_root / "tiny_update3.pt", expected_sha256=receipt["sha256"])
        self.assertEqual(payload["actual_optimizer_updates"], 3)
        self.assertTrue(torch.equal(model(self.s, self.features), reopened(self.s, self.features)))
        self.assertGreater(float((reopened(self.s, self.features)-self.s).abs().max()), 0)

    def test_training_gates_are_fail_closed(self):
        spec = {"model": "SignedBoundedGainDenoiser", "width": 8, "gain_bound": .3, "last_head_zero": True,
            "seed": 53131, "optimizer": "Adam", "lr": 3e-4, "max_updates": 1000,
            "checkpoints": [0, 200, 500, 1000], "train_mothers": 36, "geometry_factorial": [6, 6],
            "reuse_old_bias_heads": False}
        protocol = {"training": spec}
        decision = {"stage1_evidence_complete": True, "physical_stage2_eligible": True, "ml_stop_required": False}
        info = {"complete": True, "numerical_checks_passed": True, "optical_all_passed": True, "returned_linearizations": 24}
        gate = {"physics_eligible": True, "no_hardstop": True,
            "weakdetectable_by_sign": {"positive": {"detectable_case_ids": ["p"]}, "negative": {"detectable_case_ids": ["n"]}}}
        self.assertTrue(train.validate_training_prerequisites(protocol, gate, decision, info)[0])
        for mapping, key, bad in ((decision, "ml_stop_required", True), (info, "numerical_checks_passed", False),
                                  (gate, "physics_eligible", False), (spec, "max_updates", 2000)):
            original = mapping[key]
            mapping[key] = bad
            self.assertFalse(train.validate_training_prerequisites(protocol, gate, decision, info)[0])
            mapping[key] = original
        gate["weakdetectable_by_sign"]["negative"]["detectable_case_ids"] = []
        self.assertEqual(train.validate_training_prerequisites(protocol, gate, decision, info),
                         (False, "no_detectable_weak_both_signs_preregistered_skip"))

    def test_one_weight_needs_both_signs_normal_weak_and_null(self):
        neural, physics, weak = _validation_records()
        self.assertTrue(train.evaluate_checkpoint_gate(neural, physics, weak, 3.)["accepted"])
        neural[1]["metrics"]["template_SSE"] = 96.
        self.assertFalse(train.evaluate_checkpoint_gate(neural, physics, weak, 3.)["accepted"])
        neural[1]["metrics"]["template_SSE"] = 90.
        neural[0]["metrics"]["template_SSE"] = 106.
        self.assertFalse(train.evaluate_checkpoint_gate(neural, physics, weak, 3.)["accepted"])
        neural[0]["metrics"]["template_SSE"] = 100.
        neural[-1]["metrics"]["null_frame_FP"] = True
        self.assertFalse(train.evaluate_checkpoint_gate(neural, physics, weak, 3.)["accepted"])
        neural[-1]["metrics"]["null_frame_FP"] = False
        self.assertFalse(train.evaluate_checkpoint_gate(neural, physics, weak, 4.)["accepted"])
        self.assertFalse(train.evaluate_checkpoint_gate(neural[:-1], physics, weak, 3.)["accepted"])

    def test_O_EXCL_final_guard_never_overwrites(self):
        path = self.report_root / "guard" / "FINAL_ONCE_STARTED.json"
        digest = train.exclusive_json(path, {"attempt": 1})
        with self.assertRaises(FileExistsError):
            train.exclusive_json(path, {"attempt": 2})
        self.assertEqual(common.sha(path), digest)
        self.assertEqual(common.read_json(path), {"attempt": 1})

    def test_ineligible_final_opens_no_reserved_data(self):
        root = self.report_root / "ineligible"
        candidate_path = root / "candidate_review" / "candidate_receipt.json"
        common.write_json(candidate_path, {"eligible": False, "accepted": False})
        with mock.patch.object(common, "REPORTS", root), mock.patch.object(final, "CANONICAL_CANDIDATE", candidate_path):
            summary = final.run_final_once(device="cpu")
        self.assertEqual(summary["status"], "NOT_ELIGIBLE")
        self.assertEqual(summary["reserved_payloads_opened"], 0)
        self.assertFalse(summary["final_arrays_generated"])
        self.assertFalse((root / "FINAL_ONCE_STARTED.json").exists())

    def test_final_freeze_binds_exact_rules_stats_population_and_version(self):
        plans = _fake_plans()
        candidate = {"accepted": True, "eligible": True, "candidate_kind": "v5_physics",
            "core_selection": {"continuum_options": {}, "linear_options": {}, "device": "cpu", "dtype": "float32"},
            "checkpoint": None, "thresholds": {"z_threshold": 3.}, "statistics": statistics_plan()}
        frozen = {"status": final.FROZEN_STATUS, "candidate_kind": "v5_physics",
            "core_selection": candidate["core_selection"], "thresholds": candidate["thresholds"],
            "statistics": candidate["statistics"], "plan_hashes": plans["plan_hashes"], **final.frozen_evaluation_fields(plans)}
        protocol = {"family_plan_hashes": plans["plan_hashes"]}
        self.assertTrue(final.validate_candidate_and_protocol(candidate, frozen, protocol, plans))
        for key, bad in (("status", "STARTED"), ("candidate_kind", "old_B4"),
                         ("statistics", {}), ("case_counts", {"total_noise": 672}), ("final_MC_case_ids", [])):
            changed = copy.deepcopy(frozen)
            changed[key] = bad
            with self.assertRaises(ValueError):
                final.validate_candidate_and_protocol(candidate, changed, protocol, plans)

    def test_below_limit_has_interval_and_abstention_without_truth(self):
        output = copy.deepcopy(self.physical)
        output["template_postblur_e"] = torch.zeros_like(self.s)
        result = final.attach_proxy_uncertainty(output, self.sigma, 3.)
        self.assertTrue(bool(result["low_confidence"].all()))
        self.assertTrue(bool(result["template_abstain"].all()))
        self.assertTrue(bool((result["signed_ci_lower_e"] < 0).all()))
        self.assertTrue(bool((result["signed_ci_upper_e"] > 0).all()))
        self.assertFalse(result["interval_empirical_coverage_claimed"])
        self.assertTrue(torch.equal(result["valid"], output["valid"]))

    def test_signed_parent_bootstrap_and_missing_case_failure(self):
        plans = _fake_plans()
        mc_ids = final.final_mc_case_ids(plans)
        records = []
        for rows in plans["plans"].values():
            for row in rows:
                for method in ("v5_physics", "B1_tikhonov"):
                    meta = train._metadata(row)
                    records.append({**meta, "method": method, "raw_sha256": "raw_" + row["case_id"],
                        "status": "SUCCESS", "numerical_execution_passed": True,
                        "metrics": {"domain_pixels": 64, "accepted_pixels": 64, "valid_fraction": 1.,
                            "template_SSE": 64., "temperature_SSE_K2": 64., "template_truth_energy": 64.,
                            "template_error_sum": 0., "temperature_error_sum_K": 0.,
                            "signed_projection_numerator": 64., "null_frame_FP": False,
                            "raw_only_detectability": False, "signed_interval": {"coverage_full_truth_domain": 1.}},
                        "MC": {"status": "completed"} if row["case_id"] in mc_ids and method == "v5_physics" else None,
                        "diagnostics": {"seconds": .001}})
        summary = final.summarize_final_records(records, "v5_physics", "B1_tikhonov", mc_ids)
        self.assertTrue(summary["scientific_acceptance_passed"])
        negative = summary["mother_bootstrap_CI95"]["signed_confirmation/normal/negative"]["template_MSE_e2"]
        self.assertEqual(negative["mothers"], 36)
        self.assertEqual(negative["unit"], "mother_scene")
        self.assertEqual(negative["CI95"], [1., 1.])
        reduced = [r for r in records if not (r["method"] == "v5_physics" and r["case_id"] == plans["plans"]["final_once"][-1]["case_id"])]
        failed = final.summarize_final_records(reduced, "v5_physics", "B1_tikhonov", mc_ids)
        self.assertFalse(failed["scientific_acceptance_passed"])
        self.assertFalse(failed["numerical_execution_passed"])

    def test_abstention_nan_output_preserved_and_reopened(self):
        path = self.data_root / "nan_abstention.npz"
        values = {"template_postblur_e": np.asarray([np.nan, 0., -1.])}
        digest = final._save_prediction(path, values)
        self.assertEqual(common.sha(path), digest)
        self.assertTrue(np.array_equal(common.arrays(path)["template_postblur_e"], values["template_postblur_e"], equal_nan=True))


def main():
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TailControls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {"schema": "v5-tail-bounded-CPU-controls-001", "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(),
        "all_passed": result.wasSuccessful(),
        "elapsed_seconds": time.perf_counter()-started, "device": "cpu", "shape": [16, 16],
        "assertions_enabled": not bool(sys.flags.optimize), "bytecode_disabled": sys.dont_write_bytecode,
        "formal_family_generation": False, "formal_training": False,
        "reserved_confirmatory_or_final_payloads_opened": 0, "GPU_calls": 0,
        "tiny_optimizer_control_updates": 3, "source_sha256": {name: common.sha(common.CODE / name)
            for name in ("signed_gain_denoiser.py", "train.py", "final.py", "tests_tail.py")},
        "checks": [test.id().split(".")[-1] for test in unittest.defaultTestLoader.loadTestsFromTestCase(TailControls)]}
    path = TailControls.report_root / "test_receipt.json"
    common.write_json(path, receipt)
    print(json.dumps({"receipt": str(path), "passed": receipt["passed"], "tests": receipt["tests"]}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
