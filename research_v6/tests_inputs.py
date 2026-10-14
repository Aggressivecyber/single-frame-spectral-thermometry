"""Bounded CPU contract checks. No formal inverse or GPU is executed."""
from __future__ import annotations

import copy
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from . import common

common.configure_runtime()
import numpy as np
import torch

from . import continuum_sources as sources
from . import inputs


class MockPhysics:
    """Small explicitly mocked forward with visible blur count/semantics."""
    def __init__(self):
        self.cfg = copy.deepcopy(sources.frozen_selection()["physics_config"])
        self.cfg["shape"] = [8, 8]
        self.M = torch.zeros((7, 3), dtype=torch.float32)
        self.T0, self.Tscale, self.Sscale = 3000., 1000., 60.
        self.blur_calls = 0
        self.op = SimpleNamespace(object_blur=self.blur)

    def blur(self, value):
        self.blur_calls += 1
        return value*2

    def raw_to_electrons(self, raw):
        return raw*3

    def components(self, state, beta):
        continuum = (state[:, 0:1]+4).expand(-1, 7, -1, -1)
        return continuum, None, None, None


def physical_mock(physics, state, beta, raw_e):
    latent = physics.components(state, beta)[0]
    postblur = physics.op.object_blur(latent)
    expected = raw_e+1
    return {"state": state, "beta": beta, "template_e": state[:, 4]*60,
            "temperature_color_K": state[:, 0]*1000+3000,
            "continuum_images_e": postblur, "expected_e": expected,
            "valid": torch.ones_like(raw_e, dtype=torch.bool)}


class InputsTests(unittest.TestCase):
    fixture_root = None

    @classmethod
    def setUpClass(cls):
        cls.fixture_root = common.ROOT/"outputs/research_v6/implementation_fixtures/inputs"/uuid.uuid4().hex
        common.ed_path(cls.fixture_root).mkdir(parents=True, exist_ok=False)

    def setUp(self):
        self.physics = MockPhysics()
        self.selection = sources.frozen_selection()
        self.raw = torch.full((1, 8, 8), 10., dtype=torch.float32)

    def fixture_ref(self):
        token = uuid.uuid4().hex
        raw = self.raw[0].numpy()
        raw_receipt = common.save_npz(self.fixture_root/(token+"_raw.npz"), raw_dn=raw,
                                      hidden_diagnostic=np.zeros((8, 8), dtype=np.float32))
        label_receipt = common.save_npz(self.fixture_root/(token+"_truth.npz"),
            continuum_latent=np.ones((7, 8, 8), dtype=np.float32), beta=np.ones(3, dtype=np.float32),
            latent_state=np.ones((5, 8, 8), dtype=np.float32),
            temperature_color_K=np.ones((8, 8), dtype=np.float32),
            evaluation_domain=np.ones((8, 8), dtype=bool))
        return {"case_id": token, "mother_id": token, "cohort": "implementation_fixture",
                "fixture_only": True, "condition": "blank", "input_kind": "continuous",
                "raw_reference": {**raw_receipt, "array_sha256": inputs.array_hash(raw)},
                "label_reference": label_receipt}

    def test_measurement_and_diagnostics_are_separate(self):
        ref = self.fixture_ref()
        raw = inputs.load_raw(ref, self.physics)
        self.assertIsInstance(raw, torch.Tensor)
        self.assertEqual(tuple(raw.shape), (1, 8, 8))
        self.assertTrue(torch.equal(raw, self.raw))
        truth = inputs.load_diagnostic_truth(ref, self.physics)
        self.assertNotIn("raw_dn", truth)
        self.assertEqual(tuple(truth["continuum_latent"].shape), (1, 7, 8, 8))
        self.assertEqual(tuple(truth["latent_state"].shape), (1, 5, 8, 8))
        self.assertEqual(tuple(truth["beta"].shape), (1, 3))
        self.assertEqual(truth["evaluation_domain"].dtype, torch.bool)

    def test_sealed_and_unknown_cohorts_rejected_before_any_open(self):
        for cohort in ("signed_confirmation", "final_once", "final", "training", "unregistered"):
            ref = {"cohort": cohort, "raw_reference": {"path": "sealed.npz", "sha256": "a"*64}}
            with patch.object(common, "sha") as hashed, patch("numpy.load") as opened:
                with self.assertRaises(inputs.InputIntegrityError):
                    inputs.load_raw(ref, self.physics)
                hashed.assert_not_called()
                opened.assert_not_called()

    def test_raw_array_hash_and_file_hash_both_required(self):
        ref = self.fixture_ref()
        ref["raw_reference"]["array_sha256"] = "0"*64
        with self.assertRaises(inputs.InputIntegrityError):
            inputs.load_raw(ref, self.physics)
        ref["raw_reference"]["sha256"] = "0"*64
        with self.assertRaises(inputs.InputIntegrityError):
            inputs.load_raw(ref, self.physics)

    def test_integrity_and_path_escape(self):
        ref = self.fixture_ref()
        self.assertTrue(inputs.integrity_receipt([ref], payload_audit=True)["passed"])
        ref["raw_reference"]["path"] = "outputs/research_v5/run_001/signed_confirmation/raw/forbidden.npz"
        with patch.object(common, "sha") as hashed:
            with self.assertRaises(inputs.InputIntegrityError):
                inputs.load_raw(ref, self.physics)
            hashed.assert_not_called()

    def test_frozen_selection_is_old_init_and_rejects_variant(self):
        self.assertEqual(self.selection["method"], "old_init__joint_update_v3")
        bad = copy.deepcopy(self.selection)
        bad["method"] = "consistent_init__joint_update_v3"
        with self.assertRaises(inputs.InputIntegrityError):
            sources._check_selection(self.physics, bad)

    def test_v3_deployment_signature_and_exact_scalar_call(self):
        self.assertEqual(list(inspect.signature(sources.infer_v3).parameters), ["raw_dn", "physics", "selection"])
        captured = {}
        def inverse(physics, raw_e, method, config):
            captured.update(raw_e=raw_e.clone(), method=method, config=config)
            state = raw_e.new_zeros((1, 5, 8, 8))
            state[:, 0] = .25
            state[:, 4] = -2
            return state, raw_e.new_zeros((1, 3)), [{"joint": []}], .01
        with patch.object(sources, "_check_selection", return_value={"contract_test": True}), \
             patch.object(sources, "_v3_functions", return_value=(inverse, physical_mock)):
            result = sources.infer_v3(self.raw, self.physics, self.selection)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(captured["method"], "old_init__joint_update_v3")
        self.assertEqual(captured["config"], self.selection["solver_config"])
        self.assertTrue(torch.equal(captured["raw_e"], self.raw*3))
        self.assertEqual(self.physics.blur_calls, 1)
        self.assertTrue(torch.equal(result["continuum_postblur"], result["continuum_latent"]*2))
        self.assertTrue(torch.equal(result["state"][:, 4]*60, result["baseline_output"]["template_latent_e"]))
        self.assertTrue(torch.equal(result["baseline_output"]["expected_e"], self.raw*3+1))

    def test_source_failures_and_saturated_raw_are_retained(self):
        def failed(*args):
            raise RuntimeError("mock retained source failure")
        with patch.object(sources, "_check_selection", return_value={}), \
             patch.object(sources, "_v3_functions", return_value=(failed, physical_mock)):
            result = sources.infer_v3(self.raw, self.physics, self.selection)
        self.assertEqual(result["status"], "SOURCE_FAILED")
        self.assertIn("mock retained source failure", result["error"])
        self.assertIsNone(result["continuum_latent"])
        with patch.object(sources, "_check_selection", return_value={}), \
             patch.object(sources, "_v3_functions") as importer:
            result = sources.infer_v3(torch.full_like(self.raw, 4095), self.physics, self.selection)
            importer.assert_not_called()
        self.assertEqual(result["status"], "SOURCE_FAILED")

    def test_bad_cache_never_calls_inverse(self):
        with patch.object(sources, "_check_selection", return_value={}), \
             patch.object(sources, "_v3_functions") as importer:
            result = sources.cached_v3(self.raw, self.physics, self.selection, None)
            importer.assert_not_called()
        self.assertEqual(result["status"], "SOURCE_FAILED")
        with patch.object(sources, "_v5_options", return_value={}), \
             patch.object(common, "frozen_v5") as importer:
            result = sources.cached_v5(self.raw, self.physics, None)
            importer.assert_not_called()
        self.assertEqual(result["status"], "SOURCE_FAILED")

    def test_v5_frozen_estimator_uses_electrons_not_diagnostic_mapping(self):
        frozen = {"warm_iterations": 30}
        captured = {}
        def estimate(physics, raw_e, options):
            captured.update(raw_e=raw_e.clone(), options=options)
            return {"continuum_latent": torch.ones((1, 7, 8, 8)),
                    "continuum_postblur": torch.ones((1, 7, 8, 8))*2,
                    "temperature_color_K": torch.ones_like(raw_e)*3000,
                    "valid": torch.ones_like(raw_e, dtype=torch.bool)}
        with patch.object(sources, "_v5_options", return_value=frozen), \
             patch.object(common, "frozen_v5", return_value=SimpleNamespace(estimate_continuum=estimate)) as importer:
            result = sources.infer_v5(self.raw, self.physics)
        self.assertEqual(result["status"], "SUCCESS")
        importer.assert_called_once_with("continuum_solver")
        self.assertEqual(captured["options"], frozen)
        self.assertTrue(torch.equal(captured["raw_e"], self.raw*3))
        rejected = sources.infer_v5({"raw_dn": self.raw, "truth": {}}, self.physics)
        self.assertEqual(rejected["status"], "SOURCE_FAILED")

    def test_successful_v5_cache_float32_and_uint16_keep_original_hash(self):
        for dtype in (np.float32, np.uint16):
            original_raw = self.raw[0].numpy().astype(dtype)
            latent = np.ones((1, 7, 8, 8), dtype=np.float32)*4
            saved = common.save_npz(self.fixture_root/(uuid.uuid4().hex+"_v5.npz"),
                continuum_latent=latent, continuum_postblur=latent*2,
                temperature_color_K=np.ones((1, 8, 8), dtype=np.float32)*3000,
                continuum_valid=np.ones((1, 8, 8), dtype=bool))
            receipt = {**saved, "cache_kind": "frozen_v5_development_latent",
                       "raw_dtype": str(original_raw.dtype), "raw_shape": [8, 8],
                       "raw_array_sha256": inputs.array_hash(original_raw)}
            # Only the fixture location is injected. Hash, geometry, raw dtype
            # roundtrip and latent/postblur checks all remain the real code.
            with patch.object(sources, "_v5_options", return_value={}), \
                 patch.object(sources, "_path_in", side_effect=lambda path, parent: common.ROOT/path), \
                 patch.object(common, "frozen_v5") as inverse:
                result = sources.cached_v5(self.raw, self.physics, receipt)
            self.assertEqual(result["status"], "SUCCESS", result.get("error"))
            self.assertEqual(result["source_metadata"]["raw_array_sha256"], inputs.array_hash(original_raw))
            self.assertTrue(result["source_metadata"]["original_dtype_roundtrip_verified"])
            inverse.assert_not_called()
            if dtype == np.uint16:
                changed = self.raw.clone()
                changed[0, 0, 0] += .25
                with patch.object(sources, "_v5_options", return_value={}):
                    failed = sources.cached_v5(changed, self.physics, receipt)
                self.assertEqual(failed["status"], "SOURCE_FAILED")
                self.assertIn("lossless", failed["error"])

    def test_successful_v3_state_beta_cache_never_calls_inverse(self):
        key = uuid.uuid4().hex*2
        state = np.zeros((5, 8, 8), dtype=np.float32)
        state[0] = .125
        state[4] = -1
        saved = common.save_npz(self.fixture_root/(key+".npz"), state=state, beta=np.zeros(3, dtype=np.float32))
        provenance = {"method": self.selection["method"],
                      "physics_config_sha256": common.canonical_hash(self.physics.cfg),
                      "solver_config_sha256": common.canonical_hash(self.selection["solver_config"])}
        raw_hash = inputs.array_hash(self.raw[0].numpy())
        meta = {"cache_key": key, "cache_sha256": saved["sha256"], "raw_hash": raw_hash,
                "method": provenance["method"], "physics_config_sha256": provenance["physics_config_sha256"],
                "calibration_sha256": provenance["physics_config_sha256"],
                "solver_config_sha256": provenance["solver_config_sha256"],
                "engine": "single_frame_cached_Jacobian_v3_math", "labels_used": False,
                "seconds": .123, "history": {"joint": []}}
        metadata_path = self.fixture_root/(key+".json")
        common.write_json(metadata_path, meta)
        receipt = {**saved, "metadata_path": common.relative(metadata_path),
                   "metadata_sha256": common.sha(metadata_path), "legacy_metadata": meta,
                   "cache_kind": "frozen_v4_state_beta", "raw_array_sha256": raw_hash,
                   "raw_dtype": "float32", "raw_shape": [8, 8]}
        inverse = unittest.mock.Mock(side_effect=AssertionError("inverse must not run"))
        with patch.object(sources, "_check_selection", return_value=provenance), \
             patch.object(sources, "_path_in", side_effect=lambda path, parent: common.ROOT/path), \
             patch.object(sources, "_v3_functions", return_value=(inverse, physical_mock)):
            result = sources.cached_v3(self.raw, self.physics, self.selection, receipt)
        self.assertEqual(result["status"], "SUCCESS", result.get("error"))
        inverse.assert_not_called()
        self.assertEqual(self.physics.blur_calls, 1)
        self.assertTrue(torch.equal(result["state"], torch.from_numpy(state)[None]))
        self.assertEqual(result["source_metadata"]["original_source_seconds"], .123)

    def test_original16_metadata_references_only(self):
        with patch("numpy.load") as opened:
            refs = inputs.regression16_references()
            opened.assert_not_called()
        self.assertEqual(len(refs), 16)
        self.assertEqual(len({r["case_id"] for r in refs}), 16)

    def test_frozen_imports_do_not_initialize_v5_or_instantiate_models(self):
        import sys
        before = set(sys.modules)
        inverse, output = sources._v3_functions()
        self.assertEqual(inverse.__module__, "training_v4.scalar_engine")
        self.assertEqual(output.__module__, "training_v4.model")
        frozen = common.frozen_v5("continuum_solver")
        self.assertEqual(frozen.__name__, "_v6_frozen_v5_continuum_solver")
        self.assertNotIn("research_v5", set(sys.modules)-before)
        self.assertNotIn("research_v5.common", set(sys.modules)-before)


def run_contract_checks():
    folder = common.ROOT/"reports/research_v6/implementation_checks/inputs"/uuid.uuid4().hex
    common.ed_path(folder).mkdir(parents=True, exist_ok=False)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InputsTests))
    (folder/"unittest_output.txt").write_text(stream.getvalue(), encoding="utf-8")
    receipt = {"schema": "v6-inputs-sources-cpu-contract-001", "passed": result.wasSuccessful(),
               "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "scope": "small8x8 CPU mocked inverses; original16 metadata only",
               "formal_scientific_inversions": 0, "GPU_used": False, "sealed_payloads_opened": False,
               "fixture_directory": common.relative(InputsTests.fixture_root),
               "source_sha256": {common.relative(common.CODE/name): common.sha(common.CODE/name)
                                 for name in ("inputs.py", "continuum_sources.py", "tests_inputs.py")}}
    common.write_json(folder/"test_receipt.json", receipt)
    print(stream.getvalue(), end="")
    print(json.dumps({"receipt": common.relative(folder/"test_receipt.json"), "passed": receipt["passed"]}))
    return result.wasSuccessful()


if __name__ == "__main__":
    raise SystemExit(0 if run_contract_checks() else 1)
