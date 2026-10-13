"""Bounded CPU fixtures for diagnostic lineage and the oracle truth boundary.

Run with assertions enabled: ``python -B -m research_v5.tests_data_oracle``.
All fixture arrays, runtime/cache directories and the receipt stay in the owned
``reports/research_v5/implementation_checks/data_oracle`` directory.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "reports" / "research_v5" / "implementation_checks" / "data_oracle"


def _prepare() -> Path:
    if ROOT.drive.upper() not in ("E:", "D:"):
        raise RuntimeError("CPU implementation fixtures must be run from the E/D checkout")
    run = CHECKS / ("cpu_fixture_" + uuid4().hex)
    runtime = run / "runtime"
    for name in ("tmp", "torch", "cache", "mpl", "cuda"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["THERMOMETRY_RUNTIME"] = str(runtime)
    for key in ("TMP", "TEMP", "TMPDIR"):
        os.environ[key] = str(runtime / "tmp")
    os.environ["TORCH_HOME"] = str(runtime / "torch")
    os.environ["XDG_CACHE_HOME"] = str(runtime / "cache")
    os.environ["MPLCONFIGDIR"] = str(runtime / "mpl")
    os.environ["CUDA_CACHE_PATH"] = str(runtime / "cuda")
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    return run


class DataOracleContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np
        import torch
        from config import default_config
        from physics import SpectralPhysics
        from research_v5 import data
        torch.set_num_threads(2)
        cls.np, cls.torch, cls.data = np, torch, data
        cls.config = default_config(24)
        cls.physics = SpectralPhysics(cls.config, dtype=torch.float64)
        cls.old_path = ROOT / "reports" / "training_v4" / "run_001" / "data_manifest.json"
        cls.plan = data.build_diagnostic_plan(cls.old_path)
        cls.fixture_plan = cls.plan[:15]
        cls.fixture = data.generate_diagnostics(cls.physics, cls.fixture_plan,
                    RUN / "matched", {"fixture_only": True})
        with np.load(RUN / "matched" / cls.fixture["labels"][0]["path"], allow_pickle=False) as handle:
            cls.truth = {key: handle[key].copy() for key in handle.files}
        with np.load(RUN / "matched" / cls.fixture["rows"][0]["raw_path"], allow_pickle=False) as handle:
            cls.raw = handle["raw_dn"].copy()

    def test_01_plan_is_stable_balanced_opened_and_noise_is_distinct(self) -> None:
        plan = self.plan
        self.assertEqual(len(plan), 180)
        self.assertEqual(plan, self.data.build_diagnostic_plan(self.old_path))
        mothers = {d["mother_id"]: d["mother"] for d in plan}
        self.assertEqual(len(mothers), 12)
        for geometry in self.data.TEMPERATURE_GEOMETRIES:
            self.assertEqual(sum(m["temperature_geometry"] == geometry for m in mothers.values()), 2)
        self.assertTrue(all(m["split"] == "validation" for m in mothers.values()))
        self.assertTrue(all("final" not in d["mother_id"] for d in plan))
        noisy = [d for d in plan if d["input_kind"] == "noise"]
        self.assertEqual(len({d["noise_seed"] for d in noisy}), 60)
        self.assertTrue(all(not d["same_noise_realization_claimed"] for d in plan))
        for mother_id in mothers:
            subset = [d for d in plan if d["mother_id"] == mother_id]
            self.assertEqual({d["coefficient_e"] for d in subset}, set(self.data.COEFFICIENTS_E))
            self.assertEqual(len(subset), 15)

    def test_02_raw_is_only_dn_truth_is_separate_and_fields_match(self) -> None:
        np = self.np
        manifest = self.fixture
        self.assertEqual(manifest["planned_rows"], 15)
        self.assertEqual(manifest["generated_raw_rows"], 15)
        self.assertTrue(manifest["matched_continuum_background_shape_verified"])
        protected = {json.dumps(label["protected_field_hashes"], sort_keys=True) for label in manifest["labels"]}
        self.assertEqual(len(protected), 1)
        for row in manifest["rows"]:
            with np.load(RUN / "matched" / row["raw_path"], allow_pickle=False) as handle:
                self.assertEqual(handle.files, ["raw_dn"])
                raw = handle["raw_dn"]
                self.assertEqual(raw.shape, (24, 24))
                if row["input_kind"] != "continuous":
                    self.assertEqual(raw.dtype, np.dtype("uint16"))
            self.assertTrue(row["label_path"].startswith("diagnostic_labels/"))
            self.assertFalse(row["expectation_diagnostics"]["mother_fields_modified_for_legality"])
        self.assertFalse(manifest["final_test_opened"])
        self.assertFalse(manifest["confirmation_test_opened"])
        self.assertFalse(manifest["labels_accepted_by_deployment"])

    def test_03_seed_replay_and_original_files_are_immutable(self) -> None:
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in IMMUTABLE}
        replay = self.data.generate_diagnostics(self.physics, self.fixture_plan,
                    RUN / "replayed", {"fixture_only": True})
        self.assertEqual([r["raw_hash"] for r in replay["rows"]], [r["raw_hash"] for r in self.fixture["rows"]])
        after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in IMMUTABLE}
        self.assertEqual(before, after)
        with self.assertRaises(FileExistsError):
            self.data.generate_diagnostics(self.physics, self.fixture_plan, RUN / "matched", {"fixture_only": True})

    def test_04_small_grid_requires_explicit_fixture_and_c_output_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.data.generate_diagnostics(self.physics, self.plan, RUN / "not_fixture")
        with self.assertRaises(ValueError):
            self.data.generate_diagnostics(self.physics, self.fixture_plan,
                        Path("C:/must_not_write_research_v5"), {"fixture_only": True})

    def _artificial_mother(self, amplitude: float):
        torch = self.torch
        p = self.physics
        state = torch.zeros((1, 5, 24, 24), dtype=p.M.dtype)
        state[:, 0] = (2800.0 - p.T0) / p.Tscale
        state[:, 1] = amplitude
        beta = torch.zeros((1, 3), dtype=p.M.dtype)
        shape = torch.ones((24, 24), dtype=p.M.dtype)
        return state, beta, shape, {"fixture_artificial_mother": True}

    def test_05_illegal_poisson_mean_is_not_repaired(self) -> None:
        descriptors = [d for d in self.fixture_plan if d["coefficient_e"] == -180.0]
        original_state, original_beta, shape, provenance = self._artificial_mother(-8.0)
        with patch("training_v4.data.mother_state", return_value=(original_state, original_beta, shape, provenance)):
            manifest = self.data.generate_diagnostics(self.physics, descriptors,
                        RUN / "negative_expectation", {"fixture_only": True})
        self.assertEqual(manifest["planned_rows"], 3)
        self.assertEqual(manifest["generated_raw_rows"], 2)
        noisy = next(row for row in manifest["rows"] if row["input_kind"] == "noise")
        self.assertEqual(noisy["raw_generation_status"], "UNAVAILABLE_NEGATIVE_POISSON_MEAN")
        self.assertIsNone(noisy["raw_path"])
        self.assertGreater(noisy["expectation_diagnostics"]["negative_expected_pixels"], 0)
        with self.np.load(RUN / "negative_expectation" / manifest["labels"][0]["path"], allow_pickle=False) as handle:
            self.np.testing.assert_array_equal(handle["latent_state"][:4], original_state[0, :4].numpy())
            self.np.testing.assert_array_equal(handle["beta"], original_beta[0].numpy())
            self.assertEqual(float(handle["coefficient_e"]), -180.0)
        self.assertFalse(manifest["negative_concentration_claimed"])

    def test_06_adc_saturation_is_retained_without_scaling(self) -> None:
        descriptors = [d for d in self.fixture_plan if d["coefficient_e"] == 180.0 and d["input_kind"] != "noise"]
        original_state, original_beta, shape, provenance = self._artificial_mother(5.0)
        with patch("training_v4.data.mother_state", return_value=(original_state, original_beta, shape, provenance)):
            manifest = self.data.generate_diagnostics(self.physics, descriptors,
                        RUN / "saturation", {"fixture_only": True})
        quant = next(row for row in manifest["rows"] if row["input_kind"] == "quantization")
        self.assertGreater(quant["ADC_diagnostics"]["raw_saturated_pixels"], 0)
        self.assertGreater(quant["ADC_diagnostics"]["rounded_high_clip_pixels"], 0)
        with self.np.load(RUN / "saturation" / manifest["labels"][0]["path"], allow_pickle=False) as handle:
            self.np.testing.assert_array_equal(handle["latent_log_amplitude"], original_state[0, 1].numpy())

    def _mock_numerics(self, events: list):
        torch = self.torch
        continuum_module = ModuleType("research_v5.continuum_solver")
        linear_module = ModuleType("research_v5.signed_linear_solver")

        def estimate(p, raw_e, options=None):
            events.append(("estimate", id(raw_e), copy.deepcopy(options)))
            state = torch.zeros((1, 5, 24, 24), dtype=p.M.dtype)
            state[:, 1] = -0.5
            beta = torch.zeros((1, 3), dtype=p.M.dtype)
            C = p.components(state, beta)[0]
            postblur = p.op.object_blur(C)
            color, valid = p.color_temperature(postblur)
            return {"state": state, "continuum_latent": C, "continuum_postblur": postblur,
                    "temperature_color_K": color, "valid": valid, "info": {"fixture_mock": True}}

        def solve(p, raw_e, C, options=None, *, fixed_beta=None):
            events.append(("solve", id(raw_e), C.clone(), None if fixed_beta is None else fixed_beta.clone(), copy.deepcopy(options)))
            beta = torch.zeros((1, 3), dtype=p.M.dtype) if fixed_beta is None else fixed_beta.clone()
            bg = p.Bscale * torch.einsum("bl,lhw->bhw", beta, p.background_basis)
            expected = p.op(C + p.q[None, :, None, None] * bg[:, None])
            template = torch.full((1, 24, 24), 3.0, dtype=p.M.dtype)
            weights = (raw_e.clamp_min(0) + p.extra_variance).reciprocal()
            return {"template_latent_e": template, "template_postblur_e": template,
                    "beta": beta, "expected_e": expected, "weights": weights,
                    "info": {"frames": [{"accepted": True,
                      "objective_history": [{"iteration": i, "objective": value} for i, value in enumerate((4.0, 2.0, 1.0, 1.2))],
                      "CG": {"iterations": 3, "relative_true_residual": 0.3,
                             "converged": False, "stop": "fixture",
                             "true_residual_history": [1.0, 0.5, 0.2, 0.3]}}]}}

        continuum_module.estimate_continuum = estimate
        linear_module.solve_signed_linear = solve
        return patch.dict(sys.modules, {continuum_module.__name__: continuum_module,
                                       linear_module.__name__: linear_module})

    def test_07_o3_precedes_truth_and_all_levels_share_raw_solver_options(self) -> None:
        from research_v5 import oracle
        events: list = []

        class WatchedTruth(dict):
            def __getitem__(self, key):
                events.append(("truth", key))
                return super().__getitem__(key)

        with self._mock_numerics(events):
            result = oracle.run_oracle_case(self.physics, self.raw, WatchedTruth(self.truth))
        self.assertEqual(events[0][0], "estimate")
        self.assertEqual(events[1][0], "solve")
        self.assertIsNone(events[1][3])
        self.assertEqual(events[2], ("truth", "continuum_latent"))
        calls = [event for event in events if event[0] == "solve"]
        self.assertEqual(len(calls), 4)
        self.assertEqual(len({event[1] for event in calls}), 1)
        self.assertTrue(all(event[4] == oracle.LADDER_DEFAULTS for event in calls))
        self.torch.testing.assert_close(calls[0][2], calls[2][2])
        self.assertIsNone(calls[3][3])
        self.assertEqual([row["level"] for row in result["rows"]], ["O0", "O1", "O2", "O3"])
        self.assertTrue(result["same_raw_and_weights_for_all_levels"])
        self.assertFalse(result["rows"][3]["diagnostic_oracle"])
        json.dumps(result["rows"], allow_nan=False)
        self.assertEqual(result["rows"][3]["CG"]["best_iterate_diagnostic"]["best"]["iteration"], 2)
        self.assertFalse(result["rows"][3]["CG"]["best_iterate_diagnostic"]["output_selected_by_this_diagnostic"])
        self.torch.testing.assert_close(result["o3"]["template_postblur_e"], self.torch.full((1, 24, 24), 3.0, dtype=self.physics.M.dtype))

    def test_08_o3_is_invariant_to_truth_and_supplied_oracle_continuum(self) -> None:
        from research_v5 import oracle
        changed = copy.deepcopy(self.truth)
        changed["continuum_latent"] *= 2
        changed["beta"] += 1
        changed["template_postblur_e"] += 90
        changed["temperature_color_K"] += 100
        with self._mock_numerics([]):
            first = oracle.run_oracle_case(self.physics, self.raw, self.truth)
            second = oracle.run_oracle_case(self.physics, self.raw, changed,
                             {"continuum_latent": changed["continuum_latent"]})
        for key in ("template_latent_e", "template_postblur_e", "beta", "expected_e", "continuum_latent"):
            self.torch.testing.assert_close(first["o3"][key], second["o3"][key], rtol=0, atol=0)
        self.assertFalse(second["provided_continuum_consistency"]["supplied_estimate_used_for_solver"])
        self.assertGreater(second["provided_continuum_consistency"]["continuum_max_abs_difference_e"], 0)

    def test_09_ladder_rejects_metadata_and_numerical_tuning(self) -> None:
        from research_v5 import oracle
        with self._mock_numerics([]):
            with self.assertRaises(ValueError):
                oracle.run_oracle_case(self.physics, self.raw, self.truth, options={"condition": "blank"})
            with self.assertRaises(ValueError):
                oracle.run_oracle_case(self.physics, self.raw, self.truth, options={"linear": {"alpha_g": 0.1}})
            with self.assertRaises(TypeError):
                oracle.run_oracle_case(self.physics, {"raw_dn": self.raw, "truth": self.truth}, self.truth)

    def test_10_static_sources_and_original_metadata_remain_valid(self) -> None:
        for name in ("data.py", "oracle.py", "tests_data_oracle.py"):
            path = ROOT / "research_v5" / name
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.assertEqual({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in IMMUTABLE}, ORIGINAL_HASHES)

    def test_11_real_cpu_solver_and_continuum_link(self) -> None:
        from research_v5.oracle import run_oracle_case
        row = next(row for row in self.fixture["rows"] if row["coefficient_e"] == -180.0 and row["input_kind"] == "continuous")
        with self.np.load(RUN / "matched" / row["raw_path"], allow_pickle=False) as handle:
            raw = handle["raw_dn"].copy()
        with self.np.load(RUN / "matched" / row["label_path"], allow_pickle=False) as handle:
            truth = {key: handle[key].copy() for key in handle.files}
        result = run_oracle_case(self.physics, raw, truth,
                                options={"continuum": {"warm_iterations": 2, "cg_max_iterations": 30}})
        self.assertEqual(len(result["rows"]), 4)
        self.assertTrue(all(row["status"] == "RETURNED" for row in result["rows"]))
        self.assertTrue(result["same_raw_and_weights_for_all_levels"])
        self.assertTrue(all(row["metrics"]["all_template_pixels_finite"] for row in result["rows"]))
        self.assertTrue(all(row["CG"]["relative_true_residual"] is not None for row in result["rows"]))
        self.assertTrue(all(row["CG"]["best_iterate_diagnostic"]["best"] is not None for row in result["rows"]))
        self.assertLess(result["rows"][0]["metrics"]["template_RMSE_e"], 0.5)
        self.assertLess(result["rows"][0]["metrics"]["template_L2"], 0.02)
        (RUN / "real_oracle_link.json").write_text(json.dumps(result["rows"], indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    global RUN, IMMUTABLE, ORIGINAL_HASHES
    RUN = _prepare()
    IMMUTABLE = [ROOT / "physics.py", ROOT / "config.py", ROOT / "training_v4" / "data.py",
                 ROOT / "training_v4_1" / "data.py", ROOT / "reports" / "training_v4" / "run_001" / "data_manifest.json"]
    ORIGINAL_HASHES = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in IMMUTABLE}
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DataOracleContracts)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    transcript = output.getvalue()
    (RUN / "unittest_output.txt").write_text(transcript, encoding="utf-8")
    receipt = {"schema": "v5-data-oracle-implementation-check-v1", "device": "CPU", "shape": [24, 24],
               "formal_generation_executed": False, "training_executed": False,
               "tests_run": result.testsRun, "errors": len(result.errors), "failures": len(result.failures),
               "success": result.wasSuccessful(), "run_directory": str(RUN),
               "checks_scope": "stdlib registration, original-field preservation, signed/ADC stress, raw-only artifacts, mocked truth boundary and real CPU oracle link",
               "fixture_only_not_scientific_acceptance": True,
               "original_hashes": ORIGINAL_HASHES,
               "source_hashes": {name: hashlib.sha256((ROOT / "research_v5" / name).read_bytes()).hexdigest()
                                 for name in ("data.py", "oracle.py", "tests_data_oracle.py")}}
    (RUN / "test_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(transcript, end="")
    print(json.dumps({key: receipt[key] for key in ("success", "tests_run", "errors", "failures", "run_directory")}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
