"""Small CPU-only contract tests; no formal arrays, cache jobs or training.

All generated 8x8 fixtures are visibly marked CPU_MOCK_TEST and retained under
training_v4_6/_test_runs. This module never registers an execution protocol.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
import copy
import json
import os
import subprocess
import sys
import unittest
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
from training_v4_6 import data, cache

TEST_ROOT = data.ROOT / "training_v4_6/_test_runs" / (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data.require(TEST_ROOT.drive.upper() in ("E:", "D:"), "CPU fixtures require E/D storage")
        data.configure_runtime(TEST_ROOT / "runtime")
        import numpy as np
        import torch
        torch.set_num_threads(1)
        cls.np, cls.torch = np, torch
        cls.bundle = copy.deepcopy(data.load_bundle())
        cls.bundle["old"]["physics_config"]["shape"] = [8, 8]
        cfg = copy.deepcopy(cls.bundle["old"]["physics_config"])

        class MockOp:
            @staticmethod
            def object_blur(x):
                return x

            @staticmethod
            def encode_images(x):
                return x.mean(dim=1)

        class MockPhysics:
            def __init__(self):
                self.cfg, self.M, self.op = cfg, torch.tensor(0.), MockOp()
                self.T0, self.Tscale, self.Sscale = 1200., 500., 1000.
                self.adc_step = 16.
                self.extra_variance = torch.tensor(float(cfg["read_noise_e"]) ** 2)

            def components(self, z, beta):
                c = (z[:, 1].exp() * 2000 + z[:, 0] * 2)[:, None].repeat(1, 7, 1, 1)
                s = (z[:, 4] * self.Sscale)[:, None].repeat(1, 7, 1, 1)
                b = beta.mean(dim=1)[:, None, None, None].expand_as(c) * 10
                return c, s, b, None

            def __call__(self, z, beta):
                c, s, b, _ = self.components(z, beta)
                return self.op.encode_images(c + s + b)

            @staticmethod
            def color_temperature(c):
                v = c.mean(dim=1) + 1000
                return v, torch.ones_like(v, dtype=torch.bool)

            def raw_to_electrons(self, raw):
                return (raw - self.cfg["black_level_DN"]) * self.adc_step

        cls.physics = MockPhysics()

        def mother_function(p, descriptor):
            gen = torch.Generator(device="cpu").manual_seed(descriptor["seed"])
            z = torch.rand((1, 5, 8, 8), generator=gen) * .2
            beta = torch.rand((1, 2), generator=gen)
            shape = torch.rand((1, 8, 8), generator=gen) * .8 + .2
            shape /= shape.max()
            original = {"temperature_field_sha256": data.array_hash((p.T0 + p.Tscale * z[0, 0]).numpy()),
                        "amplitude_field_sha256": data.array_hash(z[0, 1].exp().numpy()),
                        "template_shape_sha256": data.array_hash(shape.numpy())}
            return z, beta, shape, original

        cls.mother_function = staticmethod(mother_function)
        cls.fixture_root = TEST_ROOT / "success/CPU_MOCK_TEST"
        cls.output = cls.fixture_root / "reports/training_v4_6/run_001"
        cls.data_dir = cls.fixture_root / "outputs/training_v4_6/run_001"
        old_signed = [r for r in cls.bundle["old"]["rows"]
                      if r["split"] == "train" and r["signed_aux"] and r["input_kind"] == "noise"]
        replacements = {}
        for index, row in enumerate(old_signed):
            rng = np.random.default_rng(1000 + index)
            expected = (1800 + rng.random((8, 8)) * 900).astype(np.float32)
            truth = {"expected_e": expected, "template_e": np.full((8, 8), data.PEAKS[row["condition"]], np.float32),
                     "temperature_color_K": expected + 1000, "band_images_e": np.repeat(expected[None], 7, axis=0),
                     "evaluation_domain": np.ones((8, 8), bool)}
            info = data.save_arrays(cls.fixture_root / row["label_path"], truth)
            replacements[row["label_path"]] = info["file_sha256"]
        for row in cls.bundle["old"]["rows"]:
            if row["label_path"] in replacements:
                row["label_sha256"] = replacements[row["label_path"]]
        cls.inherited_rows = copy.deepcopy(cls.bundle["old"]["rows"])
        cls.array_reads = []
        original_read = data.read_arrays

        def guarded_read(path):
            path = Path(path).resolve()
            if not path.is_relative_to(cls.fixture_root.resolve()):
                raise AssertionError("CPU generation attempted array access outside its mock root")
            cls.array_reads.append(path)
            return original_read(path)

        with mock.patch.object(data, "read_arrays", guarded_read):
            cls.generated = data._generate_train_rows(root=cls.fixture_root, output=cls.output,
                data_dir=cls.data_dir, bundle=cls.bundle, physics=cls.physics,
                mother_function=mother_function, mode="CPU_MOCK_TEST", progress=False)
        cls.new_rows = cls.generated["new_rows"]

    def test_01_metadata_registry_and_sampler(self):
        from training_v4_6.protocol import planned_sampling
        registry = data.check_registry()
        self.assertEqual(registry["array_access"], 0)
        sampling = planned_sampling()
        selected = cache.selected_new_case_ids(sampling["draws"], self.new_rows)
        self.assertEqual(len(selected), 120)
        self.assertEqual(set(selected), set(sampling["new_unique_cases_to_cache"]))
        selected_rows = {r["case_id"]: r for r in self.new_rows if r["case_id"] in selected}
        self.assertTrue(all(r["condition"].startswith("negative") and r["input_kind"] == "noise"
                            for r in selected_rows.values()))
        self.assertTrue(sampling["all_other_1487_case_ids_unchanged"])
        self.assertEqual(sampling["negative_noisy_positions_replaced"], 513)

    def test_02_counts_original_prefix_and_scoped_old_reads(self):
        manifest = self.generated["manifest"]
        self.assertEqual(manifest["rows"][:3352], self.inherited_rows)
        self.assertEqual(Counter(r["split"] for r in manifest["rows"]), {"train": 3008, "validation": 776})
        self.assertEqual(len(self.new_rows), 432)
        self.assertEqual(len(self.generated["new_mothers"]), 24)
        self.assertEqual(len(list((self.data_dir / "raw").rglob("*.npz"))), 432)
        self.assertEqual(len(list((self.data_dir / "labels").rglob("*.npz"))), 96)
        self.assertEqual(len(self.array_reads), 48)
        self.assertTrue(all("v4_1_signed_train_mother_" in str(p) for p in self.array_reads))
        self.assertTrue(all("validation" not in str(p) for p in self.array_reads))
        self.assertTrue(all(r["array_reopen_exact"] for r in self.new_rows))

    def test_03_realization_hashes_and_geometry_coverage(self):
        noisy = [r for r in self.new_rows if r["input_kind"] == "noise"]
        self.assertEqual(len({r["noise_seed"] for r in noisy}), 240)
        self.assertEqual(len({r["raw_hash"] for r in self.new_rows}), 432)
        self.assertEqual(len({r["raw_file_sha256"] for r in self.new_rows}), 432)
        mothers = [m for m in self.generated["manifest"]["signed_mothers"] if m["split"] == "train"]
        self.assertEqual(len({(m["temperature_geometry"], m["template_geometry"]) for m in mothers}), 36)
        by = {r["case_id"]: r for r in self.new_rows}
        for mother in self.generated["new_mothers"]:
            for condition in data.CONDITIONS:
                a, b = [by[mother["mother_id"] + "_" + condition + "_noise" + str(i)] for i in (0, 1)]
                self.assertEqual(a["label_sha256"], b["label_sha256"])
                self.assertNotEqual(a["noise_seed"], b["noise_seed"])
                self.assertNotEqual(a["raw_hash"], b["raw_hash"])

    def test_04_nonfinite_retains_seed_and_failed_values(self):
        failed_root = TEST_ROOT / "failure/CPU_MOCK_TEST"

        def invalid(p, mother):
            z, beta, shape, original = self.mother_function(p, mother)
            z[0, 0, 0, 0] = float("nan")
            return z, beta, shape, original

        output = failed_root / "reports/training_v4_6/run_001"
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            data._generate_train_rows(root=failed_root, output=output,
                data_dir=failed_root / "outputs/training_v4_6/run_001", bundle=self.bundle,
                physics=self.physics, mother_function=invalid, mode="CPU_MOCK_TEST", progress=False)
        failure = data.read(output / "generation_failure.json")
        self.assertEqual(failure["descriptor"]["seed"], self.bundle["train_parents"][0]["seed"])
        self.assertFalse(failure["failed_seed_replaced"])
        self.assertFalse(failure["failed_files_deleted"])
        self.assertEqual(failure["completed_new_rows"], 0)
        values = data.read_arrays(failed_root / failure["failed_values_path"])
        self.assertTrue(self.np.isnan(values["state"]).any())

    def test_05_raw_only_cache_then_label_score_and_metadata_guards(self):
        torch = self.torch
        selection = {"physics_config": self.physics.cfg, "method": "CPU_MOCK_TEST", "solver_config": {"iterations": 0}}
        rows = [next(r for r in self.new_rows if r["input_kind"] == "noise" and not r["existing_label_reused"]),
                next(r for r in self.new_rows if r["existing_label_reused"])]
        original_read = data.read_arrays
        for row in rows:
            array_reads = []
            label_path = (self.fixture_root / row["label_path"]).resolve()

            def read_guard(path):
                path = Path(path).resolve()
                self.assertTrue(path.is_relative_to(self.fixture_root.resolve()))
                array_reads.append(path)
                return original_read(path)

            def inverse(p, raw_e, method, config):
                self.assertNotIn(label_path, array_reads)
                self.assertEqual(method, "CPU_MOCK_TEST")
                self.assertEqual(config, {"iterations": 0})
                self.assertEqual(tuple(raw_e.shape), (1, 8, 8))
                return torch.zeros((1, 5, 8, 8)), torch.zeros((1, 2)), [[{"mock": True}]], .001

            def output_fn(p, state, beta, raw_e):
                return {"template_e": torch.zeros((1, 8, 8)), "temperature_color_K": torch.zeros((1, 8, 8)),
                        "raw_residual_e": torch.zeros_like(raw_e)}

            key_fn = lambda r, s: data.canonical_hash({"CPU_MOCK_TEST": r["raw_hash"]})
            with mock.patch.object(data, "read_arrays", read_guard):
                record = cache._cache_one(self.physics, row, selection, root=self.fixture_root,
                    data_dir=self.data_dir, inverse=inverse, output_fn=output_fn, key_fn=key_fn)
            self.assertEqual(array_reads[-1], label_path)
            self.assertFalse(record["labels_enter_inversion"])
            path = self.fixture_root / record["cache_path"]
            meta = data.read(path.with_suffix(".json"))
            score = data.read(self.fixture_root / record["score_path"])
            cache._verify_cache_identity(row, selection, record["cache_key"], path, meta, score)
            bad_score = {**score, "cache_metadata_sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "metadata drift"):
                cache._verify_cache_identity(row, selection, record["cache_key"], path, meta, bad_score)
            for field in ("physics_config_sha256", "calibration_sha256", "solver_config_sha256", "method", "engine_sha256"):
                with self.assertRaisesRegex(ValueError, "configuration drift"):
                    cache._verify_cache_identity(row, selection, record["cache_key"], path, {**meta, field: "changed"}, score)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                cache._verify_cache_identity(row, selection, record["cache_key"], path, {**meta, "raw_hash": "changed"}, score)

    def test_06_missing_execution_protocol_rejected_without_runtime_import(self):
        code = ("import json,sys; from pathlib import Path; from training_v4_6 import data,cache; "
                "data.EXECUTION_PROTOCOL=Path(sys.argv[1]); "
                "exec('try:\\n data.generate(protocol_sha256=\"0\"*64)\\nexcept ValueError as e:\\n error=str(e)\\nelse:\\n raise AssertionError(\"gate unexpectedly opened\")'); "
                "assert 'torch' not in sys.modules and 'runtime' not in sys.modules; "
                "assert not data.EXECUTION_PROTOCOL.parent.exists(); "
                "print(json.dumps({'gate_error':error,'torch_imported':False,'runtime_imported':False,'writes':0}))")
        missing = TEST_ROOT / "must_remain_absent/execution_protocol.json"
        proc = subprocess.run([sys.executable, "-B", "-c", code, str(missing)], cwd=data.ROOT,
                              env=os.environ.copy(), text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Root must register", json.loads(proc.stdout)["gate_error"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {"status": "PASS" if result.wasSuccessful() else "FAIL", "mode": "CPU_MOCK_TEST",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "device": "cpu", "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "cpu_threads": 1,
        "fixture_shape": [8, 8], "formal_data_generation": False, "formal_cache_materialization": False,
        "optimizer_updates": 0, "locked_development_arrays_opened": 0, "final_test_arrays_opened": 0,
        "test_root": str(TEST_ROOT), "time": data.now(),
        "source_sha256": {p.name: data.file_hash(p) for p in (Path(__file__), data.ROOT / "training_v4_6/data.py",
                                                           data.ROOT / "training_v4_6/cache.py")}}
    data.write_json(TEST_ROOT / "cpu_contract_test_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    sys.exit(0 if result.wasSuccessful() else 1)
