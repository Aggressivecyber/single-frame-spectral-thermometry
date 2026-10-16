"""Reopen delivered observations, replay representative decodes, and plot results."""
import runtime
from pathlib import Path
from collections import Counter
import json
import numpy as np
import torch
from research_filter_decode.operator import CodedOperator

torch.set_num_threads(2)
root = Path(__file__).resolve().parents[1]
reviews = []
for run in ["filter_decode_001", "filter_decode_002", "filter_decode_003"]:
    out = root/"reports"/run
    src = out/"source_snapshot"
    protocol = json.loads((src/"research_filter_decode/protocol.json").read_text())
    manifest = json.loads((out/"REMOTE_FILE_MANIFEST.json").read_text())["files"]
    for name, expected in manifest.items():
        if name.startswith("reports/"):
            local = root/name
        elif name == "source.tar.gz":
            local = root/f"outputs/filter_decode_transfer/{run}.tar.gz"
        elif name == "SERVER_PREFLIGHT.json":
            local = out/name
        else:
            local = src/name
        if runtime.sha256(local) != expected:
            raise RuntimeError(f"Transferred file hash mismatch: {local}")
    metrics = json.loads((out/"METRICS.json").read_text())
    result = json.loads((out/"RESULTS.json").read_text())
    ledger = [json.loads(line) for line in (out/"decode_calls.jsonl").read_text().splitlines()]
    assert [r["call"] for r in ledger] == list(range(1, 913))
    assert Counter(r["role"] for r in ledger) == {"single_channel": 24, "noiseless": 96, "noisy": 768, "stress": 24}
    assert len(metrics["noisy"]) == 768 and len(metrics["noiseless"]) == 96
    assert len(metrics["single_channel"]) == 24 and len(metrics["stress"]) == 24
    count = 0
    for raw in sorted((out/"raw_adc").glob("*.npz")):
        with np.load(raw) as data:
            dn = data["raw_DN"]
            assert dn.shape == (8, 256, 256) and dn.dtype == np.uint16
            assert dn.min() >= 0 and dn.max() <= 4095
            count += len(dn)
    assert count == 768
    op = CodedOperator(protocol["model"], torch.float64)
    replay = []
    for profile in protocol["design"]["brightness_profiles"]:
        with np.load(out/f"example_{profile}.npz") as data:
            raw = torch.tensor(data["raw_e"], dtype=torch.float64)
            got = op.decode(raw).numpy()
            delta = got-data["decoded"]
            assert np.isfinite(got).all()
            assert np.max(np.abs(delta)) < 1e-8
            step = 60000/(4095-128)
            with np.load(out/f"raw_adc/m00_e4000_{profile}.npz") as source:
                assert np.array_equal((source["raw_DN"][0].astype(np.float64)-128)*step, data["raw_e"])
            replay.append({"profile": profile, "max_absolute_CPU_replay_saved_difference_e": float(np.max(np.abs(delta))),
                           "relative_CPU_replay_saved_difference": float(np.linalg.norm(delta)/np.linalg.norm(data["decoded"]))})
    check = {"time": runtime.now(), "remote_files_hash_verified": len(manifest),
             "raw_observations_reopened": count, "ledger_calls_verified": len(ledger),
             "CPU_replay_complete_decodes": 4, "replay": replay, "all_passed": True,
             "source_preserved": True, "saved_result_device": result["device"], "all_saturated_samples": sum(r["saturated_samples"] for r in metrics["noisy"])}
    runtime.write_json(out/"DELIVERY_VERIFICATION.json", check)
    reviews.append({"run": run, "result": result, "verification": check})


first, second, third = [r["result"] for r in reviews]
improvement = 1-third["optical_detail_loss_relative_L2"]["mean"]/first["optical_detail_loss_relative_L2"]["mean"]
comparison = {"time": runtime.now(), "baseline": "filter_decode_001", "intermediate_bandwidth": "filter_decode_003",
              "evidence_level": "development simulation; same independent pre-optics image fixtures, new raw observations",
              "optical_difference_reduction_fraction": improvement,
              "intermediate_geometry_extra_screen_passed": improvement >= .30 and third["engineering_screen_passed"],
              "server_complete_decodes": 2736, "local_correctness_and_replay_decodes": 24,
              "noisy_observations": 2304, "independent_mothers": 8,
              "legacy_inverse_calls_untouched": 6933,
              "next_action": "Measure or specify filter/camera and spatial calibration; for unequal brightness, compare raw-noise-aware image decoding at the same target bandwidth before introducing learned priors.",
              "hardware_validated": False, "training_updates": 0,
              "reviewed_runs": [{"run": r["run"], "engineering_screen_passed": r["result"]["engineering_screen_passed"],
                                 "delivery_verified": r["verification"]["all_passed"]} for r in reviews]}
runtime.write_json(root/"reports/filter_decode_003/COMPARISON.json", comparison)
print(json.dumps(comparison, indent=2))
