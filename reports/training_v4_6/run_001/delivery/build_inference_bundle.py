"""Build the predetermined v4.6 update-500 diagnostic inference bundle.

Standard library only.  --check-static reads source/metadata/raw-file bytes but
does not build a bundle.  --build requires the completed registered training
receipt and its exact final checkpoint; it never substitutes another checkpoint,
executes a model, imports the project runtime, or opens a truth/state-cache file.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys
import traceback
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "reports/training_v4_6/run_001"
RUN = REPORT / "negative_noisy_repair"
BUNDLE = ROOT / "outputs/training_v4_6/run_001/delivery/inference_bundle"
ARCHIVE = BUNDLE.with_suffix(".zip")
BUILD_RECEIPT = BUNDLE.parent / "inference_bundle_build_receipt.json"
FAILURE_RECEIPT = BUNDLE.parent / "inference_bundle_build_failure.json"
PROTOCOL = REPORT / "execution_protocol.json"
OLD_MANIFEST = ROOT / "reports/training_v4_1/run_001/data_manifest.json"
BACKBONE = ROOT / "reports/training_v4/run_001/backbone_selection.json"
CHECKPOINT_REL = "training_v4_6/checkpoints/negative_noisy_repair/update_0500.pt"
CHECKPOINT_BUNDLE_REL = "checkpoints/update_0500.pt"
BACKBONE_BUNDLE_REL = "reports/training_v4/run_001/backbone_selection.json"
ENTRYPOINT = "training_v4_4/infer.py"
PROTOCOL_SHA256 = "4d49cd48853b20f62960754d07637a3cca3ae454e79e8e4ac67b8a2f7d78b807"
OLD_MANIFEST_SHA256 = "53bbc44dda8be149fdb9106a4ce344f368c3ee21994095334e612a655a351bf3"
BACKBONE_SHA256 = "d4f6cb17450308717623821995c7d2ab10d487d560c87c3f8979418708b64f71"
BACKBONE_CONFIG_SHA256 = "e8a00514ef2b761d1dbd531adbda957d09d4f2883382574dc6a4f75ff35bf667"
LICENSE_SHA256 = "a89f1bd92085bb1036ce3e29ca1de5053f60642e71cdbac39c7b6f04677609dd"
SOURCE_SHA256 = {
    "config.py": "9fcf349c020aa6dc9e82d1aea4b868ee9ecbd1e03bfdda3bf890614cc2bfc3c7",
    "evaluation_v2/__init__.py": "8e43e52fac04d7d295db8260d384e2e37927b5eba8a2a401bbe4e0c0fe4dd75f",
    "evaluation_v2/common.py": "3e48271bb95732049f97b1368d3ba6b2ab749cdb69a583ffea60dc82298438f1",
    "physics.py": "536654ce52dff3de99490182320859aa25a96e29737cf169c9ec1b7a0f7d109a",
    "repair_v3/__init__.py": "2bec3f3d689e9e6e436099ebac4da7af7f7baafdefd5de13a7354a236716931e",
    "repair_v3/solver.py": "f10f26e5eab130363ddb0e32ffce7511f6bc8f449bf2dbeb484ede49b5fb659e",
    "runtime.py": "8e764bdfee266d3bb6c33a21d9eb95a76aa53e0e520ad74c5c9a7bc20181c25f",
    "solver.py": "2c6eb39c24ea4aa4ef07c3ad6423edcdccb869b24592fad49624b86cd54e8a2c",
    "training_v4/__init__.py": "a2d0eebeff72134db3da0203098916a9cd5f9b972004c24c6040c6e839817a11",
    "training_v4/common.py": "96c6d495ec856f47dd3c8a2d5346d2364fd09316c7f2a88837aed2387d1520ae",
    "training_v4/model.py": "627ef9ffb216ad28389a60fd0040a706a8ab36aad677bcf20807a838b1a33100",
    "training_v4/scalar_engine.py": "e6f9436dbd2bf890e1b27e6e0d4a8e4feb4f3e470cc338510da1772eb772ca8c",
    "training_v4_1/__init__.py": "9d74d3f14bf4babe7edef5cc8b12b48edc709105f7cc301ec0d13bd6a0ea80e3",
    "training_v4_1/common.py": "362cadf9e2d129e20b0bee401b2350b4d8e139a04b06a10fdf403c8acd91ed0e",
    "training_v4_3/__init__.py": "43749ac1df4d23d284172d03513403561a3a3054830c46f791c0fdd10575bab9",
    "training_v4_3/residual_scaling.py": "014c34ccb1fffd4b0857453c4801f0e08a87d72b68ae106faed10decac877ab2",
    "training_v4_4/__init__.py": "5f583108b866b98133db3e2f0e1a27e4774df3f4eabe6ddde73822f3d3c4b120",
    "training_v4_4/infer.py": "62e671a3fcfa74fe164e37b020fb8c0a25f462bdaf590fa34c4854bd2a56ef56",
    "training_v4_4/model.py": "f5d0f2b4b91b5e6717cf9021c707d90ce612e6a6fe2cf2ae5ce0101bfeedfe56",
}
EXAMPLES = (
    {
        "case_id": "v4_1_signed_validation_mother_0000_negative_normal_noise",
        "condition": "negative_normal", "input_kind": "noise", "split": "validation",
        "raw_path": "outputs/training_v4_1/run_001/raw/v4_1_signed_validation_mother_0000/negative_normal_noise.npz",
        "raw_file_sha256": "6a63367dc5c0a3afab20e5a0bb2a4721f010334a33fd411a6c8da331bca6c0b2",
        "raw_hash": "12b9b406a6a8c7e04c678b361543a9465d0444974d4ade5970b2a7c8376569a6",
        "bundle_path": "examples/raw/negative_normal_noise.npz",
    },
    {
        "case_id": "v4_validation_mother_0000_weak_noise0",
        "condition": "weak", "input_kind": "noise", "split": "validation",
        "raw_path": "outputs/training_v4/run_001/raw/v4_validation_mother_0000/weak_noise0.npz",
        "raw_file_sha256": "f2e5f244924b2c0ee2dae7c6d7f8565a0322f606b5c20294aff440977ba94944",
        "raw_hash": "619c2120d59101707446711f1e86d7df734e36918bcd488af7c4fb7f7826b525",
        "bundle_path": "examples/raw/weak_noise0.npz",
    },
    {
        "case_id": "v4_validation_mother_0000_blank_quantization",
        "condition": "blank", "input_kind": "quantization", "split": "validation",
        "raw_path": "outputs/training_v4_1/run_001/raw/v4_validation_mother_0000/blank_quantization.npz",
        "raw_file_sha256": "b0d14d30528de5443b7a858937484390c2819f776751225d8246463cedcf9334",
        "raw_hash": "20bc338b2d0221e154cc82b3e8c7d8b960c103345fcfb8a8ae6097e260065ec3",
        "bundle_path": "examples/raw/blank_quantization.npz",
    },
)
ALLOWED_THIRD_PARTY = {"numpy", "torch"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def ed_path(path):
    result = Path(path).expanduser().resolve()
    require(result.drive.upper() in ("E:", "D:"), "All bundle artifacts must remain on E or D: " + str(result))
    return result


def relative_path(root, relative):
    posix = PurePosixPath(relative)
    require(not posix.is_absolute() and ".." not in posix.parts and ":" not in relative and "\\" not in relative,
            "Unsafe relative path: " + relative)
    result = Path(root).joinpath(*posix.parts).resolve()
    require(result.is_relative_to(Path(root).resolve()), "Path escaped its root: " + relative)
    return result


def module_location(root, name):
    require(name and all(part.isidentifier() for part in name.split(".")), "Invalid module name: " + str(name))
    base = Path(root).joinpath(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py").relative_to(root).as_posix(), False
    if (base / "__init__.py").is_file():
        return (base / "__init__.py").relative_to(root).as_posix(), True
    if base.is_dir():
        return None, True  # Namespace package; its imported leaves still enter the closure.
    return None, False


def dependency_closure(root):
    """Resolve all static local imports, including function bodies and parents."""
    root = Path(root).resolve()
    queued, seen, edges, external = deque([ENTRYPOINT]), set(), set(), set()

    def add(name, origin):
        target, local = module_location(root, name)
        if target:
            edges.add((origin, name, target))
            queued.append(target)
            local = True
        if local:
            parts = name.split(".")
            for count in range(1, len(parts)):
                parent, _ = module_location(root, ".".join(parts[:count]))
                if parent:
                    queued.append(parent)
        else:
            external.add(name.split(".")[0])
        return local

    while queued:
        relative = queued.popleft()
        if relative in seen:
            continue
        seen.add(relative)
        path = relative_path(root, relative)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        package = list(PurePosixPath(relative).parent.parts)
        if package == ["."]:
            package = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                dynamic = isinstance(node.func, ast.Name) and node.func.id == "__import__"
                dynamic |= isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                require(not dynamic, "Dynamic imports need a separate dependency review: " + relative)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name, relative)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    require(node.level <= len(package), "Relative import escaped package: " + relative)
                    prefix = package[:len(package) - node.level + 1]
                    name = ".".join(prefix + ([] if node.module is None else node.module.split(".")))
                else:
                    name = node.module
                require(name, "Cannot resolve an import in " + relative)
                is_local = add(name, relative)
                if is_local:
                    for alias in node.names:
                        if alias.name != "*":
                            leaf = name + "." + alias.name
                            target, local = module_location(root, leaf)
                            if target or local:
                                add(leaf, relative)
    third_party = external - set(sys.stdlib_module_names)
    require(third_party == ALLOWED_THIRD_PARTY, "Unexpected external dependency closure: " + repr(sorted(third_party)))
    return {"local_python_files": sorted(seen), "local_import_edges": [list(x) for x in sorted(edges)],
            "external_top_level_modules": sorted(external), "third_party_modules": sorted(third_party),
            "dynamic_imports_found": False, "project_modules_imported": False}


def literal_assignment(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    require(len(values) == 1, "Expected one literal assignment for " + name)
    return ast.literal_eval(values[0])


def validate_example(path, expected):
    """Verify raw-only uint16 NPZ using ZIP/NPY headers and stdlib byte hashing."""
    require(sha(path) == expected["raw_file_sha256"], "Example raw file changed: " + expected["case_id"])
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist() == ["raw_dn.npy"], "Example NPZ must contain only raw_dn")
        info = archive.getinfo("raw_dn.npy")
        require(not info.flag_bits & 1 and info.file_size <= 1024 * 1024, "Unexpected raw member format/size")
        blob = archive.read(info)  # CRC is also checked by ZipFile.read.
    stream = io.BytesIO(blob)
    require(stream.read(6) == b"\x93NUMPY", "Invalid NPY raw header")
    version = tuple(stream.read(2))
    require(version in ((1, 0), (2, 0), (3, 0)), "Unsupported NPY version")
    format_code = "<H" if version[0] == 1 else "<I"
    size = struct.unpack(format_code, stream.read(struct.calcsize(format_code)))[0]
    require(0 < size <= 65536, "Invalid NPY header size")
    header = ast.literal_eval(stream.read(size).decode("utf-8" if version[0] == 3 else "latin1"))
    require(header == {"descr": "<u2", "fortran_order": False, "shape": (256, 256)},
            "The three frozen examples must retain their native little-endian uint16 256x256 form")
    payload = stream.read()
    require(len(payload) == 256 * 256 * 2, "Unexpected raw payload length")
    h = hashlib.sha256()
    h.update(b"uint16")
    h.update(json.dumps([256, 256]).encode())
    h.update(payload)
    require(h.hexdigest() == expected["raw_hash"], "Example raw array identity changed")
    require(all(value[0] < 4095 for value in struct.iter_unpack("<H", payload)), "Frozen example is saturated or out of ADC range")
    return {"dtype": "uint16", "shape": [256, 256], "npz_members": ["raw_dn.npy"],
            "raw_hash": h.hexdigest(), "truth_or_state_members": 0, "saturated_pixels": 0}


def static_check():
    ed_path(ROOT)
    require(sys.version_info >= (3, 10), "Use Python 3.10+ for the standard-library builder")
    before = {name: sha(relative_path(ROOT, name)) for name in SOURCE_SHA256}
    require(before == SOURCE_SHA256, "A reviewed inference source changed; do not silently refresh the frozen delivery plan")
    graph = dependency_closure(ROOT)
    require(set(graph["local_python_files"]) == set(SOURCE_SHA256), "Static local dependency closure differs from reviewed plan")
    declared = literal_assignment(ROOT / ENTRYPOINT, "SOURCE_PATHS")
    require(set(declared).issubset(SOURCE_SHA256), "An inference source-hash dependency is absent")
    require(literal_assignment(ROOT / ENTRYPOINT, "FROZEN_BACKBONE_CONFIG_SHA256") == BACKBONE_CONFIG_SHA256,
            "Inference entrypoint configuration contract changed")
    require(sha(PROTOCOL) == PROTOCOL_SHA256, "Execution protocol changed")
    protocol = read(PROTOCOL)
    require(protocol["frozen"] is True and protocol["train_updates"] == 500 and protocol["final_test_used"] is False,
            "Execution scope is not the registered v4.6 experiment")
    for name, digest in SOURCE_SHA256.items():
        if name in protocol["frozen_inputs_sha256"]:
            require(protocol["frozen_inputs_sha256"][name] == digest, "Source differs from training protocol: " + name)
    require(sha(OLD_MANIFEST) == OLD_MANIFEST_SHA256 and sha(BACKBONE) == BACKBONE_SHA256,
            "Old manifest or backbone selection changed")
    backbone = read(BACKBONE)
    compact = {key: backbone[key] for key in ("method", "solver_config", "physics_config")}
    require(canonical(compact) == BACKBONE_CONFIG_SHA256, "Compact frozen backbone configuration changed")
    require(compact["physics_config"]["shape"] == [256, 256], "Native inference shape changed")
    require(sha(ROOT / "LICENSE") == LICENSE_SHA256, "Original license changed")
    manifest = read(OLD_MANIFEST)
    by_id = {row["case_id"]: row for row in manifest["rows"]}
    example_checks = []
    for expected in EXAMPLES:
        row = by_id[expected["case_id"]]
        for key, value in expected.items():
            if key != "bundle_path":
                require(row[key] == value, "Frozen example metadata changed: " + expected["case_id"] + "/" + key)
        example_checks.append({**expected, **validate_example(relative_path(ROOT, expected["raw_path"]), expected)})
    require({name: sha(relative_path(ROOT, name)) for name in SOURCE_SHA256} == before, "Source changed during static checking")
    require(not {"torch", "numpy", "runtime"} & set(sys.modules), "Builder imported a numerical/project runtime")
    return {"status": "STATIC_DEPENDENCY_AND_EXAMPLE_CHECK_PASSED", "checked_at": now(),
            "builder_sha256": sha(Path(__file__)), "execution_protocol_sha256": PROTOCOL_SHA256,
            "expected_sources_sha256": SOURCE_SHA256, "dependency_graph": graph,
            "compact_backbone_configuration_sha256": BACKBONE_CONFIG_SHA256, "examples": example_checks,
            "checkpoint_policy": {"path": CHECKPOINT_REL, "optimizer_update": 500,
                "role": "predetermined_diagnostic_endpoint", "fallback_checkpoint_allowed": False,
                "expected_file_sha256_source": "The native COMPLETE500 training receipt, checked at build time",
                "checkpoint_not_opened_by_static_check": True},
            "bundle_built": False, "model_inference_calls": 0, "GPU_calls": 0,
            "labels_opened": 0, "development_state_caches_opened": 0,
            "torch_numpy_runtime_imported": False}


def completed_training():
    """Return checked scalar receipts; never unpickle or import torch."""
    protocol = read(PROTOCOL)
    require(sha(PROTOCOL) == PROTOCOL_SHA256 and read(REPORT / "protocol.json") == protocol,
            "Execution/scoring protocol identity changed")
    for name, expected in protocol["frozen_inputs_sha256"].items():
        path = relative_path(ROOT, name)
        require(path.suffix.lower() != ".npz", "A frozen-code gate must not open data arrays")
        require(sha(path) == expected, "Frozen training input/source changed: " + name)
    status = read(RUN / "status.json")
    require(status["stage"] == "COMPLETE" and status["optimizer_updates"] == 500,
            "The registered v4.6 training must be COMPLETE500; no partial or alternate weight is accepted")
    receipt = read(RUN / "training_receipt.json")
    require(receipt["updates"] == 500 and receipt["microbatches"] == 2000 and receipt["protocol_sha256"] == PROTOCOL_SHA256,
            "Training receipt does not certify this registered completed run")
    for key in ("initial_full_development_summary_equals_v4_5", "empty_Adam_initial_state",
                "physical_buffers_unchanged", "source_hashes_unchanged_at_end",
                "same_frozen_negative_noisy_parent_repair_draws"):
        require(receipt.get(key) is True, "Training verification did not pass: " + key)
    require(receipt.get("final_test_used") is False and receipt.get("locked_development_generated") is False,
            "The completed run exceeded the planned data scope")
    expected_sources = {name: value for name, value in protocol["frozen_inputs_sha256"].items() if name.endswith(".py")}
    require(receipt["source_hashes_at_start"] == expected_sources, "Training source snapshot identities changed")
    for key, name in (("generation_receipt_sha256", "generation_receipt.json"), ("cache_receipt_sha256", "cache_receipt.json")):
        require(sha(REPORT / name) == receipt[key], "Generation/cache receipt changed after training preflight")
        stage = read(REPORT / name)
        require(stage["stage"] == "COMPLETE" and stage["protocol_sha256"] == PROTOCOL_SHA256,
                "Generation/cache does not certify this registered completed run")
    checkpoints = receipt["checkpoints"]
    require(len(checkpoints) == 6 and {item["update"] for item in checkpoints} == {0, 100, 200, 300, 400, 500},
            "Native checkpoint inventory is incomplete or duplicated")
    for item in checkpoints:
        update = item["update"]
        expected_name = "initial.pt" if update == 0 else f"update_{update:04d}.pt"
        expected_path = "training_v4_6/checkpoints/negative_noisy_repair/" + expected_name
        require(item["path"] == expected_path and sha(relative_path(ROOT, item["path"])) == item["sha256"],
                "Native checkpoint file differs from its training receipt")
        require(item["Adam_steps"] == ([] if update == 0 else [update]), "Native checkpoint Adam inventory changed")
    final = next(item for item in checkpoints if item["update"] == 500)
    require(final["path"] == CHECKPOINT_REL and final["parameters_sha256"] == receipt["final_parameters_sha256"],
            "The predetermined diagnostic endpoint does not match the completed final weights")
    replays = receipt["native_replays"]
    require(len(replays) == 3 and {item["case_id"] for item in replays} == {item["case_id"] for item in EXAMPLES},
            "The three fixed old-DEV checkpoint replay cases changed")
    require(all(item["cache_identity_unchanged"] and all(math.isfinite(value) and value < 1e-5 for value in item["differences"].values())
                for item in replays), "Saved native checkpoint replay did not pass")
    new_replays = receipt["new_cache_native_replays"]
    require(len(new_replays) == 2 and all(item["raw_state_beta_exact"] and item["outputs_exact"] for item in new_replays),
            "Saved new-TRAIN native checkpoint replays did not pass")
    summary = read(RUN / "validation_summary_0500.json")
    science = read(RUN / "science_summary_0500.json")
    require(summary["optimizer_update"] == science["optimizer_update"] == 500
            and science["protocol_sha256"] == sha(REPORT / "protocol.json"), "Endpoint development summary identity changed")
    require(science["validation_sha256"] == sha(RUN / "validation_update_0500.jsonl"), "Endpoint saved validation records changed")
    require(science["old_eligible"] == summary["eligible"]
            and science["science_candidate"] == (summary["eligible"] and science["new_development_screen_pass"]),
            "Endpoint gate bookkeeping is inconsistent")
    return protocol, status, receipt, final, summary, science


def write_bytes(root, relative, payload):
    target = relative_path(root, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
    require(target.read_bytes() == payload, "Output did not reopen exactly: " + relative)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def write_json(root, relative, value):
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    result = write_bytes(root, relative, payload)
    require(read(relative_path(root, relative)) == value, "JSON did not reopen exactly: " + relative)
    return result


VERIFY_BUNDLE_SOURCE = r'''"""Verify immutable inference payload SHA256s; no model/runtime imports."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
inventory = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
for relative, item in inventory["files"].items():
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or ":" in relative or "\\" in relative:
        raise ValueError("Unsafe inventory path")
    path = root.joinpath(*name.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Inventory path escaped bundle")
    payload = path.read_bytes()
    if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise ValueError("Bundle file changed: " + relative)
print(json.dumps({"status": "BUNDLE_PAYLOAD_HASHES_PASS", "checked_files": len(inventory["files"]),
                  "model_inference_calls": 0, "inventory_self_hash_is_in_external_build_receipt": True}))
'''


def readme_text(final, science):
    return f'''# v4.6 raw 推理诊断包

本包固定携带 v4.6 第 500 次更新的诊断权重，不按开发分数选择权重，也不自动将其认定为科学候选或可部署模型。构建时该端点的联合开发门禁记录为 `{science['science_candidate']}`；它来自已反复使用的开发集，不等于独立确认。

包内包含 raw 推理入口及其全部本地 Python 导入依赖、固定物理/求解器配置、原样 checkpoint、三个 `raw_dn` 示例、许可和 SHA256 清单。NumPy/PyTorch 运行环境需要单独提供。包内不包含 truth 标签、开发反演 state/beta 缓存、训练全数据或完整训练工作流。

## 使用

解压到 E 或 D 盘后，在 `inference_bundle` 目录运行。推理只接收一个 256×256 `raw_dn` 数组的 NPZ；CLI 不接收条件、真值或缓存 state。以下 PowerShell 示例将临时文件和输出也放在包所在 E/D 盘的相邻目录。

```powershell
python -B verify_bundle.py
$bundleParent = Split-Path -Parent (Get-Location).Path
$env:THERMOMETRY_RUNTIME = Join-Path $bundleParent 'inference_runtime'
python -B -m training_v4_4.infer --raw examples/raw/negative_normal_noise.npz --checkpoint checkpoints/update_0500.pt --backbone reports/training_v4/run_001/backbone_selection.json --device cuda --output (Join-Path $bundleParent 'raw_negative_normal_result')
```

`python` 必须指向安装了 NumPy 和合适 PyTorch wheel 的环境。`requirements-inference.txt` 给出项目的依赖范围；本包没有安装这些依赖。若使用 CPU，明确改为 `--device cpu`；CPU 与 CUDA 的数值停机路径和耗时可能不同。构建脚本没有执行上述推理，搬移后的原生运行与逐像素一致性应由后续独立回执确认。

其余固定示例为 `examples/raw/weak_noise0.npz` 和 `examples/raw/blank_quantization.npz`。每次运行使用一个新的 E/D 输出目录；已有目录会被拒绝。

输出为 `prediction.npz`、由该 raw 新求解的 `physical_inverse.npz`、输入副本和 `inference_receipt.json`。回执保存实际来源、配置、源码 hash、逐数组 hash 和分开的物理反演/CNN 时间。一次调用的耗时不是持续吞吐或 10 kHz 硬件验收。

## 权重和科学范围

- 预定端点：`{CHECKPOINT_REL}`；包内 SHA256：`{final['sha256']}`。
- 参数 hash：`{final['parameters_sha256']}`；CNN 参数数 5797。
- 结构为固定标量物理反演后的六阶段共享 CNN；LM/PCG 不在可微训练图内。
- 最终连续谱和有效色温固定为本次 raw 物理反演的结果。内部连续谱提议仍会影响 signed 校正；温度恒等属于输出约束。
- 光学、传感器和模板响应来自固定合成配置；signed 模板是数值系数，不是实测化学标定。
- 三个示例来自既有 DEV，适用于运行和搬移核对，不构成新的独立验证。没有打开 locked DEV 或 final test。
- 原样 checkpoint 可能保留优化器状态和训练协议元数据，推理入口只使用所需的 CNN、物理配置与系数校准；它不恢复训练或读取训练样本。

`MODEL_CARD.md` 记录构建时保存的开发门禁和使用限制。`provenance/bundle_manifest.json` 记录期望 source/weight/raw hash 及原始完成回执的 hash。`SHA256SUMS.json` 覆盖所有静态有效载荷文件（自身除外）；外部 `inference_bundle_build_receipt.json` 另记录它自身及整个 ZIP 的 hash。ZIP 在构建结束后逐项重开核对，不以压缩命令成功代替验收。
'''


def model_card_text(receipt, final, summary, science):
    groups = "\n".join(
        f"|{item['condition']}|{item['template_RMSE_e']:.6f}|{math.sqrt(item['MSE_limit_e2']):.6f}|{item['template_L2']:.6f}|{item['passed']}|"
        for item in science["signed_noisy_conditions"])
    return f'''# Model card: FixedContinuumCNN v4.6 / diagnostic update 500

模型用途是复现固定合成模型下从 raw DN 到物理反演后 CNN 输出的诊断路径。本包预先固定最后一个训练检查点，不执行候选排序；任何独立确认或实际使用验收都需要另行提供证据。

训练完成回执记录 500 次更新、2000 微批次。相对 v4.5，只有 513 个负向含噪位置重采样（502 个 case ID 实际改变、11 个返回原 case），其余 1487 个位置原样保留。两个负向含噪条件各覆盖 36 母体、每母体两次噪声；本臂实际使用 120 个新增案例。生成 432 个新 TRAIN 案例不代表全部参与训练；新的正向、continuous、quantization 案例未在本臂使用。母体覆盖和噪声重复同时变化，不能分别归因。

输入是单个 256×256 raw DN。输出包括连续谱有效色温、signed 模板、其他物理量和诊断；饱和输入被拒绝。在线推理从 raw 重算标量反演，不读取开发 state 缓存或真值。固定输出约束使最终连续谱/色温等于当前物理反演结果；模型未证明色温学习改善，也不是全可微反演展开网络。

构建时，第 500 更新端点主 weak pooled L2 为 {summary['weak_L2']:.6f}，主 normal RMSE 为 {summary['normal_RMSE_e']:.6f} e。旧 eligible={summary['eligible']}，新增开发筛查={science['new_development_screen_pass']}，联合开发候选标记={science['science_candidate']}。下表来自同一个端点的保存分数，未重新推理，也未把不同检查点的指标合并。

|signed noisy 条件|RMSE e|冻结 RMSE 上限 e|pooled L2|该组通过|
|---|---:|---:|---:|---|
{groups}

这些分数使用反复参与开发的 776 帧；不构成独立泛化、统计显著性、真实测温、仪器标定或硬件吞吐证据。新 locked DEV 和 final test 未用于本阶段。包内三个 raw 示例同样来自旧 DEV，只用于诊断复现与搬移测试。

已完成的训练回执 hash：`{sha(RUN / 'training_receipt.json')}`。预定权重 hash：`{final['sha256']}`。参数 hash：`{receipt['final_parameters_sha256']}`。

打包完成仅证明字节、依赖清单和 ZIP 完整性；构建器的模型/GPU 调用为 0。搬移后的原生 CLI 能否执行、输出是否逐像素一致以及实际耗时，须读取后续执行回执，不能由本 model card 推断。本包不承诺完整训练数据自包含。
'''


def build():
    audit = static_check()
    protocol, status, training, final, summary, science = completed_training()
    target = ed_path(BUNDLE)
    require(target == BUNDLE.resolve() and target.is_relative_to(ROOT.resolve()), "Unexpected bundle destination")
    ed_path(ARCHIVE); ed_path(BUILD_RECEIPT); ed_path(FAILURE_RECEIPT)
    partial = ARCHIVE.with_suffix(ARCHIVE.suffix + ".partial")
    for path in (target, ARCHIVE, partial, BUILD_RECEIPT, FAILURE_RECEIPT):
        require(not path.exists(), "Refusing to replace an existing delivery artifact: " + str(path))
    # All checks above are read-only.  No directory is created for a partial run.
    inputs = {name: sha(relative_path(ROOT, name)) for name in SOURCE_SHA256}
    inputs.update({item["raw_path"]: item["raw_file_sha256"] for item in EXAMPLES})
    watched = [PROTOCOL, REPORT / "protocol.json", OLD_MANIFEST, BACKBONE, ROOT / "LICENSE",
               RUN / "status.json", RUN / "training_receipt.json", RUN / "validation_summary_0500.json",
               RUN / "science_summary_0500.json", RUN / "validation_update_0500.jsonl",
               REPORT / "generation_receipt.json", REPORT / "cache_receipt.json", Path(__file__)]
    for path in watched:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    inputs[CHECKPOINT_REL] = final["sha256"]
    target.mkdir(parents=True, exist_ok=False)
    files = {}
    try:
        for relative, expected in SOURCE_SHA256.items():
            payload = relative_path(ROOT, relative).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == expected, "Source changed during copying: " + relative)
            files[relative] = write_bytes(target, relative, payload)
        files["LICENSE"] = write_bytes(target, "LICENSE", (ROOT / "LICENSE").read_bytes())
        payload = relative_path(ROOT, CHECKPOINT_REL).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == final["sha256"], "Final diagnostic checkpoint changed during copying")
        files[CHECKPOINT_BUNDLE_REL] = write_bytes(target, CHECKPOINT_BUNDLE_REL, payload)
        backbone = read(BACKBONE)
        compact = {key: backbone[key] for key in ("method", "solver_config", "physics_config")}
        files[BACKBONE_BUNDLE_REL] = write_json(target, BACKBONE_BUNDLE_REL, compact)
        require(canonical(read(target / BACKBONE_BUNDLE_REL)) == BACKBONE_CONFIG_SHA256, "Compact copied configuration changed")
        examples = []
        for expected in EXAMPLES:
            files[expected["bundle_path"]] = write_bytes(target, expected["bundle_path"], relative_path(ROOT, expected["raw_path"]).read_bytes())
            verified = validate_example(relative_path(target, expected["bundle_path"]), expected)
            examples.append({**expected, **verified, "purpose": "old-DEV diagnostic raw replay, not independent confirmation"})
        files["examples/manifest.json"] = write_json(target, "examples/manifest.json", {
            "examples": examples, "truth_label_files_included": 0, "development_state_caches_included": 0,
            "condition_metadata_is_not_an_inference_input": True})
        files["requirements-inference.txt"] = write_bytes(target, "requirements-inference.txt", b"torch>=2.10,<2.13\nnumpy>=2.2,<3\n")
        files["verify_bundle.py"] = write_bytes(target, "verify_bundle.py", VERIFY_BUNDLE_SOURCE.encode("utf-8"))
        files["README.md"] = write_bytes(target, "README.md", readme_text(final, science).encode("utf-8"))
        files["MODEL_CARD.md"] = write_bytes(target, "MODEL_CARD.md", model_card_text(training, final, summary, science).encode("utf-8"))
        bundled_graph = dependency_closure(target)
        require(bundled_graph == audit["dependency_graph"], "Bundled Python dependency closure differs from reviewed source tree")
        manifest = {
            "schema": "v4.6-diagnostic-raw-inference-bundle-v1", "built_at": now(),
            "artifact_role": "predetermined_diagnostic_update_0500", "candidate_selected_by_builder": False,
            "expected_sources_sha256": SOURCE_SHA256, "dependency_graph": bundled_graph,
            "checkpoint": {**final, "bundle_path": CHECKPOINT_BUNDLE_REL,
                "expected_sha256_from": "native COMPLETE500 training_receipt.json", "checkpoint_payload_loaded_by_builder": False},
            "compact_backbone_configuration_sha256": BACKBONE_CONFIG_SHA256,
            "execution_protocol_sha256": PROTOCOL_SHA256,
            "training_receipt_sha256": sha(RUN / "training_receipt.json"),
            "training_status": {key: status[key] for key in ("stage", "optimizer_updates", "limit")},
            "builder_sha256": sha(Path(__file__)), "builder_python": sys.version,
            "examples": examples, "input_files_sha256": inputs,
            "source_snapshot_matches_frozen_training_dependencies": True,
            "saved_endpoint_old_eligible": summary["eligible"],
            "saved_endpoint_science_candidate_flag": science["science_candidate"],
            "native_relocation_test": "NOT_PERFORMED_BY_BUILDER",
            "native_relocation_pixel_comparison": "NOT_PERFORMED_BY_BUILDER",
            "model_inference_calls": 0, "GPU_calls": 0, "labels_opened": 0,
            "development_state_caches_opened": 0, "labels_included": 0,
            "development_state_caches_included": 0, "full_training_data_self_contained": False,
            "locked_development_used": False, "final_test_used": False,
            "checkpoint_note": "Exact native checkpoint bytes retain optimizer/protocol metadata; no truth or cached raw inverse arrays are introduced by the builder.",
        }
        files["provenance/bundle_manifest.json"] = write_json(target, "provenance/bundle_manifest.json", manifest)
        inventory = {"schema": "sha256-file-inventory-v1", "files": dict(sorted(files.items())),
                     "self_excluded": "SHA256SUMS.json; its SHA256 is recorded in the external build receipt"}
        inventory_info = write_json(target, "SHA256SUMS.json", inventory)
        all_files = {**files, "SHA256SUMS.json": inventory_info}
        actual = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
        require(actual == set(all_files), "Unexpected file entered the inference bundle")
        for relative, item in all_files.items():
            path = relative_path(target, relative)
            require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Payload verification failed: " + relative)
        # Fixed metadata makes archive structure deterministic for a fixed payload.
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative in sorted(all_files):
                info = zipfile.ZipInfo(target.name + "/" + relative, date_time=(2026, 10, 13, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, relative_path(target, relative).read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        with zipfile.ZipFile(partial) as archive:
            expected_names = {target.name + "/" + name for name in all_files}
            require(len(archive.infolist()) == len(expected_names) and set(archive.namelist()) == expected_names,
                    "ZIP entry inventory is missing, duplicated or unexpected")
            require(archive.testzip() is None, "ZIP CRC verification failed")
            for relative, item in all_files.items():
                payload = archive.read(target.name + "/" + relative)
                require(len(payload) == item["bytes"] and hashlib.sha256(payload).hexdigest() == item["sha256"],
                        "Reopened ZIP entry changed: " + relative)
        for relative, expected in inputs.items():
            require(sha(relative_path(ROOT, relative)) == expected, "Build input changed during packaging: " + relative)
        require(not ARCHIVE.exists(), "An archive appeared during building")
        partial.rename(ARCHIVE)
        receipt = {"status": "COMPLETE_BUNDLE_BYTES_AND_ZIP_VERIFIED", "completed_at": now(),
            "bundle_path": str(target), "zip_path": str(ARCHIVE), "zip_sha256": sha(ARCHIVE),
            "zip_bytes": ARCHIVE.stat().st_size, "bundle_file_count": len(all_files),
            "bundle_files_sha256_and_bytes": dict(sorted(all_files.items())),
            "input_files_sha256": inputs, "checkpoint_update": 500, "checkpoint_sha256": final["sha256"],
            "checkpoint_role": "predetermined_diagnostic_endpoint", "checkpoint_auto_selected": False,
            "all_payload_files_reopened": True, "zip_each_entry_reopened_and_verified": True,
            "zip_CRC_check_passed": True, "dependency_closure_count": len(SOURCE_SHA256),
            "third_party_dependencies": sorted(ALLOWED_THIRD_PARTY), "model_inference_calls": 0,
            "GPU_calls": 0, "truth_label_files_included": 0, "development_state_cache_files_included": 0,
            "native_relocation_test_performed": False, "pixel_identity_verified_by_builder": False,
            "full_training_data_self_contained": False, "final_test_used": False}
        write_json(BUILD_RECEIPT.parent, BUILD_RECEIPT.name, receipt)
        require(not {"torch", "numpy", "runtime"} & set(sys.modules), "Builder imported a forbidden runtime")
        return receipt
    except BaseException as exc:
        failure = {"status": "FAILED_RETAINED_PARTIAL_ARTIFACTS", "time": now(), "type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc(), "bundle_path": str(target),
                   "files_written_before_failure": sorted(files), "partial_artifacts_deleted": False,
                   "alternate_checkpoint_attempted": False, "GPU_calls": 0}
        if not FAILURE_RECEIPT.exists():
            write_json(FAILURE_RECEIPT.parent, FAILURE_RECEIPT.name, failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-static", action="store_true", help="Read-only dependency/source/example checks; no bundle output")
    action.add_argument("--build", action="store_true", help="Build only after native COMPLETE500; reject every existing output")
    args = parser.parse_args()
    result = static_check() if args.check_static else build()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
