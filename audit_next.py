"""Unified additive audit entry point; no training or weight updates."""
import runtime
import argparse
import os
from pathlib import Path
for cache in ("cuda", "inductor", "triton"):
    (runtime.RUNTIME/cache).mkdir(parents=True, exist_ok=True)
os.environ["CUDA_CACHE_PATH"] = str(runtime.RUNTIME/"cuda")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(runtime.RUNTIME/"inductor")
os.environ["TRITON_CACHE_DIR"] = str(runtime.RUNTIME/"triton")
from evaluation_v2.audit import inventory, freeze_smoke, register_methods, compare, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("inventory", "freeze-smoke", "register-methods", "compare", "summarize"))
    parser.add_argument("--suite", default="smoke", choices=("smoke", "legacy", "diagnostics", "all"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=runtime.ROOT/"outputs/audit_v2/run_001")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output, args.data_root = args.output.resolve(), args.data_root.resolve()
    if os.name == "nt" and any(p.drive.upper() not in ("E:", "D:") for p in (args.output, args.data_root)):
        raise ValueError("Local audit data/report outputs must be on E or D")
    {"inventory": inventory, "freeze-smoke": freeze_smoke, "register-methods": register_methods, "compare": compare, "summarize": summarize}[args.phase](args)


if __name__ == "__main__":
    main()
