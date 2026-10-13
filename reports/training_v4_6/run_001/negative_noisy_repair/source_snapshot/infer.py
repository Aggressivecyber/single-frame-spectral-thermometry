"""Inference with the published, unchanged parameter-unfolded checkpoint."""
import runtime
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from network import ParameterUnfoldedNet
from scenes import make_scene

DEFAULT_CHECKPOINT = runtime.ROOT / "checkpoints/selected_checkpoint.pt"
CHECKPOINT_SHA256 = "d777888c53b086118128a824da007e1cba471f185ad9f8a973eb68c5edb434f1"


def load_model(checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
    checkpoint = Path(checkpoint)
    if checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve():
        if runtime.sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("The default checkpoint does not match its published SHA256.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ParameterUnfoldedNet(saved["config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", type=Path, help="An unprocessed DN .npy frame, shape 256x256")
    source.add_argument("--synthetic-demo", action="store_true", help="Generate a synthetic software fixture")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--scene-id", type=int, default=900)
    parser.add_argument("--output", type=Path, default=runtime.ROOT / "outputs/demo")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_model(args.checkpoint, args.device)
    if args.raw:
        raw_array = np.load(args.raw, allow_pickle=False)
        if not np.isfinite(raw_array).all():
            raise ValueError("Raw DN contains nonfinite values.")
        raw = torch.as_tensor(raw_array, dtype=torch.float32, device=args.device)
        if raw.ndim == 2:
            raw = raw[None, None]
        elif raw.ndim == 3:
            raw = raw[:, None]
    else:
        raw = make_scene(model.physics, "audit", args.scene_id)["raw_dn"]
    if raw.ndim != 4 or raw.shape[1] != 1 or tuple(raw.shape[-2:]) != tuple(model.cfg["shape"]):
        raise ValueError("Expected B x 1 x 256 x 256 raw DN input.")
    if not bool(((raw >= 0) & (raw <= 2**model.cfg["adc_bits"]-1)).all()):
        raise ValueError("Raw input is outside the configured ADC range.")
    destination = runtime.check_output(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    if args.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        predicted = model(raw)
    if args.device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    numeric = {key: value.cpu().numpy() for key, value in predicted.items()}
    if any(not np.isfinite(value).all() for value in numeric.values()):
        raise RuntimeError("Nonfinite prediction.")
    np.save(destination / "input_raw_DN.npy", raw.cpu().numpy(), allow_pickle=False)
    np.savez_compressed(destination / "prediction.npz", **numeric)
    with np.load(destination / "prediction.npz", allow_pickle=False) as reopened:
        assert set(reopened.files) == set(numeric)
        assert all(np.array_equal(reopened[key], value) for key, value in numeric.items())
    receipt = {"utc": runtime.now(), "passed": True,
               "checkpoint_sha256": runtime.sha256(args.checkpoint),
               "model": "ParameterUnfoldedNet", "parameters": sum(p.numel() for p in model.parameters()),
               "input_shape": list(raw.shape), "device": args.device,
               "forward_seconds_one_call": seconds,
               "valid_fraction": float(predicted["valid"].double().mean()),
               "saturated_pixels": int(predicted["saturated"].sum()),
               "output_arrays_reopen_exactly": True,
               "prediction_sha256": runtime.sha256(destination / "prediction.npz"),
               "response_provenance": model.cfg["response_provenance"],
               "scope": "synthetic single-template engineering prototype",
               "gas_temperature_concentration_and_hardware_validated": False}
    runtime.write_json(destination / "inference_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
