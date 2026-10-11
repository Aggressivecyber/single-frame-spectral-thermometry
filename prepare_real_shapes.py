"""Extract experimental image shapes, preserving source provenance and masks."""
import runtime
import argparse
import hashlib
import io
import json
import struct
import time
import zipfile
import zlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
import torch.nn.functional as functional

SOURCES = {
    "flat_flame": {"url": "https://data.mendeley.com/datasets/jvvvgpdy48/6", "doi": "10.17632/jvvvgpdy48.6", "authors": "Tanja Pelzmann; Etienne Robert", "license": "CC BY 4.0"},
    "urea_swirl": {"url": "https://data.mendeley.com/datasets/2tjc68s5vf/3", "doi": "10.17632/2tjc68s5vf.3", "authors": "Maria Grazia De Giorgi; Giuseppe Ciccarella; Donato Fontanarosa; Elisa Pescini; Antonio Ficarella", "license": "CC BY 4.0"},
    "leafinnox": {"url": "https://www.repository.cam.ac.uk/items/1d2e878a-d73a-4cc3-880c-fd639c883ede", "doi": "10.17863/CAM.90803", "authors": "Epaminondas Mastorakos; Pedro M de Oliveira", "license": "CC BY 4.0"},
}


def decode_native_png(payload):
    """Read noninterlaced 8/16-bit PNG pixels without an RGB 16->8 conversion."""
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG")
    position, compressed, header = 8, [], None
    while position < len(payload):
        length = struct.unpack_from(">I", payload, position)[0]
        kind = payload[position+4:position+8]
        data = payload[position+8:position+8+length]
        crc = struct.unpack_from(">I", payload, position+8+length)[0]
        if zlib.crc32(kind+data) & 0xffffffff != crc:
            raise ValueError("PNG CRC mismatch")
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", data)
        elif kind == b"IDAT":
            compressed.append(data)
        elif kind == b"IEND":
            break
        position += length+12
    width, height, bits, color, compression, filtering, interlace = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
    if channels is None or bits not in (8, 16) or compression or filtering or interlace:
        raise ValueError("Unsupported PNG format")
    bpp, stride = channels*(bits//8), width*channels*(bits//8)
    raw = zlib.decompress(b"".join(compressed))
    if len(raw) != height*(stride+1):
        raise ValueError("PNG scanline length mismatch")
    rows = np.empty((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    for y in range(height):
        start = y*(stride+1)
        kind = raw[start]
        scan = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=start+1).astype(np.int32)
        if kind == 1:
            scan = np.cumsum(scan.reshape(-1, bpp), axis=0).reshape(-1) & 255
        elif kind == 2:
            scan = (scan+previous.astype(np.int32)) & 255
        elif kind in (3, 4):
            for x in range(stride):
                a = int(scan[x-bpp]) if x >= bpp else 0
                b = int(previous[x])
                c = int(previous[x-bpp]) if x >= bpp else 0
                if kind == 3:
                    predictor = (a+b)//2
                else:
                    p = a+b-c
                    pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)
                    predictor = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                scan[x] = (scan[x]+predictor) & 255
        elif kind != 0:
            raise ValueError("Invalid PNG scanline filter")
        rows[y] = scan.astype(np.uint8)
        previous = rows[y]
    dtype = ">u2" if bits == 16 else "u1"
    image = np.frombuffer(rows.tobytes(), dtype=dtype).reshape(height, width, channels)
    # Spatial intensity descriptor only; no RGB/QE/radiance interpretation.
    gray = image[..., :3].max(-1) if channels >= 3 else image[..., 0]
    return gray.astype(np.float32), {"native_bits_per_channel": bits, "PNG_color_type": color,
                                   "shape_intensity_rule": "maximum native RGB component" if channels >= 3 else "native grayscale"}


def normalize_shape(image, crop, saturation_value=None):
    native = np.asarray(image)
    if native.ndim != 2 or not np.isfinite(native).all():
        raise ValueError("Expected a finite grayscale image")
    y0, y1, x0, x1 = crop
    selected = native[y0:y1, x0:x1].astype(np.float32)
    valid = np.ones(selected.shape, dtype=bool)
    if saturation_value is not None:
        valid &= selected < saturation_value
    if valid.mean() < .5:
        raise ValueError("Too few valid source pixels")
    height, width = selected.shape
    scale = 256/max(height, width)
    resized_shape = (max(1, round(height*scale)), max(1, round(width*scale)))
    numerator = functional.interpolate(torch.from_numpy(selected*valid)[None, None], size=resized_shape, mode="area")[0, 0].numpy()
    denominator = functional.interpolate(torch.from_numpy(valid.astype(np.float32))[None, None], size=resized_shape, mode="area")[0, 0].numpy()
    resized = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    lo, hi = np.percentile(resized[denominator > 0], [5., 99.5])
    if hi <= lo:
        raise ValueError("No spatial contrast in source")
    normalized = np.clip((resized-lo)/(hi-lo), 0., 1.).astype(np.float32)
    normalized[denominator == 0] = 0.
    canvas = np.zeros((256, 256), dtype=np.float32)
    mask = np.zeros((256, 256), dtype=np.float32)
    top, left = (256-resized_shape[0])//2, (256-resized_shape[1])//2
    canvas[top:top+resized_shape[0], left:left+resized_shape[1]] = normalized
    mask[top:top+resized_shape[0], left:left+resized_shape[1]] = denominator
    note = {"native_shape": list(native.shape), "native_dtype": str(native.dtype), "crop_y0_y1_x0_x1": list(crop),
            "cropped_saturated_or_masked_fraction": float(1.-valid.mean()), "saturation_value": saturation_value,
            "resize": "aspect-preserving area averaging with valid-pixel weights and zero letterbox",
            "normalization": "p5 to p99.5 contrast; clipped to [0,1]; not calibrated radiance",
            "normalization_low": float(lo), "normalization_high": float(hi)}
    return canvas, mask, note


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing the five original source files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-partition", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = runtime.check_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    data = args.data_root.resolve()
    files = ["FFOH_flame10_100gain_100ms.zip", "test1_CH.mat", "test1_OH.mat", "test1_BB.mat", "LEAFINNOX_UCAM_LEAF.zip"]
    source_manifest = {name: {"sha256": runtime.sha256(data/name), "bytes": (data/name).stat().st_size} for name in files}
    runtime.write_json(output / "source_manifest.json", source_manifest)
    records, backgrounds = [], {}
    indices = np.linspace(0, 99, args.frames_per_partition).round().astype(int).tolist()
    for channel in ("BB", "CH", "OH"):
        filename = f"test1_{channel}.mat"
        cube = loadmat(data/filename, variable_names=["YIC"], verify_compressed_data_integrity=True)["YIC"]
        if cube.ndim != 3 or cube.shape[2] != 100:
            raise ValueError("Unexpected MAT cube shape")
        for replicate, index in enumerate(indices):
            frame = cube[:, :, index]
            crop = (16, frame.shape[0]-16, 16, frame.shape[1]-16)
            shape, mask, note = normalize_shape(frame, crop, 1. if channel != "BB" else None)
            name = f"urea_{channel}_{index:03d}.npz"
            np.savez_compressed(output/name, shape=shape, source_valid_fraction=mask)
            metadata = {"partition": f"urea_swirl_{channel}", "dataset": "urea_swirl", "channel": channel,
                        "frame_index": index, "source_file": filename, "source_sha256": source_manifest[filename]["sha256"],
                        "processed_file": name, "processed_sha256": runtime.sha256(output/name), "preprocessing": note,
                        "source": SOURCES["urea_swirl"], "replicate": replicate}
            if channel == "BB":
                backgrounds[("urea", replicate)] = metadata
            else:
                metadata["background_source"] = backgrounds[("urea", replicate)]
                records.append(metadata)
        del cube
    flat_filename = files[0]
    with zipfile.ZipFile(data/flat_filename) as archive:
        members = sorted(name for name in archive.namelist() if name.lower().endswith(".png"))
        if len(members) != 100:
            raise ValueError("Expected 100 flat-flame PNGs")
        for replicate, index in enumerate(indices):
            payload = archive.read(members[index])
            frame, native_info = decode_native_png(payload)
            crop = (128, frame.shape[0]-32, 32, frame.shape[1]-32)
            shape, mask, note = normalize_shape(frame, crop)
            note.update(native_info)
            note["masked_camera_text"] = "top 128 rows removed; fixed before inference"
            name = f"flat_OH_{index:03d}.npz"
            np.savez_compressed(output/name, shape=shape, source_valid_fraction=mask)
            records.append({"partition": "flat_flame_OH", "dataset": "flat_flame", "channel": "OH",
                            "frame_index": index, "source_file": flat_filename, "archive_member": members[index],
                            "source_member_sha256": hashlib.sha256(payload).hexdigest(),
                            "source_sha256": source_manifest[flat_filename]["sha256"], "processed_file": name,
                            "processed_sha256": runtime.sha256(output/name), "preprocessing": note,
                            "source": SOURCES["flat_flame"], "replicate": replicate,
                            "background_source": backgrounds[("urea", replicate)]})
    leaf_filename = files[-1]
    with zipfile.ZipFile(data/leaf_filename) as archive:
        names = archive.namelist()
        for case in ("C1", "C3", "C5"):
            selected = {}
            for channel in ("Soot", "CH"):
                members = sorted(name for name in names if f"/{channel}/{case}/" in name and name.lower().endswith((".tif", ".tiff")))
                if len(members) != 30:
                    raise ValueError(f"Expected 30 LEAF {case} {channel} TIFF images")
                leaf_indices = np.linspace(0, 29, args.frames_per_partition).round().astype(int).tolist()
                for replicate, index in enumerate(leaf_indices):
                    payload = archive.read(members[index])
                    frame = np.asarray(Image.open(io.BytesIO(payload)))
                    if frame.ndim != 2:
                        raise ValueError("LEAF TIFF must be native grayscale")
                    saturation = float(np.iinfo(frame.dtype).max) if frame.dtype.kind in "ui" else None
                    shape, mask, note = normalize_shape(frame, (8, frame.shape[0]-8, 8, frame.shape[1]-8), saturation)
                    name = f"leaf_{case}_{channel}_{index:03d}.npz"
                    np.savez_compressed(output/name, shape=shape, source_valid_fraction=mask)
                    metadata = {"partition": f"leafinnox_{case}_{channel}", "dataset": "leafinnox", "channel": channel,
                                "frame_index": index, "source_file": leaf_filename, "archive_member": members[index],
                                "source_member_sha256": hashlib.sha256(payload).hexdigest(),
                                "source_sha256": source_manifest[leaf_filename]["sha256"], "processed_file": name,
                                "processed_sha256": runtime.sha256(output/name), "preprocessing": note,
                                "source": SOURCES["leafinnox"], "replicate": replicate}
                    if channel == "Soot":
                        selected[replicate] = metadata
                    else:
                        metadata["background_source"] = selected[replicate]
                        records.append(metadata)
    if len(records) != 6*args.frames_per_partition:
        raise ValueError("Unexpected prepared record count")
    for record in records:
        with np.load(output/record["processed_file"], allow_pickle=False) as reopened:
            assert reopened["shape"].shape == (256, 256) and np.isfinite(reopened["shape"]).all()
            assert reopened["shape"].max() > 0
    manifest = {"utc": runtime.now(), "records": records, "measured_shape_frames": len(records),
                "source_acquisition_datasets": 3, "source_groups": 6,
                "frames_per_partition": args.frames_per_partition, "source_files": source_manifest,
                "interchannel_sync": "unknown; shape/background pairing below is a constructed test, not a synchronized observation",
                "calibrated_absolute_intensity": False, "temperature_and_AlO_ground_truth": False,
                "scope": "experimental image morphology and texture; synthetic spectrum, temperature, coding and photons in downstream evaluation",
                "seconds": time.perf_counter()-start, "source_sha256": runtime.sha256(__file__)}
    runtime.write_json(output/"manifest.json", manifest)
    print(json.dumps({"prepared_frames": len(records), "seconds": manifest["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
