"""Explicit synthetic fixtures; replace responses with calibrated artifacts."""
import runtime
import copy
import json


def default_config(size=256):
    bins = [(51, 0), (0, 51), (51, 51), (-51, 51), (102, 0), (0, 102), (102, 51)]
    return {
        "schema": "parameter-unfold-v004",
        "shape": [size, size],
        "exposure_s": 2e-5,
        "centers_nm": [430, 450, 486, 513, 535, 700, 905],
        "widths_nm": [10, 10, 6, 10, 10, 10, 10],
        "quantum_efficiencies": [0.45, 0.45, 0.45, 0.45, 0.45, 0.5, 0.5],
        "temperature_range_K": [1200, 4500],
        "temperature_lut_step_K": 5,
        "reference_temperature_K": 3300,
        "reference_905_e": 24000,
        "carrier_bins_xy": [[round(x * size / 256), round(y * size / 256)] for x, y in bins],
        "modulation_depth": [0.85] * 7,
        "channel_gain": [1.0] * 7,
        "phase_rad": [0.3, -0.4, 0.2, -0.1, 0.1, -0.2, 0.35],
        "object_cutoff_cpp": 0.075,
        "decode_cutoff_cpp": 0.08,
        "relay_psf_sigma_px": 0.35,
        "read_noise_e": 3,
        "full_well_e": 60000,
        "adc_bits": 12,
        "black_level_DN": 128,
        "species_names": ["CH", "C2", "AlO"],
        "active_species": [False, False, True],
        "species_scale_e": 60,
        "background_scale_e": 100,
        "temperature_center_K": 3000,
        "temperature_scale_K": 1000,
        "template_response": [[1.0, 0.03, 0.08], [0.02, 0.05, 0.35], [0.01, 0.04, 1.0], [0.01, 1.0, 0.75], [0.01, 0.05, 0.22], [0, 0, 0], [0, 0, 0]],
        "background_response": [0.9, 1.0, 0.9, 0.85, 0.82, 0, 0],
        "response_provenance": {
            "kind": "synthetic_engineering_fixture",
            "matrix_is_measured": False,
            "matrix_is_derived_from_real_line_list": False,
            "QE_is_measured": False,
            "filter_response_is_measured": False,
            "warning": "All matrix numbers and flat-QE band responses are software fixtures, not physical CH/C2/AlO calibration.",
        },
        "steps": 6,
        "CNN_width": 16,
        "CG_steps_per_spectral_block": 4,
        "GN_damping": 1e-3,
        "parameter_correction_bounds": [0.025, 0.05, 0.5, 0.5, 0.5],
    }


def load_config(path=None, size=None):
    cfg = json.loads(open(path, encoding="utf-8").read()) if path else default_config(size or 256)
    validate_config(cfg)
    return copy.deepcopy(cfg)


def validate_config(cfg):
    if len(cfg["centers_nm"]) != 7 or len(cfg["template_response"]) != 7:
        raise ValueError("This implementation expects seven core bands.")
    if any(len(row) != 3 for row in cfg["template_response"]):
        raise ValueError("Template matrix must be 7 x 3.")
    if len(cfg["active_species"]) != 3:
        raise ValueError("Three explicit species flags are required.")
    if len(cfg["carrier_bins_xy"]) != 7 or min(cfg["shape"]) < 8:
        raise ValueError("Invalid coded image configuration.")
    if cfg["object_cutoff_cpp"] > cfg["decode_cutoff_cpp"]:
        raise ValueError("Decoder may not truncate the declared object bandwidth.")
    if cfg["steps"] != 6 or cfg["CNN_width"] != 16:
        raise ValueError("The first registered implementation fixes six stages and width sixteen.")
    sampled = cfg.get("sampled_instrument_response")
    if sampled:
        grid = sampled["wavelength_nm"]
        if len(grid) < 2 or any(b <= a for a, b in zip(grid, grid[1:])):
            raise ValueError("Response wavelength grid must be strictly increasing.")
        if len(sampled["response"]) != 7 or any(len(row) != len(grid) or min(row) < 0 for row in sampled["response"]):
            raise ValueError("Measured response array must be nonnegative 7 x wavelength samples.")

