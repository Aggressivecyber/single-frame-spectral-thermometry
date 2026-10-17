"""Small CPU readout diagnostic; no reconstruction, datasets, or training.

Run from the repository root with Python -B. Uses the unchanged Radiometry
implementation, float64, uniform graybody bands and deterministic gain errors.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import runtime
import json
import torch
from research_joint_imaging.core import Radiometry

torch.set_num_threads(1)
OUT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "reports/joint_imaging_v2_recovery_002/source_snapshot/research_joint_imaging_v2/protocol.json"
protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
rad = Radiometry(protocol, "cpu", torch.float64)
cases = {
    "700 +1%": [0, 0, 0, .01, 0, 0],
    "800 +1%": [0, 0, 0, 0, .01, 0],
    "905 +1%": [0, 0, 0, 0, 0, .01],
    "700 +1%, 905 -1%": [0, 0, 0, .01, 0, -.01],
    "NIR all +1%": [0, 0, 0, .01, .01, .01],
    "All six +1%": [.01] * 6,
}
points = []
fd_errors = []
identity_errors = []
with torch.no_grad():
    for temperature in (2400., 3000., 3600.):
        curve, derivative = rad.curve(torch.tensor([[temperature]], dtype=torch.float64))
        c, d = curve[:, 0, 0], derivative[:, 0, 0]
        for level in (1000., 4000.):
            amplitude = level / c[4]
            bands = (amplitude * c)[:, None, None]
            base_t, base_c, base_e, _ = rad.fit(bands)
            assert abs(float(base_t) - temperature) < 1e-8
            assert float(base_e.abs().max()) < 1e-8
            cn, dn = c[3:], d[3:]
            h = torch.dot(cn, dn) / torch.dot(cn, cn)
            j = amplitude * (dn - cn * h)
            jt = j / torch.dot(j, j)
            jc = torch.outer(c[:3], cn) / torch.dot(cn, cn)
            jc += torch.outer(amplitude * (d[:3] - c[:3] * h), jt)
            jacobian = torch.cat((jt[None], jc), dim=0) * bands[3:, 0, 0]
            for step in (1e-4, 1e-5):
                columns = []
                for index in range(3):
                    direction = torch.zeros_like(bands)
                    direction[index + 3] = step
                    plus = rad.fit(bands * torch.exp(direction))
                    minus = rad.fit(bands * torch.exp(-direction))
                    values = torch.cat(((plus[0] - minus[0]).reshape(1),
                                        (plus[1][:3] - minus[1][:3]).reshape(3)))
                    columns.append(values / (2 * step))
                difference = torch.stack(columns, dim=1) - jacobian
                # Row-normalized: temperature and electron rows have different units.
                error = (difference.norm(dim=1) / jacobian.norm(dim=1)).max().item()
                fd_errors.append(error)
                assert error < 1e-6, (temperature, level, step, error)
            responses = []
            for name, errors in cases.items():
                gain = torch.tensor(errors, dtype=torch.float64)[:, None, None]
                changed = bands * (1 + gain)
                t, continuum, excess, _ = rad.fit(changed)
                delta_v = changed[:3] - bands[:3]
                delta_c = continuum[:3] - base_c[:3]
                delta_e = excess - base_e
                identity_error = (delta_e - (delta_v - delta_c)).abs().max().item()
                identity_errors.append(identity_error)
                assert identity_error < 1e-10
                responses.append({
                    "case": name, "fractional_gain_error_six_channels": errors,
                    "temperature_bias_K": float(t - base_t),
                    "visible_excess_bias_e": delta_e.flatten().tolist(),
                    "visible_image_change_e": delta_v.flatten().tolist(),
                    "extrapolated_continuum_change_e": delta_c.flatten().tolist(),
                })
            points.append({
                "temperature_K": temperature, "mean_800nm_e": level,
                "baseline_band_e": bands.flatten().tolist(),
                "jacobian_columns": ["log_I700", "log_I800", "log_I905"],
                "jacobian_rows": ["T_K", "C431.5_e", "C486_e", "C516.5_e"],
                "jacobian": jacobian.tolist(), "gain_responses": responses,
            })

results = {
    "scope": "analytic readout sensitivity only; uniform noiseless graybody, no network or device data",
    "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): runtime.sha256(p)
                      for p in (PROTOCOL, ROOT / "research_joint_imaging/core.py", Path(__file__))},
    "precision": "CPU float64; unchanged Radiometry.fit (10 iterations)",
    "gain_error_location": "recovered band intensities entering Radiometry, not a re-encoded raw sensor perturbation",
    "finite_difference_log_steps": [1e-4, 1e-5],
    "maximum_row_relative_jacobian_error": max(fd_errors),
    "maximum_excess_decomposition_error_e": max(identity_errors),
    "points": points,
}
runtime.write_json(OUT / "CALIBRATION_SENSITIVITY.json", results)

main = next(p for p in points if p["temperature_K"] == 3000 and p["mean_800nm_e"] == 4000)
print(json.dumps({"jacobian_error": max(fd_errors), "decomposition_error_e": max(identity_errors),
                  "main_gain_responses": main["gain_responses"]}, indent=2))
