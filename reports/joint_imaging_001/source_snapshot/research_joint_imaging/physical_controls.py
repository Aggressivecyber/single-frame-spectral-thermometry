"""Small radiometric controls for interpreting an ultrafast exposure."""
import runtime
from pathlib import Path
import json
import torch
from research_joint_imaging.core import Radiometry

torch.set_num_threads(2)
root = Path(__file__).resolve().parents[1]
p = json.loads((root/"research_joint_imaging/protocol.json").read_text())
rad = Radiometry(p, "cpu", torch.float64)
field = torch.ones((4, 4), dtype=torch.float64)
cold, _ = rad.curve(field*2400)
hot, _ = rad.curve(field*3600)
uniform, _ = rad.curve(field*3000)
mixed = 0.5*cold+0.5*hot
wl = torch.tensor(p["channel_centers_nm"], dtype=torch.float64)[:, None, None]
non_gray = uniform*(wl/800)**0.3
overlap = uniform*4000
overlap[1] += 200
overlap[2] += 50
rows = []
for label, image, meaning in [("equal_duration_2400_3600K", mixed*4000, "equal gray amplitudes and equal exposure durations; arithmetic temperature mean is 3000 K"), ("non_gray_3000K", non_gray*4000, "physical temperature 3000 K; assumed emissivity proportional to wavelength power 0.3"), ("AlO_region_overlap", overlap, "200 e injected at 486 nm plus an assumed 50 e at 516.5 nm; cross-response is illustrative and uncalibrated")]:
    fitted, _, excess, residual = rad.fit(image)
    rows.append({"case": label, "meaning": meaning, "fitted_color_temperature_K": float(fitted.mean()), "thermal_relative_fit_residual": float(residual.mean()), "apparent_feature_excess_e": excess.mean((-2, -1)).tolist()})
runtime.write_json(root/"reports"/p["run_id"]/"PHYSICAL_INTERPRETATION_CONTROLS.json", {"time": runtime.now(), "evidence_level": "analytic synthetic controls, no reconstruction or camera measurements", "cases": rows})
print(json.dumps(rows), flush=True)
