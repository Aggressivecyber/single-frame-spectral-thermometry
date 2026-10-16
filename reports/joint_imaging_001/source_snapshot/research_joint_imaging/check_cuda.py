"""One noisy forward/backward on the actual server GPU, with no optimizer update."""
import runtime
from pathlib import Path
import json
import time
import torch
from research_joint_imaging.core import Imager, Radiometry, LearnedReconstruction, training_loss
from research_joint_imaging.data import mother

root = Path(__file__).resolve().parents[1]
p = json.loads((root/"research_joint_imaging/protocol.json").read_text())
torch.set_num_threads(2)
torch.manual_seed(7001)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
start = time.monotonic()
op = Imager(p["model"]).cuda()
rad = Radiometry(p, "cuda")
t, amplitude, features = mother(11001, p["model"]["shape"], "cuda")
c, _ = rad.curve(t)
pre = c*amplitude
pre = pre*(4000/pre[4].mean())
pre[:3] += 200*features
target = op.object_blur(pre)[None]
expected = op.encode_images(target)
raw = torch.poisson(expected)+3*torch.randn_like(expected)
initial, y, scale, weight, noise = op.initialize(raw)
rows = []
for method in (False, True):
    model = LearnedReconstruction(p["learning"], method).cuda()
    prediction = model(op, initial, y, weight, noise)
    loss = training_loss(prediction, target/scale)
    loss.backward()
    row = {"method": "cnn_dc" if method else "cnn_post", "loss": float(loss.detach()), "finite_output": bool(torch.isfinite(prediction).all()), "finite_gradients": all(bool(torch.isfinite(v.grad).all()) for v in model.parameters()), "final_weight_gradient_norm": float(model.net[-1].weight.grad.norm())}
    rows.append(row)
torch.cuda.synchronize()
receipt = {"time": runtime.now(), "device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda, "optimizer_updates": 0, "complete_reconstruction_observations": 3, "checks": rows, "wall_s": time.monotonic()-start, "peak_allocated_MiB": torch.cuda.max_memory_allocated()/1024**2, "passed": all(r["finite_output"] and r["finite_gradients"] and r["final_weight_gradient_norm"] > 0 for r in rows)}
runtime.write_json(root/"reports"/p["run_id"]/"CUDA_CONTRACTS.json", receipt)
print(json.dumps(receipt), flush=True)
if not receipt["passed"]:
    raise RuntimeError("Native GPU forward/backward failed")
