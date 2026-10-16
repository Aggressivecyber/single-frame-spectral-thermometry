"""Short CPU checks of the implemented physical and optimization operators."""
import runtime
from pathlib import Path
import json
import time
import torch
from research_joint_imaging.core import Imager, Radiometry, gradient, gradient_adjoint, LearnedReconstruction, training_loss

torch.set_num_threads(2)
torch.manual_seed(1407)
start = time.monotonic()
root = Path(__file__).resolve().parents[1]
p = json.loads((root/"research_joint_imaging/protocol.json").read_text())
op = Imager(p["model"], torch.float64)
x = op.object_blur(torch.rand(2, 6, *p["model"]["shape"], dtype=torch.float64)+0.2)
y = op.encode_images(x)
decode_error = float((op.project(op.decode(y))-x).norm()/x.norm())
r = torch.randn_like(y)
lhs, rhs = (y*r).sum(), (x*op.images_adjoint(r)).sum()
adjoint_error = float((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs()))
g = torch.randn_like(gradient(x))
lhs, rhs = (gradient(x)*g).sum(), (x*gradient_adjoint(g)).sum()
gradient_error = float((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs()))
weight = torch.ones_like(y)
mu = op.prior_mu[:, None, None]
recovered = op.precondition(op.normal(x, weight, mu, 0.02), weight, mu, 0.02)
normal_error = float((recovered-x).norm()/x.norm())
weight = torch.exp(0.2*op.object_blur(torch.randn_like(x))[:, 0])
right = op.normal(x, weight, mu, 0.02)
solved = op.solve(right, torch.zeros_like(x), weight, mu, 0.02, 30, 1e-11)
solve_error = float((solved-x).norm()/x.norm())
proposal = (0.8*x[:1]).detach().requires_grad_(True)
direction = op.project(torch.randn_like(proposal))
probe = op.project(torch.randn_like(proposal))
def proposal_score(value):
    return (op.data_step(y[:1], weight[:1], value, 30)*probe).sum()
score = proposal_score(proposal)
analytic = (torch.autograd.grad(score, proposal)[0]*direction).sum()
epsilon = 1e-5
finite = (proposal_score(proposal.detach()+epsilon*direction)-proposal_score(proposal.detach()-epsilon*direction))/(2*epsilon)
derivative_error = float((analytic-finite).abs()/torch.maximum(analytic.abs(), finite.abs()))
rad = Radiometry(p, "cpu", torch.float64)
temperature_errors = []
for value in (2000, 2800, 3600, 4200):
    c, _ = rad.curve(torch.full((8, 8), value, dtype=torch.float64))
    fitted, _, _, _ = rad.fit(1700*c)
    temperature_errors.append(float((fitted-value).abs().max()))
initial, yn, scale, w, noise = op.initialize(y)
net = LearnedReconstruction(p["learning"], True).double()
output = net(op, initial, yn, w, noise)
loss = training_loss(output, x/scale)
loss.backward()
gradient_finite = all(bool(torch.isfinite(parameter.grad).all()) for parameter in net.parameters())
identity_error = float(((output*scale-x).norm()/x.norm()).detach())
receipt = {"time": runtime.now(), "device": "cpu", "dtype": "float64", "geometry": op.geometry,
           "noiseless_relative_L2": decode_error, "adjoint_relative_error": adjoint_error,
           "spatial_gradient_adjoint_relative_error": gradient_error,
           "normal_preconditioner_relative_error": normal_error, "weighted_PCG_relative_error": solve_error,
           "data_proximal_directional_derivative_relative_error": derivative_error,
           "uniform_temperature_errors_K": temperature_errors, "zero_initialized_hybrid_noiseless_relative_L2": identity_error,
           "network_backward_finite": gradient_finite, "network_parameters": sum(q.numel() for q in net.parameters()),
           "complete_decode_observations": 4, "wall_seconds": time.monotonic()-start}
receipt["passed"] = max(decode_error, adjoint_error, gradient_error, identity_error) < 1e-10 and max(normal_error, solve_error) < 1e-9 and max(temperature_errors) < 0.01 and derivative_error < 1e-4 and gradient_finite
runtime.write_json(root/"reports"/p["run_id"]/"CPU_CONTRACTS.json", receipt)
print(json.dumps(receipt), flush=True)
if not receipt["passed"]:
    raise RuntimeError("Joint-imaging CPU contracts failed")
