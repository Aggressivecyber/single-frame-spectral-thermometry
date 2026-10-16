"""One synthetic native-grid GPU operator/gradient check; no training updates."""
import runtime
import torch
from .run_pilot import load_physics, REPORT
from .linear_profile import LinearProfile
from .model import ConditionalProblem, ResidualNet, infer

torch.set_num_threads(2)
torch.manual_seed(618)
torch.cuda.manual_seed_all(618)
torch.use_deterministic_algorithms(True)
_, p, _ = load_physics('cuda')
shape = p.cfg['shape']
weights = (1000+4000*torch.rand(shape, device='cuda', dtype=torch.float64)).reciprocal()
target = torch.randn(shape, device='cuda', dtype=torch.float64)
linear = LinearProfile(p, weights, 1.)
x0, info = linear.solve(target, tolerance=1e-11)
if not info['numerical_certified']:
    raise ArithmeticError('GPU conditional profile uncertified')
problem = ConditionalProblem(p, weights, target, x0)
network = ResidualNet().cuda()
features = torch.randn((1, 8, *shape), device='cuda')
value, _, proposal = infer(network, problem, features)
loss = value.square().mean()
loss.backward()
gradient = torch.cat([v.grad.flatten() for v in network.parameters()])
native_change = float((value.detach()-linear.U.unpack(x0[:-3])).abs().max())
passed = bool(torch.isfinite(gradient).all() and gradient.norm() > 0 and native_change < .001)
receipt = {'time': runtime.now(), 'device': torch.cuda.get_device_name(0),
           'torch': torch.__version__, 'training_updates': 0, 'experimental_raw_inverse_calls': 0,
           'native_profile': info, 'prox': problem.last_solve,
           'matched_zero_head_max_change_e': native_change, 'gradient_norm': float(gradient.norm()),
           'peak_GPU_MiB': torch.cuda.max_memory_allocated()/2**20, 'all_passed': passed}
runtime.write_json(REPORT/'CUDA_CONTRACTS.json', receipt)
print(receipt, flush=True)
if not passed:
    raise ArithmeticError('GPU native operator/gradient contract failed')
