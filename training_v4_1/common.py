import runtime
import torch
from training_v4.common import read, write, npz, save_npz, parameters_hash
from evaluation_v2.common import Journal, array_hash, canonical_hash

OUT = runtime.ROOT/'reports/training_v4_1/run_001'
DATA = runtime.ROOT/'outputs/training_v4_1/run_001'
OLD_OUT = runtime.ROOT/'reports/training_v4/run_001'
OLD_DATA = runtime.ROOT/'outputs/training_v4/run_001'
BASE = '62538a8bb4a0014c764bd395cfa8cb9d915e9c97'
GPU_PYTHON = 'E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe'
PROTOCOL = {'base': BASE, 'updates': 1500, 'lr': 3e-5, 'accumulation': 4, 'clip': 1.,
    'seed': 4101, 'train_protection_mothers': 48, 'validation_protection_mothers': 24,
    'signed_train_mothers': 12, 'signed_validation_mothers': 6, 'signed_aux_fraction': .05,
    'safe_weights': [1., 5.], 'tikhonov_grid': [.1, 1., 10., 100., 1000.],
    'safe_template_tolerance_e2': .25, 'safe_temperature_tolerance_K2': .04,
    'safe_denominator': 'fixed baseline MSE plus 1 e2 / 1 K2; no truth energy normalization',
    'safe_applies_to': 'all four microbatches; separate output-domain excess penalties',
    'required_weak_L2': 1., 'aspirational_weak_L2': .85, 'aspirational_normal_RMSE_e': 12.,
    'protection_admissibility': 'every geometry/strength/input subgroup mean MSE <= physics mean plus fixed tolerance',
    'noisy_temperature_admissibility': 'each condition mean temperature MSE <=1.05*physics plus .04 K2',
    'small_residual': 'baseline weighted raw RMS <0.2',
    'large_correction': 'template correction RMS>2e or color correction RMS>0.5K',
    'feature_ablation_trigger': 'both safe runs fail and >5% protection frames have small residual and large correction',
    'final_test_evaluations': 0, 'old_954_rerun': False}

def setup():
    from training_v4.common import setup as old_setup
    old_setup()
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def file_path(row, key):
    return runtime.ROOT/row[key]

def rel(path):
    return str(path.relative_to(runtime.ROOT)).replace('\\', '/')

def sources():
    return {rel(p): runtime.sha256(p) for p in sorted((runtime.ROOT/'training_v4_1').glob('*.py'))}
