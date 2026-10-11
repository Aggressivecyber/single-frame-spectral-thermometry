from .solver import DEFAULTS
from .infer import METHODS, SELECTED

PROTOCOL = {
    'schema': 'physical-repair-v3-first-round', 'baseline_commit': '5f2ae6307a644dbfbbab426b9c239397981d24b7',
    'checkpoint_sha256': 'd777888c53b086118128a824da007e1cba471f185ad9f8a973eb68c5edb434f1',
    'config': DEFAULTS, 'ablation_new_methods': list(METHODS), 'smoke_method': SELECTED,
    'old_baseline': 'B2_six_stage_no_CNN; 12 old data updates, exact audit_v2 input/prediction hashes',
    'old_CNN_reference': 'B4_published reused, no removal or weight changes',
    'gate': {'continuous': {'temperature_RMSE_K_max': 1., 'template_RMSE_e_max': 5., 'historical60_FP_max': 0.},
             'quantization_only': {'temperature_RMSE_K_max': 3., 'template_RMSE_e_max': 6., 'historical60_FP_max': 0.},
             'nonzero_template_relative_L2_max': .25,
             'scope': 'proposed development gates, not hardware standards'},
    'gate_regressions': ['same old front +210e, +45e, -120e templates',
                         'front 70 degrees null, front 25 degrees +210e',
                         'flat 3000K null and +210e; continuous and quantized for each'],
    'gate_required_frames': 16, 'noisy_old_front_diagnostic_frames': 1,
    'smoke_new_inferences': 48, 'smoke_known_calibration_oracles': 8,
    'validation_null_new_inferences': 24,
    'smoke_allowed_only_after': 'all 16 selected-method development gates and production tests passed',
    'smoke_reporting': ['nominal', 'known_calibration_oracle', 'unknown_channel_gain_3pct', 'non_gray_continuum'],
    'measurement_likelihood': 'raw observed electrons only; fixed measurement-derived weights',
    'warm_start_likelihood': 'decoded 700/905 bands only for warm start; never combined with raw as independent data',
    'joint_first_step_decomposition': 'spectral components are a scoring-only preview; joint acceptance is atomic',
    'optimizer_updates': 0, 'CNN_weight_updates': 0, 'network_modules_added': 0,
    'device': 'local CUDA RTX4060', 'TF32': False, 'phase_wall_s': 2400.,
    'host_RSS_MiB_max': 4096., 'GPU_allocated_MiB_max': 2500.,
    'stop_after': 'physical gates, existing 48 smoke, report and GitHub publication; no training',
}
