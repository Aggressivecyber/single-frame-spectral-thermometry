"""Bounded, native-grid first training experiment."""
from repair_v3.solver import DEFAULTS

BASE = '9a7ac7e86339ba9b9d3006765fef731f10941029'
PROTOCOL = {
    'schema': 'physical-inverse-post-CNN-v4-first-training', 'baseline_commit': BASE,
    'model_name': '物理反演后CNN校正器', 'unfolded_training': False,
    'native_shape': [256, 256], 'train_mothers': 256, 'validation_mothers': 64,
    'registered_unopened_final_mothers': 64,
    'conditions': ['blank', 'weak', 'normal', 'strong'], 'independent_noise_replicates': 2,
    'clean_identity_per_mother': 1,
    'temperature_geometries': ['front', 'multipeak', 'ring', 'stripe', 'flat', 'smooth_random'],
    'template_geometries': ['blob', 'multipeak', 'ring', 'stripe', 'filament', 'smooth_random'],
    'selection_validation_mothers': 4,
    'backbones': ['old_init__joint_update_v3', 'consistent_init__joint_update_v3'],
    'template_regularization_candidates': [1e-3, 1e-2, 1e-1],
    'traditional_tikhonov_candidates': [1e-3, 1e-2, 1e-1],
    'warm_omission_rule': 'all 16 development gates; validation loss <= 1.05 times combined v3; otherwise keep warm start',
    'backbone_config': DEFAULTS,
    'CNN': {'inputs': 19, 'width': 16, 'shared_corrections': 6, 'parameters': 5797,
            'heads': [2, 3], 'head_initialization': 'zero', 'stage_features_recomputed': True},
    'training': {'optimizer': 'Adam', 'lr': 1e-4, 'batch': 1, 'accumulation': 4,
                 'gradient_clip': 1., 'optimizer_updates': 2000, 'validation_interval': 100,
                 'seed': 101, 'additional_seeds_only_after_clear_validation_gain': True},
    'loss': {'template_scale_e': 60., 'temperature_scale_K': 75., 'band_scale_e': 300.,
             'template': 1., 'color': .2, 'bands': .2, 'raw': .01,
             'signed_blank': 1., 'truth_source': 'synthesis only, never B4/v3'},
    'checkpoint_selection': 'lowest full-validation supervised loss, final test never opened',
    'detection': 'historical signed prediction >60 e, never raised for success',
    'overfit_updates': 200,
    'all_written_files_on': 'E/D', 'training_device': 'local CUDA RTX4060',
    'old_954_full_rerun': False, 'old_CNN_and_physics_files_preserved': True,
}
