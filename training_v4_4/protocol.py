"""Freeze the next finite intervention before any new development inference."""
import runtime
from training_v4.common import read, write
from training_v4_1.common import OUT as OLD
from evaluation_v2.common import canonical_hash

OUT = runtime.ROOT/'reports/training_v4_4/run_001'
WEIGHTS = runtime.ROOT/'training_v4_4/checkpoints'
INITIAL = runtime.ROOT/'training_v4/checkpoints/seed_101_final_2000.pt'
CALIBRATION = runtime.ROOT/'reports/training_v4_3/run_001/residual_scaled/coefficient_calibration.json'
ARMS = ('residual_scaled', 'negative_balanced')
SIGNED_CONDITIONS = ('negative_weak', 'negative_normal', 'positive_weak', 'positive_normal')


def register():
    path = OUT/'protocol.json'
    if path.exists():
        return read(path)
    draws = read(OLD/'sampling_sequence.json')['draws'][:500]
    rows = read(OLD/'data_manifest.json')['rows']
    validation = [r['case_id'] for r in rows if r['split']=='validation']
    assert len(draws)==500 and len(validation)==776
    value = {
        'registered_at': runtime.now(), 'baseline_commit': '623568a4f866859a9185f8ddb2ab16bc14c0d2bf',
        'name': 'v4.4 final continuum constraint',
        'zero_update_endpoints': {a: {'path': f'training_v4_3/checkpoints/{a}/update_0500.pt',
             'sha256': runtime.sha256(runtime.ROOT/f'training_v4_3/checkpoints/{a}/update_0500.pt')} for a in ARMS},
        'one_changed_variable': 'Only final continuum state equals detached cached physical state; six internal CNN stages unchanged.',
        'train_updates': 500, 'maximum_gpu_seconds_per_training_arm': 1800,
        'maximum_gpu_seconds_zero_audit': 600, 'checkpoint_interval': 100,
        'seed': 4101, 'lr': 3e-5, 'accumulation': 4, 'clip': 1., 'safe_weight': 1.,
        'initial_checkpoint': str(INITIAL.relative_to(runtime.ROOT)).replace('\\','/'),
        'initial_sha256': runtime.sha256(INITIAL), 'draws_sha256': canonical_hash(draws),
        'manifest_sha256': runtime.sha256(OLD/'data_manifest.json'),
        'validation_ids_sha256': canonical_hash(validation),
        'calibration_sha256': runtime.sha256(CALIBRATION),
        'original_gates_changed': False, 'old_eligible_definition_changed': False,
        'v4_4_additional_development_screen': {
            'signed_conditions': list(SIGNED_CONDITIONS), 'template_relative_MSE_factor': 1.05,
            'template_absolute_MSE_tolerance_e2': .25,
            'separate_signed_weak_pooled_L2_strictly_less_than': 1.,
            'all_development_invalid_output_fraction': 0.,
            'finite_all_frames': True,
            'interpretation': 'New development rules motivated by exposed failure cases; not independent validation or a confidence bound.'},
        'selection_rule': 'After completing the registered comparisons, among candidates passing old eligible and all new screens, minimize main weak L2; ties use normal RMSE then earlier checkpoint.',
        'noisy_signed_threshold_e': 60, 'parameters': 5797,
        'internal_continuum_still_trainable': True, 'final_test_used': False,
        'final_test_requires_frozen_candidate_and_separate_protocol': True,
        'public_push_authorized_in_this_stage': False,
        'hardware': 'Windows RYLAN RTX 4060 Laptop; one root-controlled GPU job at a time',
        'user_authorized_parallel_subagents': True,
        'subagent_requested_model': 'gpt-6-astra', 'subagent_requested_reasoning_effort': 'max',
        'host_default_service_tier': 'priority', 'actual_subagent_service_tier_receipt': None,
        'sources': {str(p.relative_to(runtime.ROOT)).replace('\\','/'): runtime.sha256(p)
                    for p in sorted((runtime.ROOT/'training_v4_4').glob('*.py'))},
    }
    write(path, value)
    return value


if __name__=='__main__':
    p = register()
    print({'registered': str(OUT/'protocol.json'), 'updates': p['train_updates'], 'final_test_used': False})
