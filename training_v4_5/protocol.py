"""Register the missing fixed-C / balanced-draws cell, only after v4.4 closure."""
import copy
import runtime
from training_v4.common import read, write
from evaluation_v2.common import canonical_hash

OUT=runtime.ROOT/'reports/training_v4_5/run_001'
WEIGHTS=runtime.ROOT/'training_v4_5/checkpoints'
INITIAL=runtime.ROOT/'training_v4/checkpoints/seed_101_final_2000.pt'
CALIBRATION=runtime.ROOT/'reports/training_v4_3/run_001/residual_scaled/coefficient_calibration.json'
BALANCED=runtime.ROOT/'reports/training_v4_3/run_001/negative_balanced'
PREVIOUS=runtime.ROOT/'reports/training_v4_4/run_001'


def register():
    path=OUT/'protocol.json'
    if path.exists(): return read(path)
    assert read(PREVIOUS/'fixed_continuum/status.json')['stage']=='COMPLETE'
    assert read(PREVIOUS/'fixed_continuum/training_receipt.json')['updates']==500
    selection=read(PREVIOUS/'candidate_selection.json')
    assert selection['candidate_count']==0 and selection['selected'] is None
    old=read(PREVIOUS/'protocol.json'); value=copy.deepcopy(old)
    sampling=read(BALANCED/'sampling_sequence.json')
    assert len(sampling['draws'])==500 and canonical_hash(sampling['draws'])==sampling['draw_sha256']
    value.update(name='v4.5 final-C constraint with existing negative-balanced draws',
        registered_at=runtime.now(), predecessor_protocol_sha256=runtime.sha256(PREVIOUS/'protocol.json'),
        predecessor_training_receipt_sha256=runtime.sha256(PREVIOUS/'fixed_continuum/training_receipt.json'),
        predecessor_candidate_selection_sha256=runtime.sha256(PREVIOUS/'candidate_selection.json'),
        independent_review_sha256=runtime.sha256(PREVIOUS/'science_review/NEXT_DECISION_REVIEW.md'),
        one_changed_variable='Relative to old v4.3 negative_balanced: only final C is fixed. Relative to v4.4: only the exact negative-balanced draws are changed.',
        paired_controls=['v4.3 residual_scaled','v4.3 negative_balanced','v4.4 fixed_continuum'],
        draws_sha256=sampling['draw_sha256'],
        sampling_source='reports/training_v4_3/run_001/negative_balanced/sampling_sequence.json',
        sampling_file_sha256=runtime.sha256(BALANCED/'sampling_sequence.json'),
        zero_update_endpoints={}, new_training_cases=0, restart_from_old_initial_not_continuation=True,
        hard_stop='Complete at most this one 500-update arm. If every checkpoint fails, stop new optimization on this 12-mother signed pool; no extra seeds, steps, losses or C variants.',
        inherited_screen_unchanged=True, final_test_used=False)
    value['sources']={str(p.relative_to(runtime.ROOT)).replace('\\','/'):runtime.sha256(p)
                      for p in sorted((runtime.ROOT/'training_v4_5').glob('*.py'))}
    write(path,value)
    return value


if __name__=='__main__':
    p=register(); print({'registered':str(OUT/'protocol.json'),'updates':p['train_updates'],'new_cases':0})
