"""Preserve old development decisions and report the separately frozen v4.4 screen."""
import math
import runtime
import torch
from training_v4.common import read, write
from training_v4_1.evaluate import evaluate as old_evaluate
from evaluation_v2.common import Journal
from training_v4_4.protocol import OUT, SIGNED_CONDITIONS
from training_v4_4.model import FixedContinuumCNN
from training_v4.model import physical_output


class AuditedFixedContinuumCNN(FixedContinuumCNN):
    """A read-only observer of the identical forward result, enabled only for evaluation."""
    def __init__(self, cfg, calibration):
        super().__init__(cfg, calibration)
        self.capture = False
        self.forward_audits = []

    def forward_electrons(self, raw_e, cached_state, cached_beta):
        result = super().forward_electrons(raw_e, cached_state, cached_beta)
        if self.capture:
            with torch.no_grad():
                base = physical_output(self.physics, cached_state, cached_beta, raw_e)
                same_c = torch.equal(result['state'][:, :2], cached_state[:, :2])
                same_s = torch.equal(result['state'][:, 2:], result['unrestricted_scaled_state'][:, 2:])
                same_t = torch.equal(result['temperature_color_K'], base['temperature_color_K'])
                assert same_c and same_s and same_t
                variance = raw_e.clamp_min(0)+self.physics.extra_variance
                self.forward_audits.append({'final_C_equals_physics': same_c,
                    'signed_state_equals_scaled_proposal': same_s, 'temperature_equals_physics': same_t,
                    'weighted_raw_RMS': float((result['raw_residual_e'].square()/variance).mean().sqrt()),
                    'physical_weighted_raw_RMS': float((base['raw_residual_e'].square()/variance).mean().sqrt()),
                    'maximum_template_correction_e': float((result['template_e']-base['template_e']).abs().max()),
                    'alpha': float(result['correction_coefficient']),
                    'initial_weighted_raw_RMS': float(result['initial_weighted_raw_RMS'])})
        return result


def scientific_screen(folder, update, *, protocol_path=None):
    protocol_path = protocol_path or OUT/'protocol.json'
    protocol = read(protocol_path)['v4_4_additional_development_screen']
    summary = read(folder/f'validation_summary_{update:04d}.json')
    records = Journal(folder/f'validation_update_{update:04d}.jsonl').rows()
    assert len(records)==776 and len({r['case_id'] for r in records})==776
    finite = all(r['status']=='SUCCESS' and math.isfinite(r['metrics']['template_MSE_e2'])
                 and math.isfinite(r['metrics']['temperature_MSE_K2']) for r in records)
    invalid = max(r['invalid_output_fraction'] for r in records)
    maximum_temperature_correction = max(r['temperature_correction_RMS_K'] for r in records)
    assert maximum_temperature_correction==0., 'The final continuum was not exactly fixed'
    signed = []
    for condition in SIGNED_CONDITIONS:
        rs = [r for r in records if r['input_kind']=='noise' and r['condition']==condition]
        assert len(rs)==6 and all(r['signed_aux'] for r in rs)
        model_mse = sum(r['metrics']['template_MSE_e2'] for r in rs)/len(rs)
        physical_mse = sum(r['physical']['template_MSE_e2'] for r in rs)/len(rs)
        limit = protocol['template_relative_MSE_factor']*physical_mse+protocol['template_absolute_MSE_tolerance_e2']
        energy = sum(r['metrics']['truth_energy'] for r in rs)
        l2 = math.sqrt(sum(r['metrics']['template_SSE'] for r in rs)/energy)
        projection = sum(r['metrics']['signed_projection']*r['metrics']['truth_energy'] for r in rs)/energy
        recall_key = 'negative_recall60' if condition.startswith('negative') else 'positive_recall60'
        recalls = [r['metrics'][recall_key] for r in rs if r['metrics'][recall_key] is not None]
        weak_ok = l2 < protocol['separate_signed_weak_pooled_L2_strictly_less_than'] if condition.endswith('_weak') else True
        signed.append({'condition': condition, 'frames': len(rs), 'template_MSE_e2': model_mse,
            'physical_template_MSE_e2': physical_mse, 'MSE_limit_e2': limit,
            'template_RMSE_e': math.sqrt(model_mse), 'template_L2': l2,
            'energy_weighted_signed_projection': projection,
            'macro_recall_at_fixed_60e': sum(recalls)/len(recalls) if recalls else None,
            'MSE_pass': model_mse<=limit, 'weak_L2_pass': weak_ok, 'passed': model_mse<=limit and weak_ok})
    added_pass = finite and invalid==0 and all(r['passed'] for r in signed)
    value = {'optimizer_update': update, 'old_eligible': summary['eligible'],
        'new_development_screen_pass': added_pass, 'science_candidate': summary['eligible'] and added_pass,
        'all_frames_finite': finite, 'maximum_invalid_output_fraction': invalid,
        'maximum_temperature_correction_RMS_K': maximum_temperature_correction,
        'signed_noisy_conditions': signed, 'validation_records': len(records),
        'validation_sha256': runtime.sha256(folder/f'validation_update_{update:04d}.jsonl'),
        'protocol_sha256': runtime.sha256(protocol_path), 'final_test_used': False}
    write(folder/f'science_summary_{update:04d}.json', value)
    return value


def evaluate(model, rows, selection, folder, update, *, protocol_path=None):
    model.forward_audits = []
    model.capture = True
    try:
        summary = old_evaluate(model, rows, selection, folder, update, 1.)
    finally:
        model.capture = False
    assert len(model.forward_audits)==len(rows)+16
    audits = Journal(folder/f'forward_audit_{update:04d}.jsonl')
    assert not audits.path.exists()
    for row, audit in zip(rows, model.forward_audits[:len(rows)]):
        audits.append({'case_id': row['case_id'], 'raw_hash': row['raw_hash'],
            'condition': row['condition'], 'input_kind': row['input_kind'], **audit})
    write(folder/f'original_gate_forward_audit_{update:04d}.json', model.forward_audits[len(rows):])
    screen = scientific_screen(folder, update, protocol_path=protocol_path)
    print({'update': update, 'gates': summary['original_gates_pass_count'],
        'protection': sum(g['passed'] for g in summary['protection_groups']),
        'weak_L2': summary['weak_L2'], 'normal_RMSE_e': summary['normal_RMSE_e'],
        'old_eligible': summary['eligible'], 'science_candidate': screen['science_candidate'],
        'signed_failures': [r['condition'] for r in screen['signed_noisy_conditions'] if not r['passed']]}, flush=True)
    return summary, screen
