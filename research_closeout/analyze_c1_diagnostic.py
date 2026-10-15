"""Read-only, full-denominator audit of the completed four-cell C1 diagnostic."""
from . import storage as s
from . import calibration_quick as c
from . import quick_validation as q
from datetime import datetime
import json
from pathlib import Path
import numpy as np


def main():
    registration, digest = c.verify_registration()
    result_path = c.REPORT_DIR/'QUICK_RESULT.json'
    result = s.read(result_path)
    planned = c.planned_cells(registration['references'])
    rows = result['summary']['cells']
    assert len(planned) == len(rows) == 4
    assert {(r['case_id'], r['method']) for r in rows} == {
        (r['case_id'], r['method']) for r in planned}
    assert result['registration_sha256'] == digest
    refs = {r['case_id']: r for r in registration['references']}
    report_rows, arrays_by_case, verified = [], {}, set()
    for row in rows:
        identity = c.identity_for(row, digest)
        for name in ('inference_receipt', 'score_receipt', 'output_cache', 'output_metadata'):
            q._verify_file_ref(row[name])
            verified.add(row[name]['path'])
        out = q._load_output(row, identity=identity)
        receipt = c.parameter_receipt(row, digest, registration)
        assert receipt is not None
        c.load_parameters(receipt, identity, registration)
        verified.update(receipt[name]['path'] for name in ('parameter_cache', 'parameter_metadata'))
        ref = refs[row['case_id']]
        for name in ('raw_reference', 'label_reference'):
            source = s.LEGACY/ref[name]['path']
            assert s.sha(source) == ref[name]['sha256']
            verified.add(str(source))
        with np.load(s.LEGACY/ref['label_reference']['path'], allow_pickle=False) as labels:
            domain = labels['evaluation_domain'].astype(bool)
            tu = labels['template_postblur_e'].astype(np.float64)
            tt = labels['temperature_color_K_reference'].astype(np.float64)
        u = q._array(out['template_postblur_e_ref']).squeeze()
        t = q._array(out['temperature_color_K']).squeeze()
        assert u.dtype == t.dtype == np.float64
        assert u.shape == t.shape == domain.shape == (256, 256)
        assert int(domain.sum()) == 65536
        du, dt = (u-tu)[domain], (t-tt)[domain]
        independent = {
            'template_RMSE_e': float(np.sqrt(np.mean(du*du))),
            'temperature_reference_color_RMSE_K': float(np.sqrt(np.mean(dt*dt))),
            'signed_projection': float(np.sum(u[domain]*tu[domain])/np.sum(tu[domain]**2)),
            'domain_pixels': int(domain.sum()),
        }
        for name, value in independent.items():
            assert np.isclose(value, row['metrics'][name], rtol=1e-12, atol=1e-10), name
        engine = row['info']['numerical_engine']
        stages = engine['stages']
        comparisons = engine['output_comparisons']
        assert len(stages) >= 3 and len(comparisons) >= 2
        assert all(x['qualified_endpoint'] for x in stages[-3:])
        assert all(x['passed'] for x in comparisons[-2:])
        assert all(x['gradient']['scaled_projected_linf'] <= x['configuration']['gtol']
                   for x in stages[-3:])
        for x in comparisons[-2:]:
            assert x['template_RMS_difference_e'] <= .17
            assert x['color_RMS_difference_K'] <= .2
        assert row['production_numerical_certified'] and row['finite_output']
        assert row['truth_loaded_after_inference'] and not row['truth_loaded_before_persisting']
        assert not row['labels_used_by_inference']
        compact = {
            'case_id': row['case_id'], 'initializer': row['initializer'],
            'finite': row['finite_output'], 'numerical_certified': row['production_numerical_certified'],
            'return_stage': engine['returned_stage'],
            'final_requested_gtol': stages[-1]['configuration']['gtol'],
            'final_true_scaled_projected_gradient': stages[-1]['gradient']['scaled_projected_linf'],
            'stages': [{key: x[key] for key in ('stage', 'configuration', 'gradient',
                        'qualified_endpoint', 'template_inner_difference_RMS_e')} for x in stages],
            'output_comparisons': comparisons, 'inverse_wall_s': engine['wall_s'],
            'independently_recomputed_metrics': independent,
            'v001_output_difference': row['v001_output_comparison']['differences'],
            'detector_available': row['metrics']['frame_detection_v2'].get('available', False),
        }
        report_rows.append(compact)
        arrays_by_case.setdefault(row['case_id'], []).append((row, t, u))
    cross = []
    for case, entries in arrays_by_case.items():
        assert len(entries) == 2
        (a, ta, ua), (b, tb, ub) = entries
        ea, eb = (x['info']['numerical_engine'] for x in (a, b))
        cross.append({'case_id': case,
            'color_RMS_difference_K': float(np.sqrt(np.mean((ta-tb)**2))),
            'template_RMS_difference_e': float(np.sqrt(np.mean((ua-ub)**2))),
            'objective_absolute_difference': abs(ea['stages'][-1]['returned_endpoint_objective']
                                                - eb['stages'][-1]['returned_endpoint_objective']),
            'same_physical_regularizer_contract': a['info']['anchor_regscale_contract']
                                                 == b['info']['anchor_regscale_contract']})
    ledger = list(s.rows(s.REPORTS/'inverse_calls.jsonl'))
    process = s.read(c.REPORT_DIR/'QUICK_PROCESS.json')
    elapsed = (datetime.fromisoformat(result['time'])-datetime.fromisoformat(process['time'])).total_seconds()
    actual_new = sum(not r.get('reused', False) for r in ledger)
    audit = {
        'time': s.now(), 'schema': 'C1v3-native-four-cell-root-audit-v001',
        'result_sha256': s.sha(result_path), 'registration_sha256': digest,
        'registered_source_count': len(registration['source_sha256']),
        'verified_input_receipt_and_cache_files': len(verified),
        'planned_cells': 4, 'finite_cells': 4, 'numerically_certified_cells': 4,
        'full_pixels_per_cell': 65536, 'recomputed_scoring_pixels': 4*65536,
        'new_complete_inverse_calls_in_completed_run': result['summary']['new_inverse_calls_this_run'],
        'global_actual_new_inverse_calls': actual_new, 'controller_elapsed_s': elapsed,
        'rows': report_rows, 'cross_initializer_comparison': cross,
        'source_and_cache_byte_hashes_verified': True,
        'calibration_and_physical_objective_explicitly_changed': True,
        'certification_rule': 'three consecutive qualified endpoints and two output platforms under tighter configurations',
        'all_four_stopped_at_stage3_gtol_1e_minus6': all(r['final_requested_gtol'] == 1e-6 for r in report_rows),
        'gtol_1e_minus7_or_global_optimum_claimed': False,
        'scientific_summary': result['summary']['by_method'],
        'scientific_method_selection_performed': False, 'final_payloads_opened': False,
        'limitations': ['two development raws, one fixed regularizer pair, no generalization or final selection',
                        'numerical certification is distinct from physical accuracy',
                        'stage1/2 latent parameters were not stored: their platform and gradients are frozen-log checks',
                        'final-stage independent endpoint audit is a separate additive report',
                        'original v001 endpoint latent states are unavailable; causal attribution remains bounded'],
        'storage_ledger': result['storage_ledger'], 'new_inverse_calls_by_this_audit': 0,
        'source_sha256': {'research_closeout/analyze_c1_diagnostic.py': s.sha(Path(__file__))},
        'whole_closeout_complete': False,
    }
    path = c.REPORT_DIR/'NATIVE_C1_ANALYSIS.json'
    s.write(path, audit)
    print(json.dumps({'path': str(path), 'sha256': s.sha(path), 'certified': '4/4',
        'elapsed_s': elapsed, 'global_actual_new_inverse_calls': actual_new, 'new_inverse_calls': 0}))


if __name__ == '__main__':
    main()
