"""Read-only native reconstruction and external scoring of all18 prescreen cells."""
from . import storage as s
from . import c1_development as d
from . import c1_small_panel as panel
from .analyze_small_panel import MAIN_FIELDS, _same, _ledger
import argparse
import json
from pathlib import Path
import time
import numpy as np


def run(device='cuda'):
    from .smooth_calibration import physics, contract
    from .compact_joint_cache import runtime_identity
    from . import metrics
    import torch
    store = d.EStudyStore()
    destination = d.REPORT_DIR/'NATIVE_PRESCREEN_ANALYSIS.json'
    if destination.exists():
        raise FileExistsError('Retain and reuse the completed native prescreen audit')
    registration, digest = d.verify_registration(store)
    terminal_path = d.REPORT_DIR/d.RESULT_NAME
    terminal = panel.read_sealed(store, terminal_path)
    assert terminal['status'] == 'C1V3_PRESCREEN_COMPLETE'
    assert terminal['registration_sha256'] == digest
    assert terminal['summary']['planned_cells'] == terminal['summary']['returned_cell_rows'] == 18
    assert terminal['summary']['numerically_qualified_candidates'] == 9
    before = _ledger()
    assert before['charged_new_calls'] == 5115 and before['rows'] == 5655
    started = time.perf_counter()
    rows, pairs, pending = [], [], {}
    with d.phase_locks(store, digest, 'C1-prescreen-root-read-only-forward-audit'):
        torch.set_num_threads(2)
        p = physics(device)
        assert p.M.device.type == 'cuda' and contract(p) == registration['calibration_contract']
        qualification = panel.read_sealed(store, d.REPORT_DIR/d.QUALIFICATION_NAME)
        panel.require_qualification(qualification, registration, digest, runtime_identity(p))
        s.frozen_bridge()
        from research_v6 import inputs
        reference = registration['reference']
        raw = inputs.load_raw(reference, p)
        for cell in registration['planned_cells']:
            prior = next(row for row in terminal['summary']['cells']
                         if d.semantic_key(row) == d.semantic_key(cell))
            cache = d.cache_for(store, p, cell, registration, digest,
                                qualification=cell['execution_role'] == 'REUSE_PRIOR_C1V3')
            assert all((cache.report/name).is_file() for name in ('PREPARED.json', 'PARAMETERS.json', 'RESULT.json'))
            # No begin/resume/inference or persistence mutation in this audit.
            out, prepared, receipt, verified = cache._verify_forward(raw)
            assert verified['all_byte_identical']
            assert prepared['original_numerical_valid'] is True and out['numerical_valid'] is True
            evidence = d.numerical_evidence(prepared['original_info'], True)
            assert evidence['passed']
            truth = inputs.load_diagnostic_truth(reference, p)
            fresh = metrics.score(out, truth, raw, p, detector=None, method=cell['method'],
                                  application_domain=d.VERSION+'-single-development-raw')['metrics']
            matches = {key: _same(fresh.get(key), prior['metrics'].get(key)) for key in MAIN_FIELDS}
            assert all(matches.values()), (cell['method'], matches)
            assert fresh['domain_pixels'] == fresh['accepted_pixels'] == 65536
            values = {key: out[key].detach().cpu().numpy() for key in
                      ('temperature_color_K', 'template_postblur_e_ref')}
            key = cell['candidate_id']
            if key in pending:
                first = pending.pop(key)
                pairs.append({'candidate_id': key, 'case_id': cell['case_id'], 'pixels': 65536,
                    'color_RMS_difference_K': float(np.sqrt(np.mean((values['temperature_color_K']-first['temperature_color_K'])**2))),
                    'template_RMS_difference_e': float(np.sqrt(np.mean((values['template_postblur_e_ref']-first['template_postblur_e_ref'])**2)))})
            else:
                pending[key] = values
            rows.append({**cell, 'metrics': fresh, 'numerical_evidence': evidence,
                'original_info': prepared['original_info'], 'parameter_receipt': receipt,
                'all_canonical_and_latent_hashes_match': True, 'scoring_matches': matches,
                'canonical_hashes': verified['reconstructed_canonical_arrays'],
                'continuum_latent_hash': verified['reconstructed_continuum_latent']})
            print('audited '+cell['method'], flush=True)
        assert len(rows) == 18 and len(pairs) == 9 and not pending
        after = _ledger()
        assert before == after
        baseline_rows = []
        for name in ('development_baseline_reuse.jsonl', 'development_results.jsonl'):
            source = s.REPORTS/name
            for row in s.rows(source):
                if row.get('case_id') == d.CASE_ID and row.get('method') in (
                        'v3_physics', 'old_B4', 'B1_lambda100', 'B1_lambda0.1'):
                    assert row['raw_sha256'] == reference['raw_reference']['sha256']
                    assert row['metrics']['domain_pixels'] == 65536
                    baseline_rows.append({**row, 'cached_metric_source': panel.file_ref(source)})
        assert len(baseline_rows) == 4
        value = {'schema': 'C1-prescreen-native-root-analysis-v001', 'time': s.now(),
            'terminal': panel.file_ref(terminal_path), 'registration_sha256': digest,
            'analysis_source': panel.file_ref(Path(__file__)),
            'analysis_shared_helpers': panel.file_ref(s.ROOT/'research_closeout/analyze_small_panel.py'),
            'planned_cells': 18, 'native_reconstructed_cells': 18, 'original_new_inverse_cells': 16,
            'original_reused_cells': 2, 'distinct_raw_inputs': 1, 'distinct_parent_mothers': 1,
            'recomputed_scoring_pixels': 18*65536, 'qualified_candidates': 9,
            'all_canonical_latent_hashes_and_scoring_match': True,
            'pairwise_initializers': pairs, 'rows': rows,
            'baseline_cached_metrics': baseline_rows,
            'baseline_scope': 'same-raw frozen metric reuse; no baseline inverse or rescoring',
            'runtime': runtime_identity(p), 'ledger_before': before, 'ledger_after': after,
            'analysis_complete_inverse_calls': 0, 'analysis_optimizer_calls': 0,
            'analysis_inner_solver_calls': 0, 'wall_s': time.perf_counter()-started,
            'scientific_method_selection': False, 'pixel_counts_are_not_independent_scientific_units': True,
            'final_payloads_opened': False, 'whole_closeout_complete': False}
        store.write(destination, s.plain(value))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    args = parser.parse_args()
    value = run(args.device)
    print(json.dumps({key: value[key] for key in ('planned_cells', 'qualified_candidates',
        'all_canonical_latent_hashes_and_scoring_match', 'analysis_complete_inverse_calls', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
