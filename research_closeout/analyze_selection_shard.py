"""Read-only native byte reconstruction and rescoring of a finished shard.

No begin/resume, optimization, inner solve, raw generation or inverse counting
is used. A shard is a scheduling unit, not independent scientific replication.
"""
from . import storage as s
from . import c1_selection as selection
from . import c1_development as dev
from . import c1_small_panel as panel
from .analyze_small_panel import MAIN_FIELDS, _same, _ledger
import argparse
import json
from pathlib import Path
import time
import traceback
import uuid
import numpy as np


def run(shard_receipt, device='cuda'):
    from .smooth_calibration import physics, contract
    from .compact_joint_cache import runtime_identity
    from . import metrics
    import torch
    store = selection.SelectionStore()
    reg, digest = selection.verify_registration(store)
    shard_path = Path(shard_receipt).resolve()
    if store.report_root/'shards' not in shard_path.parents:
        raise ValueError('Choose a completed registered selection shard')
    terminal = panel.read_sealed(store, shard_path)
    assert terminal['status'] == 'FIRST_STAGE_SHARD_COMPLETE'
    assert terminal['registration_sha256'] == digest and not terminal['score_only']
    shard_id = terminal['shard_id']
    destination = store.report_root/'analysis'/shard_id/'NATIVE_ANALYSIS.json'
    if destination.exists():
        raise FileExistsError('Reuse the completed native analysis')
    refs = [ref for ref in reg['references'] if ref['case_id'] in terminal['raw_ids']]
    cells = [cell for cell in reg['planned_cells'] if cell['case_id'] in terminal['raw_ids']]
    assert len(refs)*18 == len(cells) == terminal['planned_shard_cells']
    states = {dev.semantic_key(row): row for row in selection.collect_rows(reg, store, digest)}
    for receipt in terminal['case_state_receipts']: panel.verify_ref(receipt)
    before = _ledger()
    started = time.perf_counter()
    rows, pairs, pending = [], [], {}
    with dev.phase_locks(store, digest, 'C1-first-stage-read-only-native-shard-analysis'):
        try:
            torch.set_num_threads(2)
            p = physics(device)
            assert p.M.device.type == 'cuda' and contract(p) == reg['calibration_contract']
            qualification = panel.read_sealed(store, store.report_root/selection.QUALIFICATION_NAME)
            panel.require_qualification(qualification, reg, digest, runtime_identity(p))
            s.frozen_bridge()
            from research_v6 import inputs
            for ref in refs:
                raw = inputs.load_raw(ref, p)
                for cell in (cell for cell in cells if cell['case_id'] == ref['case_id']):
                    prior = states[dev.semantic_key(cell)]
                    if not prior['finite_output']:
                        rows.append({**cell, 'original_status': prior['status'],
                            'native_reconstruction_performed': False,
                            'reason': 'No frozen finite endpoint; retained denominator, no refit'})
                        continue
                    if cell['execution_role'] == 'REUSE_PRIOR_C1V3':
                        locator = reg['reuse_catalog'][cell['reuse_key']]['locator']
                    else:
                        locator = {'kind': 'compact', 'report': prior['compact_report_root'],
                            'data': prior['compact_data_root'], 'case_id': cell['case_id'],
                            'method': cell['method'], 'raw_sha256': cell['raw_sha256'],
                            'expected_physical_options': cell['physical_options'],
                            'expected_calibration_fingerprint': reg['calibration_contract']['fingerprint']}
                    out, proof = selection.read_origin(locator, p, raw)
                    info = out['info']
                    evidence = dev.numerical_evidence(info, out['numerical_valid'])
                    assert evidence['passed'] == prior['production_numerical_certified']
                    truth = inputs.load_diagnostic_truth(ref, p)
                    fresh = s.plain(metrics.score(out, truth, raw, p, detector=None,
                        method=cell['method'], application_domain=selection.VERSION+'-12-mother-first-stage')['metrics'])
                    fields = MAIN_FIELDS+('matched_dictionary_frame_score', 'legacy_frame_detection_score',
                        'full_FOV_valid_fraction', 'raw_saturated_pixels', 'raw_lower_clipped_pixels')
                    matches = {key: _same(fresh.get(key), prior['metrics'].get(key)) for key in fields}
                    assert all(matches.values()), (cell['method'], matches)
                    assert fresh['domain_pixels'] == 65536
                    key = (cell['case_id'], cell['candidate_id'])
                    arrays = {name: out[name].detach().cpu().numpy() for name in
                              ('temperature_color_K', 'template_postblur_e_ref')}
                    if key in pending:
                        first = pending.pop(key)
                        pairs.append({'case_id': key[0], 'candidate_id': key[1],
                            'color_RMS_difference_K': float(np.sqrt(np.mean((arrays['temperature_color_K']-first['temperature_color_K'])**2))),
                            'template_RMS_difference_e': float(np.sqrt(np.mean((arrays['template_postblur_e_ref']-first['template_postblur_e_ref'])**2)))})
                    else:
                        pending[key] = arrays
                    rows.append({**cell, 'original_status': prior['status'], 'native_reconstruction_performed': True,
                        'byte_proof': proof, 'numerical_evidence': evidence, 'original_info': info,
                        'metrics': fresh, 'scoring_matches': matches,
                        'canonical_array_descriptors': {name: selection.compact.descriptor(out[name]) for name in
                            selection.compact.CANONICAL_KEYS+('continuum_latent',)}})
                    print('AUDITED '+cell['case_id']+' '+cell['method'], flush=True)
            assert len(rows) == len(cells)
            after = _ledger()
            assert before == after
            baseline_rows = []
            for name in ('development_baseline_reuse.jsonl', 'development_results.jsonl'):
                source = s.REPORTS/name
                for row in s.rows(source):
                    if row.get('case_id') in terminal['raw_ids'] and row.get('method') in (
                            'v3_physics', 'old_B4', 'B1_lambda100', 'B1_lambda0.1'):
                        ref = next(ref for ref in refs if ref['case_id'] == row['case_id'])
                        assert row['raw_sha256'] == ref['raw_reference']['sha256']
                        baseline_rows.append({**row, 'cached_metric_source': panel.file_ref(source)})
            value = {'schema': 'C1-selection-shard-native-analysis-v001', 'time': s.now(),
                'terminal': panel.file_ref(shard_path), 'registration_sha256': digest,
                'analysis_source': panel.file_ref(Path(__file__)), 'planned_shard_cells': len(cells),
                'native_reconstructed_cells': sum(row['native_reconstruction_performed'] for row in rows),
                'original_new_inverse_cells': terminal['new_inverse_calls_this_shard'],
                'original_reused_cells': sum(cell['execution_role'] == 'REUSE_PRIOR_C1V3' for cell in cells),
                'distinct_raw_inputs': len(refs), 'distinct_parent_mothers': len({ref['mother_id'] for ref in refs}),
                'full_first_stage_denominator': 1080,
                'all_reconstructed_canonical_latent_hashes_and_main_scoring_match': True,
                'score_field_count_per_endpoint': len(MAIN_FIELDS)+5, 'pairwise_initializers': pairs,
                'incomplete_finite_initializer_pairs': len(pending), 'rows': rows,
                'baseline_cached_metrics': baseline_rows,
                'baseline_scope': 'same-raw frozen metric reuse, no baseline inverse or rescoring',
                'runtime': runtime_identity(p), 'ledger_before': before, 'ledger_after': after,
                'analysis_complete_inverse_calls': 0, 'analysis_optimizer_calls': 0,
                'analysis_inner_solver_calls': 0, 'wall_s': time.perf_counter()-started,
                'scientific_method_selection': False, 'pixel_counts_are_not_independent_scientific_units': True,
                'final_payloads_opened': False, 'whole_closeout_complete': False}
            store.write(destination, s.plain(value))
            return value
        except Exception as error:
            store.write(destination.with_name('FAILURE_'+uuid.uuid4().hex+'.json'),
                {'time': s.now(), 'error_type': type(error).__name__, 'error': str(error),
                 'traceback': traceback.format_exc(), 'analysis_complete_inverse_calls': 0,
                 'audited_rows_before_error': len(rows), 'ledger_before': before, 'ledger_after': _ledger()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard-receipt', required=True, type=Path)
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    args = parser.parse_args()
    value = run(args.shard_receipt, args.device)
    print(json.dumps({key: value[key] for key in ('planned_shard_cells', 'native_reconstructed_cells',
        'all_reconstructed_canonical_latent_hashes_and_main_scoring_match',
        'analysis_complete_inverse_calls', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
