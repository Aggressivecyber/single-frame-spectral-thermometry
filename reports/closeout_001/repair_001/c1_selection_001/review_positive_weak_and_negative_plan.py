"""Root metadata review; no inverse, optimizer, forward model or truth read."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from research_closeout import c1_selection as c

s = c.s
s.configure()
folder = c.REPORT_DIR
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
ref = c.panel.file_ref
store = c.SelectionStore()
reg, digest = c.verify_registration(store)
calls, prefix = c.dev.call_snapshot()
charged = sum(not row.get('reused') for row in calls)
assert charged == 5149 and len(calls) == 5689
ledger_sha = s.sha(s.REPORTS/'inverse_calls.jsonl')
native_path = folder/'analysis/3081b8b3242aacc888a9c5ce90b2832f61f5b50dd041ee2f329dc0951e6ac5b0/NATIVE_ANALYSIS.json'
native = read(native_path)
cpu_path = folder.parent/'c1_positive_weak_analysis_001/FINAL_REVIEW_RECEIPT.json'
cpu = read(cpu_path)
assert s.sha(cpu_path) == '9e1c7ab89e063154a45134a79244b413ff468ce20be4a52558031a9b22b0c0b4'
for key in ('findings', 'metadata'):
    c.panel.verify_ref(cpu[key])
spec = importlib.util.spec_from_file_location('ownedNT', folder.parent/'v3_tangent_probe_002/startup_observer.py')
nt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nt)
processes = [nt.process(pid) for pid in (27240, 57640)]
locks = [s.REPORTS/'PHASE_ACTIVE.json', folder/'PHASE_ACTIVE.json']
rows = c.collect_rows(reg, store, digest)
summary = c.summary(reg, rows)
assert len(summary['cells']) == 1080 and len(rows) == 62
done = [row for row in rows if row.get('status') == 'FINITE_NUMERICALLY_CERTIFIED']
assert len(done) == 62
checks = {
    'two_processes_exited': all(p['available'] is False for p in processes),
    'both_locks_absent': all(not p.exists() for p in locks),
    '18_finite_certified_scored': len(native['rows']) == 18 and all(r['original_status'] == 'FINITE_NUMERICALLY_CERTIFIED' for r in native['rows']),
    'new18_reused0': native['original_new_inverse_cells'] == 18 and native['original_reused_cells'] == 0,
    'all18_native_hashes_and26_metrics': native['all_reconstructed_canonical_latent_hashes_and_main_scoring_match'] and native['score_field_count_per_endpoint'] == 26,
    'full_65536_pixel_denominator': all(r['metrics']['domain_pixels'] == 65536 for r in native['rows']),
    'root_audit_no_inverse_optimizer_inner': all(native[k] == 0 for k in ('analysis_complete_inverse_calls', 'analysis_optimizer_calls', 'analysis_inner_solver_calls')),
    'root_audit_ledger_byte_unchanged': native['ledger_before'] == native['ledger_after'] and native['ledger_after']['sha256'] == ledger_sha,
    'independent_CPU_review_passed': cpu['all_review_checks_passed'] and cpu['all_frozen_objects_byte_unchanged'],
    'stderr_empty': (folder/'positive_weak_shard_002.stderr.txt').stat().st_size == 0,
    '62_of_1080_actual_cells': len(done) == 62,
}
assert all(checks.values()), checks
completion = {
    'schema': 'root-positive-weak-shard-completion-v001', 'time': s.now(), 'all_passed': True,
    'checks': checks, 'processes': processes, 'terminal': native['terminal'],
    'native_audit': ref(native_path), 'independent_CPU_review': ref(cpu_path),
    'case_denominator': 18, 'new_inverse_calls': 18, 'reused_cells': 0,
    'full_stage_planned': 1080, 'full_stage_finite': 62, 'full_stage_unexecuted': 1018,
    'distinct_parents_this_shard': 1, 'distinct_raws_this_shard': 1,
    'postblur_initializer_max_RMS_color_K': max(p['color_RMS_difference_K'] for p in native['pairwise_initializers']),
    'postblur_initializer_max_RMS_template_e': max(p['template_RMS_difference_e'] for p in native['pairwise_initializers']),
    'C1_color_RMSE_K_range': [min(r['metrics']['temperature_RMSE_K'] for r in native['rows']), max(r['metrics']['temperature_RMSE_K'] for r in native['rows'])],
    'C1_template_RMSE_e_range': [min(r['metrics']['template_RMSE_e'] for r in native['rows']), max(r['metrics']['template_RMSE_e'] for r in native['rows'])],
    'baseline_cached_metrics': [{'method': r['method'], 'temperature_RMSE_K': r['metrics']['temperature_RMSE_K'], 'template_RMSE_e': r['metrics']['template_RMSE_e']} for r in native['baseline_cached_metrics']],
    'all_actual_endpoint_gtols': sorted({r['numerical_evidence']['actual_requested_endpoint_gtol'] for r in native['rows']}),
    'stage_1e7_executed': False, 'charged_new_inverse_calls': charged, 'ledger_sha256': ledger_sha,
    'truth_scientific_selection': False, 'final_payloads_opened': False, 'whole_closeout_complete': False,
}
destination = folder/'PWEAK_SHARD_COMPLETION_VERIFICATION.json'
assert not destination.exists()
store.write(destination, completion)

raw_ids = ['v5_development_selection_mother_0000_negative_weak_noise0',
           'v5_development_selection_mother_0000_negative_normal_noise0']
refs = c.choose_shard(reg, raw_ids=raw_ids)
cells = [cell for cell in reg['planned_cells'] if cell['case_id'] in raw_ids]
assert len(cells) == 36 and all(cell['execution_role'] == 'NEW_INVERSE' for cell in cells)
assert all(not any(not r.get('reused') and r.get('method') == cell['method'] and r.get('case_id') == cell['case_id'] for r in calls) for cell in cells)
source_checks = {}
for name, expected in reg['source_sha256'].items():
    data = (s.ROOT/name).read_bytes()
    ast.parse(data, filename=name)
    index = subprocess.run(['git', 'show', ':'+name], cwd=s.ROOT, check=True, capture_output=True).stdout
    assert data == index and hashlib.sha256(data).hexdigest() == expected, name
    source_checks[name] = expected
assert len(source_checks) == 42
dev_used = sum(not r.get('reused') and r.get('phase') in c.DEVELOPMENT_PHASES for r in calls)
assert dev_used == 508 and dev_used+36 <= 3400 and charged+36 < 12000
reserve = store.budget().check(160*2**20)
assert s.sha(s.REPORTS/'inverse_calls.jsonl') == ledger_sha
plan = {
    'schema': 'root-two-negative-conditions-scheduling-review-v001', 'time': s.now(), 'all_passed': True,
    'checks': {'same_frozen_registration': True, '36_new_zero_reuse': True, 'no_prior_same_physical_attempt': True,
               '42_sources_AST_worktree_index_match': True, '661_registered_dependencies_verified': len(reg['read_dependencies']) == 661,
               'no_active_phase_lock': all(not p.exists() for p in locks)},
    'raw_ids': raw_ids, 'raw_references': [{'case_id': r['case_id'], 'raw_reference': r['raw_reference']} for r in refs],
    'registration_sha256': digest, 'source_sha256': source_checks,
    'ledger_sha256': ledger_sha, 'ledger_rows': len(calls), 'charged': charged,
    'planned_cells': 36, 'reused_cells': 0, 'maximum_new_complete_inverse_calls': 36,
    'maximum_inverse_wall_s_excluding_overhead': 6480, 'maximum_after_shard': charged+36,
    'full_stage_coverage_before': 62, 'full_stage_maximum_after': 98, 'full_stage_denominator': 1080,
    'semantic_development_used_before': dev_used, 'semantic_development_maximum_after': dev_used+36,
    'semantic_development_cap': 3400, 'human_cumulative_cap': 16000, 'current_stricter_guard': 12000,
    'reserve_check': reserve, 'scheduling_only_within_registered1080': True,
    'new_grid_or_method': False, 'new_inverse_calls_by_this_review': 0, 'selection_performed': False,
    'one_complete_mother_after_batch_is_not12_independent_mothers': True, 'final_payloads_opened': False,
}
assert all(plan['checks'].values())
store.write(folder/'NEGATIVE_PAIR_PLAN_REVIEW.json', plan)
print(json.dumps({'completion': ref(destination), 'plan': ref(folder/'NEGATIVE_PAIR_PLAN_REVIEW.json'),
                  'color_RMSE_range': completion['C1_color_RMSE_K_range'], 'template_RMSE_range': completion['C1_template_RMSE_e_range'],
                  'reserve': reserve['global_new_study'], 'charged': charged, 'new_planned': 36}, indent=2))
