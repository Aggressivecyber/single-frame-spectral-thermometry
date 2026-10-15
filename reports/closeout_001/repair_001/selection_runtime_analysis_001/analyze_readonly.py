"""Read-only CPU timeline audit; exactly three tree_ledger timing calls.

No numerical module, truth, NPZ payload, forward, optimizer or GPU is run.
Only this new report directory may receive new exclusive files.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import ast
import hashlib
import json
import statistics
import io
import zipfile
import time
import sys

ROOT = Path('E:/CodexWork/thermometry/closeout_001').resolve()
DEST = ROOT/'reports/closeout_001/repair_001/selection_runtime_analysis_001'
SELECTION = ROOT/'reports/closeout_001/repair_001/c1_selection_001'
PRESCREEN = ROOT/'reports/closeout_001/repair_001/c1_development_001'
NEXT_RAW = 'v5_development_selection_mother_0000_positive_weak_noise0'
BLANK_RAW = 'v5_development_selection_mother_0000_blank_noise0'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def stamp(value): return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
def iso(value): return datetime.fromtimestamp(value, timezone.utc).isoformat()


def ref(path):
    path = Path(path).resolve(); stat = path.stat()
    return {'path': str(path), 'sha256': sha(path), 'bytes': stat.st_size,
            'mtime_utc': iso(stat.st_mtime), 'mtime_ns': stat.st_mtime_ns}


def sealed(path):
    path = Path(path).resolve(); sidecar = path.with_name(path.name+'.sha256.json')
    meta = read(sidecar)
    if Path(meta['path']).resolve() != path or meta['sha256'] != sha(path):
        raise RuntimeError('Changed sealed receipt: '+str(path))
    return read(path)


def event(path, *, must_seal=True):
    path = Path(path)
    if not path.exists(): return None
    value = sealed(path) if must_seal else read(path)
    result = {'file': ref(path), 'record_time_utc': value.get('time')}
    if value.get('time'):
        result['record_to_mtime_s'] = path.stat().st_mtime-stamp(value['time'])
    sidecar = path.with_name(path.name+'.sha256.json')
    if sidecar.exists(): result['sha_sidecar'] = ref(sidecar)
    return result


def cell_key(cell, digest):
    value = {'registration_sha256': digest, 'method': cell['method'], 'raw_sha256': cell['raw_sha256']}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def row_key(cell):
    value = [cell['case_id'], cell['method'], cell['initializer'], cell['raw_sha256']]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def dt(events, first, last):
    a, b = events.get(first), events.get(last)
    if not a or not b or not a.get('record_time_utc') or not b.get('record_time_utc'): return None
    return stamp(b['record_time_utc'])-stamp(a['record_time_utc'])


def stat_summary(values):
    values = [float(value) for value in values if value is not None]
    return {'n': len(values), 'sum_s': sum(values), 'min_s': min(values) if values else None,
            'median_s': statistics.median(values) if values else None,
            'max_s': max(values) if values else None}


def file_groups(root):
    files = list(Path(root).rglob('*'))
    files = [path for path in files if path.is_file()]
    buckets = Counter()
    for path in files:
        name = path.name
        kind = ('parameters_npz' if name == 'PARAMETERS.npz' else 'dense_reserve' if name.startswith('FALLBACK_RESERVED_')
            else 'fallback_npz' if name == 'FALLBACK.npz' else 'pending' if '.pending.' in name
            else 'json_sidecar' if name.endswith('.sha256.json') else 'json' if path.suffix == '.json'
            else 'other')
        buckets[kind] += path.stat().st_size
    return {'files': len(files), 'bytes': sum(buckets.values()), 'by_kind_bytes': dict(buckets)}


def gather_cells(base, cells, digest, *, selection):
    results = []
    for cell in cells:
        if cell.get('execution_role') != 'NEW_INVERSE': continue
        receipt_root = base/'cell_receipts'/cell_key(cell, digest)
        score_value = sealed(receipt_root/'SCORE_SUCCESS.json')
        row = score_value['row']
        report = Path(row['compact_report_root']); data = Path(row['compact_data_root'])
        prepared = sealed(report/'PREPARED.json')
        engine = prepared['original_info']['numerical_engine']
        events = {name: event(path) for name, path in (
            ('started', report/'STARTED.json'), ('reservation', report/'RESERVATION.json'),
            ('call', receipt_root/'CALL_RECORDED.json'), ('prepared', report/'PREPARED.json'),
            ('parameters', report/'PARAMETERS.json'), ('result', report/'RESULT.json'),
            ('cache_score', report/'SCORE_panel_external_v001.json'), ('score_success', receipt_root/'SCORE_SUCCESS.json'))}
        if selection:
            states = sorted((base/'case_states'/row_key(cell)).glob('STATE_*.json'))
            states = [path for path in states if not path.name.endswith('.sha256.json')]
            events['case_state'] = event(states[-1]) if states else None
        gaps = {first+'_to_'+last+'_s': dt(events, first, last) for first, last in (
            ('started', 'call'), ('call', 'prepared'), ('prepared', 'parameters'), ('parameters', 'result'),
            ('prepared', 'result'), ('result', 'cache_score'), ('cache_score', 'score_success'),
            ('result', 'score_success'), ('score_success', 'case_state'), ('started', 'case_state'))}
        if gaps['call_to_prepared_s'] is not None:
            gaps['call_to_prepared_minus_recorded_engine_s'] = gaps['call_to_prepared_s']-engine['wall_s']
        parameters = sealed(report/'PARAMETERS.json')['parameter_cache']
        if sha(parameters['path']) != parameters['sha256']: raise RuntimeError('Changed parameter archive')
        results.append({'case_id': cell['case_id'], 'method': cell['method'], 'candidate_id': cell['candidate_id'],
            'initializer': cell['initializer'], 'numerically_certified': row['production_numerical_certified'],
            'finite_output': row['finite_output'], 'score_status': row['score_status'],
            'engine_wall_s': engine['wall_s'], 'engine_returned_stage': engine['returned_stage'],
            'engine_stage_wall_s': [stage['wall_s'] for stage in engine['stages']],
            'engine_stage_optimizer_evaluations': [stage['optimizer'].get('function_evaluations') for stage in engine['stages']],
            'engine_total_evaluations': engine.get('outer_history_summary', {}).get('evaluations'),
            'endpoint_gtol': next(stage['configuration']['gtol'] for stage in engine['stages'] if stage['stage'] == engine['returned_stage']),
            'events': events, 'gaps_between_record_creation_times': gaps,
            'parameter_archive': ref(parameters['path']), 'compact_report_files': file_groups(report),
            'compact_data_files': file_groups(data)})
    return sorted(results, key=lambda item: stamp(item['events']['started']['record_time_utc']))


def source_calls(paths):
    output = []
    names = {'tree_ledger', 'ledger', 'check', '_budget_check', 'write_path', 'write', 'read_sealed',
             '_write', '_write_parameters', 'verify_registration', 'verify_sources', 'verify_ref', 'verify_file',
             'load_for_score', 'forward_reconstruct', '_verify_forward', 'persist_row', 'collect_rows'}
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf8'))
        class Visitor(ast.NodeVisitor):
            def __init__(self): self.stack = []
            def visit_FunctionDef(self, node):
                self.stack.append(node.name); self.generic_visit(node); self.stack.pop()
            def visit_ClassDef(self, node):
                self.stack.append(node.name); self.generic_visit(node); self.stack.pop()
            def visit_Call(self, node):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
                if name in names:
                    output.append({'path': path.relative_to(ROOT).as_posix(), 'line': node.lineno,
                                   'scope': '.'.join(self.stack), 'called_name': name})
                self.generic_visit(node)
        Visitor().visit(tree)
    return output


def fixture_audit(base, registration):
    folder = base/'cpu_contracts'/'fixtures'
    receipts = []
    for path in (base/'cpu_contracts').glob('*.json'):
        value = read(path)
        if value.get('fixture_root'):
            receipts.append({'file': ref(path), 'fixture_root': str(Path(value['fixture_root']).resolve()),
                'all_passed': value.get('all_passed'), 'GPU_used': value.get('GPU_used'),
                'formal_complete_inversions': value.get('formal_complete_inversions'),
                'truth_loaded': value.get('truth_loaded'), 'source_sha256': value.get('source_sha256', {})})
    dependency_hits = [item for item in registration['read_dependencies'] if folder.resolve() in Path(item['path']).resolve().parents]
    import numpy as np
    runs = []
    for run in sorted(folder.iterdir() if folder.exists() else []):
        if not run.is_dir(): continue
        matching = [value for value in receipts if Path(value['fixture_root']) == run.resolve()]
        headers = []
        for archive in run.rglob('*.npz'):
            try:
                with zipfile.ZipFile(archive) as zipped:
                    for member in zipped.namelist():
                        # Header metadata only; no actual array payload is loaded.
                        with zipped.open(member) as handle:
                            version = np.lib.format.read_magic(handle)
                            shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
                        headers.append({'path': str(archive), 'member': member, 'shape': list(shape),
                            'dtype': str(dtype), 'fortran_order': fortran})
            except Exception as error:
                headers.append({'path': str(archive), 'header_error': type(error).__name__+': '+str(error)})
        flags = bool(matching and all(value['all_passed'] is True and value['GPU_used'] is False
            and value['formal_complete_inversions'] == 0 and value['truth_loaded'] is False for value in matching))
        runs.append({'path': str(run.resolve()), 'ledger': file_groups(run), 'completion_receipts': matching,
            'all_matched_receipts_passed_CPU_zero_formal_inverse': flags,
            'classification': 'completed_CPU_synthetic_fixture' if flags else 'orphan_or_incomplete_CPU_fixture_requires_root_failure_summary',
            'npz_headers_only': headers})
    return {'fixture_root': str(folder.resolve()), 'ledger': file_groups(folder),
        'registration_read_dependency_hits_inside_fixture_root': dependency_hits, 'runs': runs,
        'all_runs_have_passed_zero_inverse_CPU_receipt': bool(runs) and all(run['all_matched_receipts_passed_CPU_zero_formal_inverse'] for run in runs),
        'test_source': ref(ROOT/('research_closeout/tests/c1_selection.py' if base == SELECTION else 'research_closeout/tests/c1_development.py')),
        'reconstruction_scope': 'test code uses fixed handcrafted small CPU arrays and injected failures, not copied scientific raw/latent payloads; UUID/time/owner/file hashes are run-specific, so exact old directory bytes are not reproducible from a seed',
        'cleanup_scope': 'root decision only; retain top-level contract receipts, source, this file/shape manifest and failure summaries; never delete real inference/qualification/case_states/cell_receipts/raw/v3 caches'}


def main():
    if DEST.exists() and (DEST/'READONLY_RUNTIME_ANALYSIS.json').exists(): raise FileExistsError('Never overwrite an analysis receipt')
    DEST.mkdir(parents=True, exist_ok=True)
    reg = sealed(SELECTION/'SELECTION_REGISTRATION.json'); digest = sha(SELECTION/'SELECTION_REGISTRATION.json')
    old_reg = sealed(PRESCREEN/'PRESCREEN_REGISTRATION.json'); old_digest = sha(PRESCREEN/'PRESCREEN_REGISTRATION.json')
    current = read(SELECTION/'SUMMARY_CURRENT.json'); prescreen = sealed(PRESCREEN/'PRESCREEN_RESULT.json')
    phase = 'C1v3_selection_first_stage_v001'
    calls_path = ROOT/'reports/closeout_001/inverse_calls.jsonl'
    calls = [json.loads(line) for line in calls_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    tracked = [ROOT/name for name in reg['source_sha256']]
    tracked += [SELECTION/'SELECTION_REGISTRATION.json', SELECTION/'SUMMARY_CURRENT.json',
        SELECTION/'BLANK_SHARD_PROCESS.json', SELECTION/'blank_shard_001.stdout.txt', SELECTION/'blank_shard_001.stderr.txt',
        PRESCREEN/'PRESCREEN_REGISTRATION.json', PRESCREEN/'PRESCREEN_RESULT.json', calls_path]
    for pattern in ('inference/**/*.json', 'case_states/**/*.json', 'cell_receipts/**/*.json', 'shards/**/*.json'):
        tracked += list(SELECTION.glob(pattern))
    before = {str(path): sha(path) for path in tracked}
    for name, expected in reg['source_sha256'].items():
        if sha(ROOT/name) != expected: raise RuntimeError('Frozen source changed: '+name)
    blank_cells = [cell for cell in reg['planned_cells'] if cell['case_id'] == BLANK_RAW]
    next_cells = [cell for cell in reg['planned_cells'] if cell['case_id'] == NEXT_RAW]
    cells = gather_cells(SELECTION, blank_cells, digest, selection=True)
    old_cells = gather_cells(PRESCREEN, old_reg['planned_cells'], old_digest, selection=False)
    shard_files = [path for path in (SELECTION/'shards').rglob('RUN_*.json') if not path.name.endswith('.sha256.json')]
    if len(shard_files) != 1: raise RuntimeError('Expected exactly one immutable completed shard')
    shard = sealed(shard_files[0]); launch = read(SELECTION/'BLANK_SHARD_PROCESS.json')
    old_launch = read(PRESCREEN/'PRESCREEN_PROCESS.json')
    charged = [call for call in calls if not call.get('reused')]
    phase_calls = [call for call in charged if call.get('phase') == phase]
    development_phases = ('development', 'repair_quick', 'C1v3_small_panel_v003', 'C1v3_development_prescreen_v001', phase)
    semantic_dev = [call for call in charged if call.get('phase') in development_phases]
    old_methods = {row['method']: row for row in old_cells}
    paired = [{'method': row['method'], 'blank_engine_s': row['engine_wall_s'],
        'positive_normal_engine_s': old_methods[row['method']]['engine_wall_s'],
        'difference_s': row['engine_wall_s']-old_methods[row['method']]['engine_wall_s'],
        'blank_evaluations': row['engine_total_evaluations'],
        'positive_normal_evaluations': old_methods[row['method']]['engine_total_evaluations']} for row in cells]
    gaps = {name: stat_summary(row['gaps_between_record_creation_times'].get(name) for row in cells)
            for name in cells[0]['gaps_between_record_creation_times']}
    for index, row in enumerate(cells):
        if index+1 < len(cells):
            row['next_new_started_minus_case_record_s'] = stamp(cells[index+1]['events']['started']['record_time_utc'])-stamp(row['events']['case_state']['record_time_utc'])
    timeline = {'launch_start_utc': launch['start_utc'], 'summary_record_time_utc': current['time'],
        'summary_file_mtime_utc': iso((SELECTION/'SUMMARY_CURRENT.json').stat().st_mtime),
        'shard_record_time_utc': shard['time'], 'shard_file_mtime_utc': iso(shard_files[0].stat().st_mtime),
        'stdout_last_write_utc': iso((SELECTION/'blank_shard_001.stdout.txt').stat().st_mtime),
        'launch_to_summary_record_s': stamp(current['time'])-stamp(launch['start_utc']),
        'launch_to_summary_mtime_s': (SELECTION/'SUMMARY_CURRENT.json').stat().st_mtime-stamp(launch['start_utc']),
        'launch_to_shard_record_s': stamp(shard['time'])-stamp(launch['start_utc']),
        'launch_to_stdout_last_write_s': (SELECTION/'blank_shard_001.stdout.txt').stat().st_mtime-stamp(launch['start_utc']),
        'pre_first_STARTED_record_s': stamp(cells[0]['events']['started']['record_time_utc'])-stamp(launch['start_utc']),
        'last_case_record_to_summary_record_s': stamp(current['time'])-stamp(cells[-1]['events']['case_state']['record_time_utc']),
        'sum_engine_wall_s': sum(row['engine_wall_s'] for row in cells),
        'sum_next_new_started_minus_case_record_s': sum(row.get('next_new_started_minus_case_record_s', 0) for row in cells)}
    timeline['launch_to_summary_record_minus_engine_s'] = timeline['launch_to_summary_record_s']-timeline['sum_engine_wall_s']
    old_start = old_launch.get('start_utc', old_launch.get('requested_utc'))
    comparison = {'different_raw_condition_not_a_controlled_replay': True,
        'prescreen_raw': old_reg['reference']['case_id'], 'selection_raw': BLANK_RAW,
        'prescreen_launch_start_utc': old_start, 'prescreen_terminal_record_utc': prescreen['time'],
        'prescreen_launch_to_terminal_record_s': stamp(prescreen['time'])-stamp(old_start),
        'prescreen_sum_engine_wall_s': sum(row['engine_wall_s'] for row in old_cells),
        'prescreen_launch_to_terminal_minus_engine_s': stamp(prescreen['time'])-stamp(old_start)-sum(row['engine_wall_s'] for row in old_cells),
        'engine_stats_by_initializer': {initializer: {
            'blank': stat_summary(row['engine_wall_s'] for row in cells if row['initializer'] == initializer),
            'positive_normal': stat_summary(row['engine_wall_s'] for row in old_cells if row['initializer'] == initializer)} for initializer in ('v3', 'smooth_demod')},
        'paired16_methods': paired}
    # Exactly3 plain CPU metadata traversals. No budget/check/controller method
    # is invoked, and there is no instrumentation of an inverse or forward.
    sys.path.insert(0, str(ROOT))
    from research_closeout.compact_joint_cache import tree_ledger
    roots = tuple(Path(path) for path in reg['config']['global_budget_roots'])
    scans = []
    for index in range(3):
        begin = time.perf_counter(); ledger = tree_ledger(roots); elapsed = time.perf_counter()-begin
        scans.append({'index': index+1, 'wall_s': elapsed, 'roots': [str(path) for path in roots], 'ledger': ledger})
        print('TREE_LEDGER_CPU_SNAPSHOT '+json.dumps(scans[-1]), flush=True)
    source_paths = [ROOT/name for name in ('research_closeout/c1_selection.py', 'research_closeout/c1_development.py',
        'research_closeout/c1_small_panel.py', 'research_closeout/compact_joint_cache.py', 'research_closeout/storage.py')]
    fixtures = [fixture_audit(base, reg) for base in (PRESCREEN, SELECTION)]
    total_fixture_files = sum(value['ledger']['files'] for value in fixtures)
    total_fixture_bytes = sum(value['ledger']['bytes'] for value in fixtures)
    fixture_total = {'files': total_fixture_files, 'bytes': total_fixture_bytes,
        'fraction_of_current_global_files': total_fixture_files/scans[-1]['ledger']['files'],
        'fraction_of_current_global_bytes': total_fixture_bytes/scans[-1]['ledger']['bytes'],
        'current_global_denominator': scans[-1]['ledger']}
    after = {str(path): sha(path) for path in tracked}
    planned_next_old = [cell for cell in next_cells if cell['execution_role'] != 'NEW_INVERSE']
    result = {'schema': 'selection-runtime-readonly-analysis-v001', 'time': datetime.now(timezone.utc).isoformat(),
        'execution_scope': {'GPU': 0, 'inverse': 0, 'inner_solve': 0, 'forward': 0, 'truth_load': 0,
            'NPZ_payload_loads': 0, 'formal_controller_entry_points': 0, 'tree_ledger_CPU_snapshots': 3},
        'registration_sha256': digest, 'frozen_sources_verified': len(reg['source_sha256']),
        'terminal_audit': {'shard_status': shard['status'], 'planned_shard_cells': shard['planned_shard_cells'],
            'new_inverse_calls_this_shard': shard['new_inverse_calls_this_shard'],
            'new16_finite_certified_scored': all(row['finite_output'] and row['numerically_certified'] and row['score_status'] == 'SCORED' for row in cells),
            'full_matrix': {key: current['summary'][key] for key in ('planned_cells', 'state_rows_present', 'finite_output_cells',
                'numerically_certified_cells', 'not_executed_cells', 'scoring_failed_cells', 'new_inverse_calls_selection_total', 'independent_sample_units')},
            'stderr_bytes': (SELECTION/'blank_shard_001.stderr.txt').stat().st_size,
            'global_lock_exists': (ROOT/'reports/closeout_001/PHASE_ACTIVE.json').exists(),
            'local_lock_exists': (SELECTION/'PHASE_ACTIVE.json').exists(),
            'full_stage_final_exists': (SELECTION/'FIRST_STAGE_RESULT.json').exists()},
        'timeline': timeline, 'new16_cells': cells, 'gap_stats': gaps, 'prescreen_comparison': comparison,
        'tree_ledger_CPU_snapshots': scans, 'tree_ledger_scan_wall_s': stat_summary(row['wall_s'] for row in scans),
        'CPU_fixture_audit': fixtures, 'CPU_fixture_combined_fraction': fixture_total,
        'accounting_source_call_sites': source_calls(source_paths),
        'storage_file_sizes': {'selection_reports': file_groups(SELECTION),
            'selection_DATA': file_groups(ROOT/'outputs/closeout_001/repair_001/c1_selection_001'),
            'prescreen_reports': file_groups(PRESCREEN), 'registration_bytes': (SELECTION/'SELECTION_REGISTRATION.json').stat().st_size},
        'call_budget': {'true_ledger': ref(calls_path), 'rows': len(calls), 'charged': len(charged),
            'human16000_remaining': 16000-len(charged), 'legacy12000_enforced_remaining': 12000-len(charged),
            'semantic_development_phases': list(development_phases), 'semantic_development_used': len(semantic_dev),
            'semantic_development3400_remaining': 3400-len(semantic_dev), 'selection_phase_used': len(phase_calls),
            'selection1052_remaining': 1052-len(phase_calls), 'raw_phase_counts': dict(Counter(call['phase'] for call in charged)),
            'remaining1036_if_all_new_global_charged': len(charged)+1036,
            'remaining1036_if_all_new_development_used': len(semantic_dev)+1036},
        'next_input': {'case_id': NEXT_RAW, 'planned_cells': len(next_cells), 'registered_reused': len(planned_next_old),
            'registered_new': len(next_cells)-len(planned_next_old),
            'same_method_raw_prior_true_ledger_new_attempts': [call for call in charged if call.get('case_id') == NEXT_RAW and call.get('method') in {cell['method'] for cell in next_cells}],
            'registered_reused_cells': planned_next_old},
        'evidence_limits': [
            'Internal JSON time is record-construction time, not completion; mtime and SHA-sidecar mtime bracket later writes but are wall-clock filesystem observations.',
            'Numerical engine wall_s is a perf_counter interval of its function; it includes host/device work and waits and is not CUDA-kernel profiling.',
            'Outside-engine gaps bundle source/input hashing, parsing, budget walks, reserve/fsync/compression, forward byte verification, external scoring and host scheduling; these traces do not uniquely allocate their cost.',
            'The three current ledger snapshots are current CPU metadata timings, not historical runtime profiles; overlapping tree roots are counted once by resolved path.',
            'Prescreen and blank shard use different raws and different controller metadata workloads; timing ratios cannot identify a numerical or I/O cause.'],
        'old_files_unchanged': before == after, 'protected_old_file_count': len(before),
        'protected_old_sha256_before': before, 'protected_old_sha256_after': after,
        'analysis_script': ref(Path(__file__))}
    output = DEST/'READONLY_RUNTIME_ANALYSIS.json'
    with output.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps({'report': ref(output), 'timeline': timeline, 'comparison': {key: comparison[key] for key in (
        'prescreen_launch_to_terminal_record_s', 'prescreen_sum_engine_wall_s', 'prescreen_launch_to_terminal_minus_engine_s')},
        'gap_stats': gaps, 'call_budget': {key: value for key, value in result['call_budget'].items() if key != 'true_ledger'},
        'next_input_counts': {key: result['next_input'][key] for key in ('case_id', 'planned_cells', 'registered_reused', 'registered_new')},
        'tree_ledger_scan_wall_s': result['tree_ledger_scan_wall_s'], 'old_files_unchanged': before == after}, indent=2), flush=True)


if __name__ == '__main__': main()
