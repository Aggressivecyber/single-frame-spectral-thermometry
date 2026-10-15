"""Root byte/denominator audit. This performs no optimization or raw generation."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research_closeout.runtime_entry import install_platform_fallback
install_platform_fallback()
from research_closeout import storage as s
from research_closeout import c1_selection as c
from research_closeout import c1_small_panel as panel


def main():
    store = c.SelectionStore()
    reg, digest = c.verify_registration(store)
    checked_sources = {}
    for name, expected in reg['source_sha256'].items():
        data = (ROOT/name).read_bytes()
        ast.parse(data, filename=name)
        index = subprocess.run(['git', 'show', ':'+name], cwd=ROOT,
                               check=True, capture_output=True).stdout
        if hashlib.sha256(data).hexdigest() != expected or index != data:
            raise RuntimeError('Source/AST/index bytes differ: '+name)
        checked_sources[name] = expected
    contract = s.read(reg['contracts'][0]['path'])
    if contract['all_passed'] is not True or len(contract['checks']) != 45:
        raise RuntimeError('Expected final45 CPU contracts')
    protected = contract['old_protected_file_sha256_before']
    if protected != contract['old_protected_file_sha256_after']:
        raise RuntimeError('CPU review changed protected evidence')
    for path, expected in protected.items():
        if s.sha(path) != expected:
            raise RuntimeError('Protected object changed since CPU review: '+path)
    cells = reg['planned_cells']
    blank = 'v5_development_selection_mother_0000_blank_noise0'
    shard = [row for row in cells if row['case_id'] == blank]
    calls, prefix = c.dev.call_snapshot()
    charged = sum(not row.get('reused') for row in calls)
    groups = {key: sum(row['execution_role'] == key for row in cells)
              for key in ('REUSE_PRIOR_C1V3', 'NEW_INVERSE')}
    checks = {'full1080': len(cells) == 1080, 'reuse28': groups['REUSE_PRIOR_C1V3'] == 28,
              'new1052': groups['NEW_INVERSE'] == 1052,
              'blank18': len(shard) == 18,
              'blank_reuse2': sum(row['execution_role'] == 'REUSE_PRIOR_C1V3' for row in shard) == 2,
              'blank_new16': sum(row['execution_role'] == 'NEW_INVERSE' for row in shard) == 16,
              'same_charged5115': charged == 5115, 'no_active_global_lock': not (s.REPORTS/'PHASE_ACTIVE.json').exists(),
              'no_active_local_lock': not (store.report_root/'PHASE_ACTIVE.json').exists(),
              'approved16000_dependency_frozen': any(Path(ref['path']).name == 'INVERSE_BUDGET_EXTENSION_16000.json' for ref in reg['read_dependencies'])}
    if not all(checks.values()): raise RuntimeError(checks)
    value = {'schema': 'root-C1-selection-registration-review-v001', 'time': s.now(),
             'all_passed': True, 'checks': checks, 'registration_sha256': digest,
             'source_AST_worktree_index_byte_matches': checked_sources,
             'read_dependency_count': len(reg['read_dependencies']),
             'protected_prior_objects_unchanged': len(protected),
             'CPU_contracts': 45, 'planned_cells': len(cells), 'reuse_groups': groups,
             'first_shard_raw': blank, 'first_shard_cells': len(shard),
             'global_charged_inverse_calls': charged, 'inverse_ledger': prefix,
             'new_complete_inverse_calls': 0, 'raw_generation': 0, 'truth_loaded': False,
             'independent_sample_units': '12 planned parents/60raws; first shard one raw; routes are not independent samples',
             'first_shard96MiB_reserve_check': store.budget().check(96*2**20),
             'qualification_pending': True, 'scientific_selection_performed': False,
             'whole_study_complete': False}
    target = store.report_root/'ROOT_REGISTRATION_REVIEW.json'
    panel.seal(store, target, value)
    print(json.dumps({key: value[key] for key in ('all_passed', 'registration_sha256', 'CPU_contracts',
          'planned_cells', 'reuse_groups', 'first_shard_raw', 'global_charged_inverse_calls',
          'read_dependency_count', 'protected_prior_objects_unchanged')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
