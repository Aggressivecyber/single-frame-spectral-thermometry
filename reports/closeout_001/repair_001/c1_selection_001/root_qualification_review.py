"""Audit the separate four-endpoint native cache qualification, without a solve."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research_closeout.runtime_entry import install_platform_fallback
install_platform_fallback()
from research_closeout import storage as s
from research_closeout import c1_selection as c
from research_closeout import c1_small_panel as panel

store = c.SelectionStore()
reg, digest = c.verify_registration(store)
path = store.report_root/c.QUALIFICATION_NAME
value = panel.read_sealed(store, path)
panel.require_qualification(value, reg, digest, value['runtime'])
checked = {}


def walk(node):
    if isinstance(node, dict):
        if isinstance(node.get('path'), str) and isinstance(node.get('sha256'), str):
            ref = Path(node['path'])
            if not ref.is_absolute():
                raise RuntimeError('Qualification file references must be absolute')
            panel.verify_ref(node)
            checked[str(ref)] = node['sha256']
        for child in node.values(): walk(child)
    elif isinstance(node, list):
        for child in node: walk(child)


walk(value)
before = reg['prior_call_prefix']
if s.sha(before['path']) != before['prefix_sha256']:
    raise RuntimeError('Forward-only qualification changed inverse ledger')
if value['runtime']['device'] != 'cuda:0' or value['runtime']['dtype'] != 'torch.float64':
    raise RuntimeError('Qualification was not native float64')
for key in ('complete_inverse_calls', 'optimizer_calls', 'inner_solver_calls'):
    if value[key] != 0: raise RuntimeError('Qualification executed a solve')
if value['truth_loaded'] or (s.REPORTS/'PHASE_ACTIVE.json').exists() or (store.report_root/'PHASE_ACTIVE.json').exists():
    raise RuntimeError('Truth loaded or qualification process still owns locks')
release_errors = list((store.report_root/'qualification').rglob('RESERVATION_RELEASE_FAILURE*.json'))
if release_errors: raise RuntimeError('Unreleased namespace reservations')
result = {'schema': 'root-C1-selection-native-qualification-review-v001', 'time': s.now(),
          'all_passed': True, 'passed_existing_endpoints': 4,
          'qualification_sha256': s.sha(path), 'registration_sha256': digest,
          'checked_file_references': checked, 'source_files': len(reg['source_sha256']),
          'runtime': value['runtime'], 'new_namespace_canonical_and_latent_byte_identical_to_saved_origins': True,
          'origin_new_derived_and_latent_previous_self_persistence_only': True,
          'unsaved_v001_latent_not_compared': True,
          'global_inverse_ledger_byte_unchanged': before['prefix_sha256'],
          'complete_inverse_calls': 0, 'optimizer_calls': 0, 'inner_solver_calls': 0,
          'truth_loaded': False, 'both_locks_cleared': True, 'shard_not_started': True}
panel.seal(store, store.report_root/'ROOT_QUALIFICATION_REVIEW.json', result)
print(json.dumps({key: result[key] for key in ('all_passed', 'passed_existing_endpoints', 'qualification_sha256',
                 'registration_sha256', 'complete_inverse_calls', 'both_locks_cleared')}, indent=2), flush=True)
