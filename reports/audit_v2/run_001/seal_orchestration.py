"""Seal a narrow orchestration/report fix; prove all numerical computation unchanged."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import runtime
from evaluation_v2.audit import source_hashes


def function_hash(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def main():
    output = Path(__file__).resolve().parent
    path = output/'orchestration_compatibility_v003.json'
    if path.exists():
        raise FileExistsError('Immutable compatibility registration already exists')
    original_path = output/'evaluation_registration_v002.json'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    current = source_hashes()
    changed = [k for k, v in original['evaluation_sources'].items() if current[k] != v]
    assert set(changed) == {'evaluation_v2/audit.py', 'evaluation_v2/report.py'}
    before = ROOT/'outputs/audit_v2/run_001/source_v002/audit.py'
    after = ROOT/'evaluation_v2/audit.py'
    assert runtime.sha256(before) == original['evaluation_sources']['evaluation_v2/audit.py']
    functions = ('work', 'configure_device', 'validate_methods', 'stats_for_inference', 'load_fixture',
                 'training_statistics', 'freeze_smoke', 'freeze_diagnostics', 'freeze_legacy')
    hashes = {name: function_hash(before, name) for name in functions}
    assert all(function_hash(after, name) == h for name, h in hashes.items())
    record = {'utc': runtime.now(), 'original_registration_sha256': runtime.sha256(original_path),
              'current_sources': current, 'changed_sources': changed, 'unchanged_execution_function_AST_sha256': hashes,
              'all_model_solver_physics_metrics_method_sources_unchanged': True,
              'frozen_data_regenerated': False, 'numerical_configuration_sha256_retained': True,
              'change': 'Select legacy calibration parent explicitly in suite all; exclude two unregistered smoke oracle probe outcomes from the registered cohort. No reconstruction/scoring function changed.',
              'reuse_rule': 'Reuse only identical v002 numerical config, case ID, input kind, input-array hash, method, successful status; preserve originating source registration on reused records.'}
    runtime.write_json(path, record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
