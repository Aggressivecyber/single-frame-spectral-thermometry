"""One-time, explicit seal for a reuse-cohort correction after physical gates."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import runtime
import ast
from repair_v3.run import sources, read, original_preservation


def main():
    output = Path(__file__).resolve().parent
    destination = output/'orchestration_compatibility_v3_001.json'
    if destination.exists():
        raise FileExistsError('Compatibility seal is immutable')
    registration = read(output/'registration.json')
    before_path = output/'source_initial/run.py'
    current_path = ROOT/'repair_v3/run.py'
    assert runtime.sha256(before_path) == registration['sources']['repair_v3/run.py']
    before = ast.parse(before_path.read_text(encoding='utf-8'))
    after = ast.parse(current_path.read_text(encoding='utf-8'))
    old = {n.name: ast.dump(n, include_attributes=False) for n in before.body if isinstance(n, ast.FunctionDef)}
    new = {n.name: ast.dump(n, include_attributes=False) for n in after.body if isinstance(n, ast.FunctionDef)}
    changed = sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))
    assert changed == ['guard', 'old_rows']
    numeric_functions = sorted(set(old)-set(changed))
    before.body = [n for n in before.body if not isinstance(n, ast.FunctionDef) or n.name not in changed]
    after.body = [n for n in after.body if not isinstance(n, ast.FunctionDef) or n.name not in changed]
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    current = sources()
    assert [p for p in current if current[p] != registration['sources'].get(p)] == ['repair_v3/run.py']
    assert original_preservation(registration)
    runtime.write_json(destination, {
        'schema': 'physical-repair-v3-cohort-compatibility-001', 'utc': runtime.now(),
        'reason': 'The old journal contains two extra unregistered known-calibration resume probes; reuse only the exact eight audit_v2 registered methods.',
        'registration_sha256': runtime.sha256(output/'registration.json'),
        'gate_receipt_sha256_at_correction': runtime.sha256(output/'gate_receipt.json'),
        'original_source_snapshot': 'source_initial/run.py',
        'original_source_sha256': runtime.sha256(before_path),
        'current_sources': current,
        'changed_function_names': changed,
        'unchanged_function_names_ast_verified': numeric_functions,
        'non_function_module_ast_unchanged': True,
        'additional_regression_test_sha256': runtime.sha256(ROOT/'tests/test_repair_v3_orchestration.py'),
        'additional_tests_xml_sha256': runtime.sha256(output/'cohort_tests.xml'),
        'original_registration_rewritten': False, 'physical_inference_changed': False,
        'gate_input_or_result_recomputed': False, 'original_audit_journal_modified': False})
    print({'sealed': True, 'changed_functions': changed, 'numeric_functions_unchanged': len(numeric_functions)})


if __name__ == '__main__':
    main()
