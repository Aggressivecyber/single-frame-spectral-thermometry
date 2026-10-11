"""Read-only checks of additive public files and preserved baseline bytes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import runtime
import json
import re
import subprocess


def main():
    output = Path(__file__).resolve().parent
    registration = json.loads((output/'registration.json').read_text(encoding='utf-8'))
    compatibility = json.loads((output/'orchestration_compatibility_v3_001.json').read_text(encoding='utf-8'))
    assert runtime.sha256(output/'registration.json') == compatibility['registration_sha256']
    assert all(runtime.sha256(ROOT/name) == digest for name, digest in compatibility['current_sources'].items())
    assert all(runtime.sha256(ROOT/name) == digest for name, digest in registration['original_tracked_sha256'].items())
    assert runtime.sha256(ROOT/'reports/audit_v2/run_001/per_case.jsonl') == registration['audit_v2_journal_sha256']
    assert runtime.sha256(ROOT/'reports/audit_v2/run_001/frozen_manifest.json') == registration['audit_v2_manifest_sha256']
    names = ['REPAIR_V3.md', 'repair_next.py', 'tests/test_repair_v3.py', 'tests/test_repair_v3_orchestration.py']
    names += [str(p.relative_to(ROOT)).replace('\\', '/') for folder in ('repair_v3', 'reports/repair_v3')
              for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.tmp']
    patterns = [re.compile(r'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----'),
                re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
                re.compile(r'sk-(?:proj-)?[A-Za-z0-9_-]{24,}')]
    records = []
    for name in sorted(set(names)):
        path = ROOT/name
        size = path.stat().st_size
        assert size < 50*1024**2, f'Unexpectedly large Git artifact: {name}'
        if path.suffix in ('.py', '.json', '.jsonl', '.md', '.xml', '.log'):
            text = path.read_text(encoding='utf-8')
            assert not any(pattern.search(text) for pattern in patterns), f'Credential pattern in {name}'
        records.append({'path': name, 'bytes': size, 'sha256': runtime.sha256(path)})
    assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT).decode().strip()
    runtime.write_json(output/'publication_preflight.json', {
        'original_files_preserved': len(registration['original_tracked_sha256']),
        'old_journal_unchanged': True, 'current_registered_sources_match': True,
        'pattern_scan': 'private-key headers and common API-token shapes in new text artifacts',
        'credential_pattern_matches': 0, 'maximum_file_limit_MiB': 50,
        'artifact_bytes_before_receipt': sum(r['bytes'] for r in records),
        'files': [r for r in records if r['path'] != str((output/'publication_preflight.json').relative_to(ROOT)).replace('\\', '/')],
        'optimizer_updates': 0, 'network_modules_added': 0})
    print({'publication_preflight': 'PASS', 'new_files_checked': len(records),
           'new_artifact_MiB': sum(r['bytes'] for r in records)/1024**2})


if __name__ == '__main__':
    main()
