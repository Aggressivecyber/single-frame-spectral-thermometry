"""Restore only the frozen 16 development inputs into a fresh BASE checkout."""
import shutil
import subprocess
import runtime
from training_v4.common import OUT, setup, read, write
from training_v4.protocol import BASE


def prepare():
    setup()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime.ROOT).decode().strip()
    assert head == BASE, 'Reproduce in the detached baseline checkout made by reproduce.ps1'
    index = read(runtime.ROOT/'training_v4/development_inputs/index.json')
    for entry in index:
        source = runtime.ROOT/'training_v4/development_inputs'/entry['path'].split('/')[-1]
        assert runtime.sha256(source) == entry['file_sha256']
        destination = runtime.ROOT/'outputs/repair_v3/run_001'/entry['path']
        if destination.exists():
            assert runtime.sha256(destination) == entry['file_sha256']
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=runtime.ROOT).decode('utf-8').strip('\0').split('\0')
    write(OUT/'preserved_baseline.json', {'commit': BASE,
        'original_files': {name: runtime.sha256(runtime.ROOT/name) for name in tracked}})
    print({'frozen_development_inputs_restored': len(index), 'old_954_evaluations': 0})


if __name__ == '__main__':
    prepare()
