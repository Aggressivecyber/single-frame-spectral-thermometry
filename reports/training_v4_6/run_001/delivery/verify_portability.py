"""Extract the finished ZIP outside the checkout and run one isolated raw CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[4]
DELIVERY = ROOT / 'outputs/training_v4_6/run_001/delivery'
BUILD = DELIVERY / 'inference_bundle_build_receipt.json'
CASE = 'v4_1_signed_validation_mother_0000_negative_normal_noise'
REFERENCE = ROOT / 'reports/training_v4_6/run_001/native_replay/execution/cases' / CASE / 'raw_branch'
RELOCATED = ROOT.parent / 'portable_checks/v46_20261013'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def run(python):
    started = time.monotonic()
    assert ROOT.drive.upper() in ('E:', 'D:')
    assert RELOCATED.resolve().drive.upper() in ('E:', 'D:')
    assert not RELOCATED.resolve().is_relative_to(ROOT.resolve())
    assert not RELOCATED.exists() and not (DELIVERY / 'portability_receipt.json').exists()
    build = read(BUILD)
    assert build['status'] == 'COMPLETE_BUNDLE_BYTES_AND_ZIP_VERIFIED'
    training_folder = ROOT / 'reports/training_v4_6/run_001/negative_noisy_repair'
    training_status = read(training_folder / 'status.json')
    training = read(training_folder / 'training_receipt.json')
    assert training_status['stage'] == 'COMPLETE' and training_status['optimizer_updates'] == 500
    assert training['updates'] == 500 and training['microbatches'] == 2000
    endpoint, = [row for row in training['checkpoints'] if row['update'] == 500]
    assert endpoint['path'] == 'training_v4_6/checkpoints/negative_noisy_repair/update_0500.pt'
    assert endpoint['Adam_steps'] == [500] and endpoint['sha256'] == build['checkpoint_sha256']
    assert sha(ROOT / endpoint['path']) == endpoint['sha256']
    replay_folder = ROOT / 'reports/training_v4_6/run_001/native_replay/execution'
    replay = read(replay_folder / 'receipt.json')
    watchdog = read(replay_folder / 'watchdog_receipt.json')
    assert replay['status'] == 'SUCCESS' and replay['passed'] and replay['case_count'] == 3
    assert all(row['passed'] and row['checkpoint_sha256'] == endpoint['sha256'] for row in replay['records'])
    assert watchdog['status'] == 'SUCCESS' and watchdog['passed']
    assert watchdog['worker_receipt_sha256'] == sha(replay_folder / 'receipt.json')
    reference = read(REFERENCE / 'inference_receipt.json')
    assert reference['status'] == 'SUCCESS' and reference['checkpoint_optimizer_updates'] == 500
    archive = Path(build['zip_path'])
    assert sha(archive) == build['zip_sha256']
    assert reference['checkpoint_sha256'] == build['checkpoint_sha256']
    RELOCATED.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'STARTING', 'case_id': CASE, 'started_at': datetime.now(timezone.utc).isoformat(),
               'source_sha256': sha(Path(__file__)), 'maximum_subprocess_seconds': 180,
               'build_receipt_sha256': sha(BUILD), 'zip_sha256': sha(archive),
               'training_receipt_sha256': sha(training_folder / 'training_receipt.json'),
               'raw_replay_receipt_sha256': sha(replay_folder / 'receipt.json'),
               'raw_replay_watchdog_receipt_sha256': sha(replay_folder / 'watchdog_receipt.json'),
               'relocation_root': str(RELOCATED), 'old_development_example_count': 1,
               'independent_scientific_confirmation': False, 'final_test_used': False}
    write(RELOCATED / 'execution_plan.json', receipt)
    try:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert len(z.namelist()) == len(set(z.namelist()))
            expected = {'inference_bundle/' + name for name in build['bundle_files_sha256_and_bytes']}
            assert set(z.namelist()) == expected
            for item in z.infolist():
                destination = (RELOCATED / item.filename).resolve()
                assert destination.is_relative_to(RELOCATED.resolve())
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('xb') as handle:
                    handle.write(z.read(item.filename))
        bundle = RELOCATED / 'inference_bundle'
        for name, item in build['bundle_files_sha256_and_bytes'].items():
            assert sha(bundle / name) == item['sha256']
            assert (bundle / name).stat().st_size == item['bytes']
        target = RELOCATED / 'negative_normal_result'
        runtime = RELOCATED / 'runtime'
        (runtime / 'tmp').mkdir(parents=True, exist_ok=False)
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        environment.update(THERMOMETRY_RUNTIME=str(runtime), PYTHONDONTWRITEBYTECODE='1',
                           CUDA_CACHE_PATH=str(runtime / 'cuda'), TEMP=str(runtime / 'tmp'),
                           TMP=str(runtime / 'tmp'), TMPDIR=str(runtime / 'tmp'))
        # -I removes caller PYTHONPATH, user site and script-directory imports.
        # Only the extracted package is explicitly added to the module path.
        child = '''import json, pathlib, runpy, sys
bundle = pathlib.Path(sys.argv[1]).resolve()
old_root = pathlib.Path(sys.argv[2]).resolve()
output = pathlib.Path(sys.argv[3]).resolve()
sys.path.insert(0, str(bundle))
sys.argv = ['training_v4_4.infer', '--raw', str(bundle/'examples/raw/negative_normal_noise.npz'),
    '--checkpoint', str(bundle/'checkpoints/update_0500.pt'), '--backbone',
    str(bundle/'reports/training_v4/run_001/backbone_selection.json'), '--device', 'cuda', '--output', str(output)]
runpy.run_module('training_v4_4.infer', run_name='__main__')
loaded = {}
for name, module in sorted(sys.modules.items()):
    location = getattr(module, '__file__', None)
    if not location:
        continue
    path = pathlib.Path(location).resolve()
    assert not path.is_relative_to(old_root), (name, str(path))
    if path.is_relative_to(bundle):
        loaded[name] = str(path)
assert loaded['runtime'] == str(bundle/'runtime.py')
with (output/'loaded_local_modules.json').open('x', encoding='utf-8') as handle:
    json.dump(loaded, handle, indent=2)
'''
        command = [str(Path(python).resolve()), '-I', '-B', '-c', child, str(bundle), str(ROOT), str(target)]
        with (RELOCATED / 'native_cli.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=str(bundle), env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=180, check=False,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, 'See relocated native_cli.log'
        actual = read(target / 'inference_receipt.json')
        assert actual['status'] == 'SUCCESS'
        for key in ('checkpoint_sha256', 'parameters_sha256', 'raw_hash', 'raw_file_sha256',
                    'backbone_configuration_sha256', 'source_hashes', 'prediction_array_hashes',
                    'physical_inverse_array_hashes'):
            assert actual[key] == reference[key], key
        assert not actual['labels_loaded'] and not actual['development_cache_loaded']
        assert actual['inverse_recomputed_from_raw'] and actual['checkpoint_optimizer_updates'] == 500
        # The comparison process reads saved output arrays only, after the CLI exits.
        import numpy as np
        checks = {}
        for filename in ('prediction.npz', 'physical_inverse.npz'):
            with np.load(target / filename, allow_pickle=False) as a, np.load(REFERENCE / filename, allow_pickle=False) as b:
                assert set(a.files) == set(b.files)
                checks[filename] = {}
                for key in a.files:
                    assert a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]), (filename, key)
                    checks[filename][key] = {'shape': list(a[key].shape), 'dtype': str(a[key].dtype), 'exact': True}
        for name, item in build['bundle_files_sha256_and_bytes'].items():
            assert sha(bundle / name) == item['sha256'], name
        receipt.update(status='PASS', exit_code=result.returncode, elapsed_seconds=time.monotonic()-started,
            all_prediction_arrays_exact=True, physical_state_and_beta_exact=True, array_checks=checks,
            bundle_payload_unchanged=True, original_checkout_module_imports=0,
            isolated_python=True, inherited_pythonpath_removed=True,
            loaded_local_modules=read(target / 'loaded_local_modules.json'),
            physical_inverse_seconds=actual['physical_inverse_seconds'], cnn_seconds=actual['cnn_seconds'],
            inference_receipt_path=str(target / 'inference_receipt.json'),
            inference_receipt_sha256=sha(target / 'inference_receipt.json'),
            reference_receipt_sha256=sha(REFERENCE / 'inference_receipt.json'),
            log_sha256=sha(RELOCATED / 'native_cli.log'))
        write(DELIVERY / 'portability_receipt.json', receipt)
        write(RELOCATED / 'portability_receipt.json', receipt)
        print(json.dumps({key: receipt[key] for key in ('status', 'all_prediction_arrays_exact',
            'physical_state_and_beta_exact', 'elapsed_seconds', 'original_checkout_module_imports')}))
        return receipt
    except BaseException as error:
        receipt.update(status='FAILED', error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=time.monotonic()-started, partial_artifacts_retained=True)
        write(RELOCATED / 'failure_receipt.json', receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args()
    assert args.execute, 'Explicitly execute only after training, raw replay and packaging complete'
    run(args.python)
