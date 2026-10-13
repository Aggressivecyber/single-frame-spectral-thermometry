"""Two predeclared existing endpoints, zero updates, full development audit."""
import time
import runtime
import torch
from training_v4.common import read, write, parameters_hash
from training_v4_1.common import setup, OUT as OLD, OLD_OUT, rel
from evaluation_v2.common import Journal
from training_v4_4.protocol import OUT, ARMS, register
from training_v4_4.evaluate import AuditedFixedContinuumCNN, evaluate


def run():
    setup()
    protocol = register()
    root = OUT/'zero_audit'
    assert not (root/'receipt.json').exists(), 'Do not overwrite completed comparisons'
    selection = read(OLD_OUT/'backbone_selection.json')
    rows = [r for r in read(OLD/'data_manifest.json')['rows'] if r['split']=='validation']
    started = time.monotonic()
    receipt = []
    write(root/'status.json', {'stage': 'RUNNING', 'time': runtime.now(), 'optimizer_updates': 0})
    try:
        for arm in ARMS:
            assert time.monotonic()-started < protocol['maximum_gpu_seconds_zero_audit']
            folder = root/arm
            endpoint = protocol['zero_update_endpoints'][arm]
            path = runtime.ROOT/endpoint['path']
            assert runtime.sha256(path)==endpoint['sha256']
            ck = torch.load(path, map_location='cpu', weights_only=False)
            model = AuditedFixedContinuumCNN(ck['physics_config'], ck['coefficient_calibration']).cuda().eval()
            model.load_cnn(ck['cnn_state_dict'])
            before = parameters_hash(model)
            assert before==ck['parameters_sha256']
            buffers = {k:v.clone() for k,v in model.named_buffers()}
            summary, screen = evaluate(model, rows, selection, folder, 0)
            current = Journal(folder/'validation_update_0000.jsonl').rows()
            reference_path = runtime.ROOT/f'reports/training_v4_3/run_001/{arm}/validation_update_0500.jsonl'
            reference = {r['case_id']:r for r in Journal(reference_path).rows()}
            differences = []
            for r in current:
                b = reference[r['case_id']]
                assert r['raw_hash']==b['raw_hash'] and r['label_sha256']==b['label_sha256']
                assert r['physical']['cache_key']==b['physical']['cache_key']
                differences.append(max(abs(r['metrics'][k]-b['metrics'][k])
                    for k in ('template_SSE','template_MSE_e2','template_RMSE_e','gradient_energy')))
            assert max(differences)==0., 'Only C may change; signed endpoint predictions must match'
            assert parameters_hash(model)==before
            assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
            receipt.append({'arm': arm, 'original_endpoint_updates': 500, 'new_optimizer_updates': 0,
                'source_checkpoint': endpoint['path'], 'checkpoint_sha256': runtime.sha256(path),
                'parameters_sha256': before, 'model_and_physics_unchanged': True,
                'paired_cases': len(current), 'maximum_signed_metric_difference': max(differences),
                'old_eligible': summary['eligible'], 'science_candidate': screen['science_candidate'],
                'reference_validation_sha256': runtime.sha256(reference_path)})
            write(root/'status.json', {'stage': 'RUNNING', 'time': runtime.now(), 'completed_arms': len(receipt),
                                     'optimizer_updates': 0})
        write(root/'receipt.json', {'arms': receipt, 'optimizer_updates': 0, 'elapsed_s': time.monotonic()-started,
            'source_sha256': runtime.sha256(__file__),
            'model_source_sha256': runtime.sha256(runtime.ROOT/'training_v4_4/model.py'),
            'evaluator_source_sha256': runtime.sha256(runtime.ROOT/'training_v4_4/evaluate.py'),
            'protocol_sha256': runtime.sha256(OUT/'protocol.json'), 'final_test_used': False})
        write(root/'status.json', {'stage': 'COMPLETE', 'time': runtime.now(), 'optimizer_updates': 0})
    except BaseException as exc:
        write(root/'failure_record.json', {'type': type(exc).__name__, 'message': str(exc),
                                         'completed_arms': receipt, 'deleted_records': 0})
        write(root/'status.json', {'stage': 'FAILED', 'time': runtime.now(), 'message': str(exc)})
        raise


if __name__=='__main__':
    run()
