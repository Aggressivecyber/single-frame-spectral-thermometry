"""Reopen actual checkpoints, test native identity, then export preserved bytes."""
import shutil
import sys
import torch
import runtime
from training_v4.common import OUT, DATA, setup, read, write, npz, parameters_hash
from training_v4.model import PhysicsPostInverseCNN, physical_output
from training_v4.cache import load_tensors
from infer import load_model


def verify():
    setup()
    manifest = read(OUT/'data_manifest.json')
    selection = read(OUT/'backbone_selection.json')
    row = next(r for r in manifest['rows'] if r['split'] == 'validation' and r['clean_identity'])
    inputs, cache = load_tensors(row, selection, labels=False)
    model = PhysicsPostInverseCNN(selection['physics_config']).cuda().eval()
    original_physics = load_model(device='cuda').physics
    buffers_match = {name: torch.equal(value, model.physics.state_dict()[name])
                     for name, value in original_physics.state_dict().items()}
    assert all(buffers_match.values()), 'Original calibrated physics buffers must match'
    records = []
    with torch.no_grad():
        raw_e = model.physics.raw_to_electrons(inputs[0])
        before = physical_output(model.physics, inputs[1], inputs[2], raw_e)
        after = model(*inputs)
        identity = {key: torch.equal(before[key], after[key]) for key in
                    ('state', 'beta', 'template_e', 'temperature_color_K', 'band_images_e', 'expected_e')}
        assert all(identity.values())
        for folder in sorted(OUT.glob('seed_*')):
            if not (folder/'training_receipt.json').exists():
                continue
            receipt = read(folder/'training_receipt.json')
            assert receipt['optimizer_updates'] == 2000
            selected = receipt['selected_checkpoint'] or receipt['final_checkpoint']
            payload = torch.load(selected, map_location='cpu', weights_only=False)
            final_payload = torch.load(receipt['final_checkpoint'], map_location='cpu', weights_only=False)
            selected_steps = [int(v['step']) for v in payload['optimizer_state']['state'].values()]
            final_steps = [int(v['step']) for v in final_payload['optimizer_state']['state'].values()]
            assert selected_steps and set(selected_steps) == {payload['optimizer_updates']}
            assert final_steps and set(final_steps) == {2000}
            model.load_cnn(payload['cnn_state_dict'])
            assert parameters_hash(model) == payload['parameters_sha256']
            predicted = model(*inputs)
            saved = npz(DATA/'validation_predictions'/folder.name/row['case_id']/'trained_CNN.npz')
            errors = {key: float((predicted[key][0]-torch.as_tensor(saved[key], device='cuda')).abs().max())
                      for key in ('template_e', 'temperature_color_K')}
            assert max(errors.values()) <= 1e-5
            export = runtime.ROOT/'training_v4/checkpoints'
            export.mkdir(exist_ok=True)
            finals = [('final_2000', receipt['final_checkpoint']), ('validation_selected', selected)]
            for name, source in finals:
                destination = export/f'seed_{receipt["seed"]}_{name}.pt'
                shutil.copy2(source, destination)
                assert runtime.sha256(destination) == runtime.sha256(source)
            records.append({'seed': receipt['seed'], 'complete_training_updates': 2000,
                'selected_checkpoint_update': payload['optimizer_updates'],
                'selected_passes_all_development_gates': receipt['selected_checkpoint'] is not None,
                'selected_parameters_hash_verified': True, 'native_repeat_inference_max_errors': errors,
                'selected_actual_Adam_steps': selected_steps, 'final_actual_Adam_steps': final_steps,
                'selected_export': f'training_v4/checkpoints/seed_{receipt["seed"]}_validation_selected.pt',
                'selected_sha256': runtime.sha256(selected),
                'final_export': f'training_v4/checkpoints/seed_{receipt["seed"]}_final_2000.pt',
                'final_sha256': receipt['final_checkpoint_sha256']})
    assert records
    write(OUT/'native_identity_and_checkpoint_receipt.json', {'native_shape': [256, 256],
        'case_id': row['case_id'], 'raw_hash': row['raw_hash'], 'cache_key': cache['cache_key'],
        'zero_head_bitwise_identity': identity, 'checkpoint_checks': records, 'truth_used_in_inference': False,
        'original_calibrated_physics_buffers_equal': buffers_match,
        'torch': str(torch.__version__), 'CUDA': torch.version.cuda,
        'GPU': torch.cuda.get_device_name(0), 'Python': sys.version, 'interpreter': sys.executable})
    print({'native_identity_passed': True, 'new_checkpoints_reopened_and_exported': len(records)*2})


if __name__ == '__main__':
    verify()
