"""Diagnostic precision confirmation; no estimator, gate or source mutation.

Repeat the mathematical inner products using the exact original float32 probe,
directions, calibration buffers and W, converted to float64. Original failed
checks stay failed. This script cannot change a canonical decision or train.
"""
from pathlib import Path
import copy
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research_v5 import common
common.configure_runtime()
import torch
from research_v5.identifiability import _mode_bank, _directions


@torch.no_grad()
def main():
    started = time.perf_counter()
    protocol = common.require_registered()
    folder = common.REPORTS/'numerical_precision'
    registration = common.read_json(folder/'registration_002.json')
    assert registration['script_sha256'] == common.sha(Path(__file__))
    source = common.read_json(common.REPORTS/'FISHER_IDENTIFIABILITY.json')
    assert common.sha(common.REPORTS/'FISHER_IDENTIFIABILITY.json') == registration['original_fisher_sha256']
    rows = source['rows']
    assert [row['case_id'] for row in rows] == registration['case_ids']
    p32 = common.make_physics('cuda', torch.float32, config=protocol['physics_config'])
    p64 = copy.deepcopy(p32)._apply(lambda value: value.to(
        dtype=torch.complex128 if value.is_complex() else torch.float64)
        if value.is_complex() or value.is_floating_point() else value)
    modes, descriptions, omitted = _mode_bank(p32, protocol['fisher_options'])
    outcomes = []
    for index, row in enumerate(rows):
        assert row['status'] == 'RETURNED'
        data_folder = common.DATA/'oracle_data'
        raw_path, truth_path = data_folder/row['raw_path'], data_folder/row['label_path']
        assert raw_path.resolve().is_relative_to(data_folder.resolve())
        assert truth_path.resolve().is_relative_to(data_folder.resolve())
        assert common.sha(raw_path) == row['raw_file_sha256']
        assert common.sha(truth_path) == row['label_sha256']
        raw = common.arrays(raw_path, raw_only=True)['raw_dn']
        truth = common.arrays(truth_path)
        z32 = torch.as_tensor(truth['latent_state'],device='cuda',dtype=torch.float32)[None]
        b32 = torch.as_tensor(truth['beta'],device='cuda',dtype=torch.float32)[None]
        y32 = common.raw_electrons(p32, raw)
        w32 = (y32.clamp_min(0)+p32.extra_variance).reciprocal()
        generator = torch.Generator(device='cuda').manual_seed(protocol['fisher_options']['seed'])
        residual32 = torch.randn(y32.shape,generator=generator,dtype=torch.float32,device='cuda')
        directions = _directions(z32,b32,modes,p32)
        original = row['fisher']['frames'][0]
        assert len(directions)==len(original['adjoint_checks'])
        z64,b64 = z32.double(),b32.double()
        checks = []
        for ordinal,(group,mode,dz32,db32) in enumerate(directions):
            previous = original['adjoint_checks'][ordinal]
            assert (group,mode)==(previous['group'],previous['mode_index'])
            dz64,db64 = dz32.double(),db32.double()
            j64 = p64.jvp(z64,b64,dz64,db64)
            result = {'group':group,'mode_index':mode}
            for kind,r32 in (('unweighted',residual32),('weighted',w32.sqrt()*residual32)):
                r64 = r32.double()
                gz64,gb64 = p64.vjp(z64,b64,r64)
                left = (j64*r64).sum()
                right = (gz64*dz64).sum()+(gb64*db64).sum()
                absolute = (left-right).abs()
                relative = float(absolute/(left.abs()+right.abs()).clamp_min(torch.finfo(torch.float64).tiny))
                scale = (torch.linalg.vector_norm(j64)*torch.linalg.vector_norm(r64)
                         +torch.linalg.vector_norm(gz64)*torch.linalg.vector_norm(dz64)
                         +torch.linalg.vector_norm(gb64)*torch.linalg.vector_norm(db64))
                result[kind] = {'lhs_float64':float(left),'rhs_float64':float(right),
                    'relative_error_float64':relative,'absolute_error_float64':float(absolute),
                    'norm_scaled_error_float64':float(absolute/scale.clamp_min(1e-300)),
                    'tolerance_float64':registration['float64_tolerance'],
                    'passed_float64':relative<=registration['float64_tolerance'],
                    'original_float32':previous[kind]}
            checks.append(result)
        passed = all(check[kind]['passed_float64'] for check in checks for kind in ('unweighted','weighted'))
        outcome = {'case_id':row['case_id'],'mother_id':row['mother_id'],
            'raw_file_sha256':row['raw_file_sha256'],'label_sha256':row['label_sha256'],
            'original_adjoint_all_passed':original['adjoint_all_passed'],
            'original_float64_FD_all_passed':original['float64_finite_difference']['all_checked_passed'],
            'same_probe_and_directions_and_weights':True,'float64_adjoint_all_passed':passed,
            'checks':checks,'diagnostic_truth_only':True,'no_reconstruction_called':True}
        common.append_jsonl(folder/'same_probe_checks_002.jsonl',outcome)
        outcomes.append(outcome)
        print(f'float64 same-probe case {index+1}/24: passed={passed}',flush=True)
    summary = {'schema':'v5-same-probe-float64-adjoint-confirmation-v1','time':common.now(),
        'cases':len(outcomes),'checks':sum(2*len(row['checks']) for row in outcomes),
        'all_passed':len(outcomes)==24 and all(row['float64_adjoint_all_passed'] for row in outcomes),
        'maximum_relative_error':max(check[kind]['relative_error_float64'] for row in outcomes
            for check in row['checks'] for kind in ('unweighted','weighted')),
        'original_float32_failed_checks':sum(not check[kind]['original_float32']['passed'] for row in outcomes
            for check in row['checks'] for kind in ('unweighted','weighted')),
        'original_float32_failures_preserved':True,'original_FD_all_passed':all(row['original_float64_FD_all_passed'] for row in outcomes),
        'same_float32_probes_directions_and_sqrtW_converted_exactly':True,
        'original_calibration_buffers_converted_without_rebuilding':True,
        'original_true_states_unchanged':True,'reconstruction_metrics_repeated':False,
        'calibration_model_regularization_thresholds_changed':False,
        'frozen_source_and_original_reports_modified':False,'training_or_final_authorized':False,
        'registration_sha256':common.sha(folder/'registration_002.json'),
        'checks_sha256':common.sha(folder/'same_probe_checks_002.jsonl'),
        'script_sha256':common.sha(Path(__file__)),'wall_s':time.perf_counter()-started}
    common.write_json(folder/'summary_002.json',summary)
    print({'all_passed':summary['all_passed'],'checks':summary['checks'],'max_relative_error':summary['maximum_relative_error'],
           'wall_s':summary['wall_s']},flush=True)
    if not summary['all_passed']: raise SystemExit(1)


if __name__=='__main__': main()
