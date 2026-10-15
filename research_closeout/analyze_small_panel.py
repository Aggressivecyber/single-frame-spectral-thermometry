"""Read-only native forward audit of the registered twelve-cell C1 panel."""
from . import storage as s
from . import c1_small_panel as panel
import argparse
import json
from pathlib import Path
import time
import numpy as np

REPORT = s.REPORTS/'repair_001/quick_v003'
MAIN_FIELDS = ('domain_pixels', 'accepted_pixels', 'nonfinite_output_pixels',
    'temperature_out_of_bounds_pixels', 'template_SSE', 'template_truth_energy',
    'template_error_sum', 'template_RMSE_e', 'template_bias_e',
    'temperature_SSE_K2', 'temperature_error_sum_K', 'temperature_RMSE_K',
    'temperature_bias_K', 'signed_projection', 'signed_projection_numerator',
    'support_pixels', 'sign_correct_pixels', 'sign_recall', 'raw_weighted_SSE',
    'raw_RMSE_e', 'temperature_reference_color_RMSE_K')


def _ledger():
    path=s.REPORTS/'inverse_calls.jsonl'
    rows=list(s.rows(path))
    return {'path':str(path), 'sha256':s.sha(path), 'rows':len(rows),
        'charged_new_calls':sum(not row.get('reused',False) for row in rows)}


def _same(a,b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return bool(np.isclose(a,b,atol=1e-8,rtol=1e-10,equal_nan=False))
    return a==b


def _pool(rows):
    result={}
    for label,predicate in (
        ('all',lambda r:True),
        ('normal',lambda r:'normal' in r['condition']),
        ('weak',lambda r:'weak' in r['condition']),
        ('blank',lambda r:r['condition']=='blank')):
        subset=[r for r in rows if predicate(r)]
        if not subset: continue
        pixels=sum(r['metrics']['domain_pixels'] for r in subset)
        result[label]={'raws':len(subset),'pixels':pixels,
            'temperature_RMSE_K':float(np.sqrt(sum(r['metrics']['temperature_SSE_K2'] for r in subset)/pixels)),
            'template_RMSE_e':float(np.sqrt(sum(r['metrics']['template_SSE'] for r in subset)/pixels))}
    return result


def run(device='cuda'):
    from .runtime_entry import install_platform_fallback
    install_platform_fallback()
    from .smooth_calibration import physics,contract
    from .compact_joint_cache import runtime_identity
    from . import metrics
    import torch
    destination=REPORT/'NATIVE_PANEL_ANALYSIS.json'
    if destination.exists():
        raise RuntimeError('Reuse the existing completed analysis; never silently overwrite it')
    store=panel.production_store()
    registration,digest=panel.verify_registration(store)
    terminal_path=store.report_root/panel.RESULT_NAME
    terminal=panel.read_sealed(store,terminal_path)
    assert terminal['status']=='C1V3_SMALL_PANEL_COMPLETE'
    assert terminal['registration_sha256']==digest
    assert terminal['summary']['planned_cells']==12
    assert terminal['summary']['returned_cell_rows']==12
    before=_ledger()
    assert before['charged_new_calls']==5099
    started=time.perf_counter()
    rows=[];pairs=[];pending={}
    with panel.phase_locks(store,digest,'C1-small-panel-root-forward-analysis'):
        torch.set_num_threads(2)
        p=physics(device)
        assert p.M.device.type=='cuda' and contract(p)==registration['calibration_contract']
        qualification=panel.read_sealed(store,store.report_root/panel.QUALIFICATION_NAME)
        panel.require_qualification(qualification,registration,digest,runtime_identity(p))
        s.frozen_bridge()
        from research_v6 import inputs
        references={r['case_id']:r for r in registration['references']}
        for cell in registration['planned_cells']:
            old=next(r for r in terminal['summary']['cells']
                if r['case_id']==cell['case_id'] and r['method']==cell['method'])
            raw=inputs.load_raw(references[cell['case_id']],p)
            cache=panel.cache_for(store,p,cell,registration,digest,
                qualification=cell['execution_role']=='REUSE_V002')
            # All parameter receipts already exist; this path neither recovers
            # missing files nor invokes resume/begin/optimization/scoring writes.
            for name in ('PREPARED.json','PARAMETERS.json','RESULT.json'):
                assert (cache.report/name).is_file()
            out,prepared,receipt,verified=cache._verify_forward(raw)
            assert verified['all_byte_identical']
            truth=inputs.load_diagnostic_truth(references[cell['case_id']],p)
            fresh=metrics.score(out,truth,raw,p,detector=None,method=cell['method'],
                application_domain='C1v3_small_panel_v003_development_regression')['metrics']
            matches={key:_same(fresh.get(key),old['metrics'].get(key)) for key in MAIN_FIELDS}
            assert all(matches.values()),(cell['case_id'],cell['method'],matches)
            assert fresh['domain_pixels']==65536 and fresh['accepted_pixels']==65536
            assert out['numerical_valid'] is True
            values={key:out[key].detach().cpu().numpy() for key in
                ('temperature_color_K','template_postblur_e_ref')}
            if cell['case_id'] in pending:
                first=pending.pop(cell['case_id'])
                pairs.append({'case_id':cell['case_id'],
                    'color_RMS_difference_K':float(np.sqrt(np.mean((values['temperature_color_K']-first['temperature_color_K'])**2))),
                    'template_RMS_difference_e':float(np.sqrt(np.mean((values['template_postblur_e_ref']-first['template_postblur_e_ref'])**2))),
                    'full_pixels':65536})
            else: pending[cell['case_id']]=values
            info=prepared['original_info']
            row={**cell,'metrics':fresh,'all_canonical_and_latent_byte_hashes_match':True,
                'scoring_matches':matches,'numeric_certified':bool(prepared['original_numerical_valid']),
                'original_info':info,'parameter_receipt':receipt,
                'compact_parameter_archive':str(cache.data/'PARAMETERS.npz'),
                'canonical_hashes':verified['reconstructed_canonical_arrays'],
                'continuum_latent_hash':verified['reconstructed_continuum_latent']}
            rows.append(row)
            print('audited '+cell['case_id']+' '+cell['initializer'],flush=True)
        assert len(rows)==12 and len(pairs)==6 and not pending
        after=_ledger()
        assert before==after
        baseline_rows=[]
        for name in ('development_baseline_reuse.jsonl','development_results.jsonl'):
            source=s.REPORTS/name
            for old in s.rows(source):
                if old.get('case_id') in references and old.get('method') in ('B1_lambda0.1','B1_lambda100','v3_physics','old_B4'):
                    assert old['raw_sha256']==references[old['case_id']]['raw_reference']['sha256']
                    assert old['metrics']['domain_pixels']==65536
                    baseline_rows.append({**old,'cached_metric_source':str(source),
                        'cached_metric_source_sha256':s.sha(source)})
        groups={}
        for method in sorted({r['method'] for r in baseline_rows}):
            selected=[r for r in baseline_rows if r['method']==method]
            assert len(selected)==6
            groups[method]={'source':'unchanged same-raw cached baseline metrics; no new inverse or baseline rescoring',
                'statistics':_pool(selected)}
        for initializer in ('v3','smooth_demod'):
            groups['C1v3__'+initializer]={'source':'root native byte-reconstruction and full-domain rescoring',
                'statistics':_pool([r for r in rows if r['initializer']==initializer])}
        value={'schema':'C1-small-panel-native-root-analysis-v001','time':s.now(),
            'terminal':panel.file_ref(terminal_path),'registration_sha256':digest,
            'analysis_source':panel.file_ref(Path(__file__)),
            'planned_cells':12,'reused_cells':4,'new_inverse_cells_in_completed_panel':8,
            'native_reconstructed_cells':len(rows),'recomputed_scoring_pixels':len(rows)*65536,
            'distinct_raw_inputs':6,'distinct_parent_mothers':len({r['mother_id'] for r in rows}),
            'all_finite_and_numerically_certified':all(r['numeric_certified'] for r in rows),
            'all_canonical_latent_hashes_and_scoring_match':True,
            'pairwise_initializers':pairs,'pooled_groups':groups,'rows':rows,
            'runtime':runtime_identity(p),'ledger_before':before,'ledger_after':after,
            'analysis_complete_inverse_calls':0,'analysis_optimizer_calls':0,'analysis_inner_solver_calls':0,
            'wall_s':time.perf_counter()-started,'scientific_method_selection':False,
            'pixel_counts_are_not_independent_scientific_units':True,
            'final_payloads_opened':False,'whole_closeout_complete':False}
        s.write(destination,s.plain(value))
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    args=parser.parse_args()
    result=run(args.device)
    print(json.dumps({key:result[key] for key in ('planned_cells','native_reconstructed_cells',
        'all_canonical_latent_hashes_and_scoring_match','analysis_complete_inverse_calls','wall_s')}),flush=True)


if __name__=='__main__': main()
