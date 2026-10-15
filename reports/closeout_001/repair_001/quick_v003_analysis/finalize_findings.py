"""Summarize the completed independent review without any physics evaluation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from research_closeout import storage as s
from datetime import datetime
import json

HERE=Path(__file__).resolve().parent
ATTEMPT=HERE/'attempt_002'
RUN=Path('D:/CodexWork/thermometry/closeout_001/reports/repair_001/quick_v003')


def ref(path): return {'path':str(Path(path).resolve()),'sha256':s.sha(path)}
def stamp(value): return datetime.fromisoformat(value)


def main():
    review=s.read(ATTEMPT/'NUMERICAL_REVIEW.json');metadata=s.read(ATTEMPT/'METADATA_REVIEW.json')
    result=s.read(RUN/'PANEL_RESULT.json');source=s.ROOT/'research_closeout/quick_v003_numerical_review.py'
    expected=review['source_sha256']['research_closeout/quick_v003_numerical_review.py']
    if s.sha(source)!=expected: raise ValueError('Reviewer source changed')
    unchanged={path:s.sha(path)==value for path,value in review['file_sha256'].items()}
    cells=[];starts=[];durations=[]
    originals={(c['case_id'],c['initializer']):c for c in result['summary']['cells'] if c['execution_role']=='NEW_INVERSE'}
    for cell in metadata['cells']:
        row=originals[(cell['case_id'],cell['initializer'])];root=Path(row['compact_report_root'])
        started=s.read(root/'STARTED.json');scored=s.read(root/'SCORE_panel_external_v001.json')
        starts.append(stamp(started['time']));duration=(stamp(scored['time'])-stamp(started['time'])).total_seconds();durations.append(duration)
        m=cell['metrics'];certificate=cell['certificate']
        cells.append({'case_id':cell['case_id'],'initializer':cell['initializer'],'review_passed':cell['passed'],
            'finite_output':row['finite_output'],'numerical_certified':row['production_numerical_certified'],
            'pixels':m['domain_pixels'],'accepted_pixels':m['accepted_pixels'],'raw_pixels':m['raw_pixels'],
            'executed_stages':[c['configuration']['gtol'] for c in certificate['stages']],
            'final_scaled_projected_gradient_inf':cell['final_gradient']['scaled_projected_linf'],
            'stages_all_qualified_by_log':all(c['passed'] for c in certificate['stages']),
            'output_platforms':certificate['platforms'],'earlier_stages_independently_replayed':False,
            'parameter_and_metadata_hashes_verified':True,
            'core_wall_s':cell['core_wall_s'],'ticket_to_compact_score_receipt_wall_s':duration,
            'template_RMSE_e':m['template_RMSE_e'],'color_temperature_RMSE_K':m['temperature_reference_color_RMSE_K'],
            'template_relative_L2':m['template_relative_L2'],'raw_weighted_RMSE':m['raw_weighted_RMSE'],
            'detector_status':m['frame_detection_v2']['status'],
            'reservation_release_warning':cell['reservation_release_warning']})
    cpu=[]
    for row in review['CPU_endpoint_checks']:
        refs=row['tighter_conditional_linear_references']
        cpu.append({'case_id':row['case_id'],'initializer':row['initializer'],'passed':row['passed'],
            'final_gtol':row['executed_final_gtol'],'actual_saved_endpoint_projected_gradient_inf':row['saved_endpoint']['gradient']['scaled_projected_linf'],
            'tighter_warm_projected_gradient_inf':refs[0]['fixed_endpoint']['gradient']['scaled_projected_linf'],
            'tighter_cold_projected_gradient_inf':refs[1]['fixed_endpoint']['gradient']['scaled_projected_linf'],
            'maximum_tighter_template_RMS_change_e':max(r['difference_from_saved_linear_solution']['template_RMS_e'] for r in refs),
            'maximum_tighter_normal_relative':max(r['actual_normal_relative_independently_recomputed'] for r in refs),
            'objective_difference_from_original_GPU_log':row['objective_difference_from_original_GPU_log'],
            'gradient_difference_from_original_GPU_log':row['gradient_difference_from_original_GPU_log']['scaled_projected_linf'],
            'original_shared_anchor_regscale_used_exactly':row['original_shared_anchor_regscale_used_exactly'],
            'latent_temperature_near_5K_knots':row['latent_temperature_near_5K_knots'],
            'wall_s':row['wall_s']})
    findings={'time':s.now(),'all_review_checks_passed':bool(review['all_review_checks_passed'] and all(unchanged.values())),
        'denominators':review['denominators'],'independent_readonly_review':ref(ATTEMPT/'NUMERICAL_REVIEW.json'),
        'metadata_review':ref(ATTEMPT/'METADATA_REVIEW.json'),'cells':cells,'CPU_recomputed_final_endpoints':cpu,
        'initializer_pairs':metadata['cross_initializer_pairs'],
        'actual_cost':{**metadata['recorded_compute'],
            'first_new_ticket_to_terminal_receipt_wall_s':(stamp(result['time'])-min(starts)).total_seconds(),
            'sum_eight_ticket_to_compact_score_receipt_wall_s':sum(durations),
            'time_basis':'wall clock from original SHA-bound receipts; includes cache/scoring when stated; no GPU-kernel-time or throughput claim'},
        'physical_metrics':metadata['physical_metrics_without_acceptance_claim'],
        'source_count':metadata['registered_sources'],'read_dependency_count':metadata['registered_read_dependencies'],
        'all_frozen_files_unchanged':all(unchanged.values()),'frozen_files_rechecked':len(unchanged),
        'counters':review['counters'],'GPU_used':False,'formal_inverse_calls':0,
        'early_stages_and_platforms_scope':'24 stage records and16 platform comparisons passed frozen-log checks; earlier parameter/output arrays are not saved and cannot be independently replayed',
        'first_review_attempt_preserved':'Top-level first attempt stopped on float64 CPU/GPU reduction roundoff before any endpoint/linear evaluation; attempt_002 verified roundoff and copied original anchor and coordinate scales exactly.',
        'scientific_interpretation':'Numerical stationarity and close initializers do not certify scientific accuracy, identifiability, detector calibration or generalization. No scientific acceptance is claimed.',
        'reservation_warning_interpretation':'All8 original compact results retain reservation-release warnings. This audit preserves them and does not repair storage or registered code.',
        'review_source':ref(source),'finalizer_source':ref(__file__)}
    s.write(HERE/'FINAL_FINDINGS.json',findings)
    receipt={'time':s.now(),'all_review_checks_passed':findings['all_review_checks_passed'],
        'findings':ref(HERE/'FINAL_FINDINGS.json'),'review':ref(ATTEMPT/'NUMERICAL_REVIEW.json'),
        'review_source':ref(source),'finalizer_source':ref(__file__),'denominators':findings['denominators'],
        'counters':review['counters'],'formal_inverse_calls':0,'GPU_used':False,'finalization_endpoint_evaluations':0,
        'all_frozen_files_unchanged':all(unchanged.values())}
    s.write(HERE/'FINAL_REVIEW_RECEIPT.json',receipt)
    print(json.dumps({'receipt':ref(HERE/'FINAL_REVIEW_RECEIPT.json'),'actual_cost':findings['actual_cost'],**receipt},indent=2))


if __name__=='__main__': main()
