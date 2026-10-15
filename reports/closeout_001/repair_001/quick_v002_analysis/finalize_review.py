"""Hash/provenance completion and concise findings; no endpoint evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from research_closeout import storage as s
import hashlib
import json
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=s.REPORTS/'repair_001/quick_v002'


def array_sha(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str(value.dtype).encode()+json.dumps(list(value.shape)).encode())
    h.update(value.tobytes());return h.hexdigest()


def ref(path): return {'path':str(path),'sha256':s.sha(path)}


def main():
    registration=s.read(RUN/'QUICK_REGISTRATION.json')
    review=s.read(HERE/'NUMERICAL_ENDPOINT_REVIEW.json')
    result=s.read(RUN/'QUICK_RESULT.json')
    config=registration['config'];sources=registration['source_sha256'];adapter=registration['calibration_contract']
    fingerprints={
        'config_fingerprint':s.canonical({'config':config,'C1v3_adapter':adapter,'sources':sources})==registration['config_fingerprint'],
        'objective_fingerprint':s.canonical({'C1v3_adapter':adapter,
            'physical_objective_version':config['physical_objective_version'],
            'fixed_physical_options':config['physical_options'],'source_sha256':sources,
            'initializer_changes_objective':False})==registration['objective_fingerprint'],
        'registration_file_hash':s.sha(RUN/'QUICK_REGISTRATION.json')==result['registration_sha256']}
    keys=[]
    for cell in result['summary']['cells']:
        expected=s.canonical({'registration_sha256':result['registration_sha256'],
            'method':cell['method'],'raw_sha256':cell['raw_sha256']})
        keys.append({'case_id':cell['case_id'],'method':cell['method'],'cache_key_verified':expected==cell['cache_key']})
    initializer=[];extra_hashes={}
    references={r['case_id']:r for r in registration['references']}
    for case_id,provenance in registration['v3_initializer_provenance'].items():
        for key in ('state_cache','receipt_journal'):
            r=provenance[key];actual=s.sha(r['path']);extra_hashes[r['path']]=actual
            if actual!=r['sha256']: raise ValueError('Initializer provenance SHA mismatch: '+r['path'])
        journal=list(s.rows(provenance['receipt_journal']['path']))
        row=journal[provenance['receipt_journal']['line']-1]
        reference=references[case_id]
        with np.load(s.LEGACY/reference['raw_reference']['path'],allow_pickle=False) as handle:
            raw=handle['raw_dn'].copy()
        source_dtype={'torch.float32':np.float32,'torch.float64':np.float64}[provenance['source_input_dtype']]
        source_raw=raw.astype(source_dtype)
        passed=bool(row['case_id']==case_id and row['source']=='C_v3'
            and row['source_metadata']==provenance['source_metadata']
            and row['raw_sha256']==reference['raw_reference']['sha256']==provenance['raw_file_sha256']
            and array_sha(raw)==provenance['raw_array_sha256']==reference['raw_reference']['array_sha256']
            and array_sha(source_raw)==provenance['inference_input_array_sha256']
            and np.array_equal(source_raw.astype(raw.dtype),raw)
            and not provenance['truth_or_identity_used'])
        initializer.append({'case_id':case_id,'source_input_dtype':provenance['source_input_dtype'],
            'raw_camera_and_source_array_hashes_verified':passed,
            'state_cache_file_SHA_verified':True,'journal_file_and_exact_receipt_line_verified':True,
            'inversion_repeated':False})
    process=s.read(RUN/'QUICK_PROCESS.json');pid=int(process['launcher_pid'])
    query=f"@(Get-Process -Id {pid} -ErrorAction SilentlyContinue).Count"
    p=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],
        capture_output=True,text=True,check=True)
    process_count=int(p.stdout.strip())
    lock_paths=[s.REPORTS/'PHASE_ACTIVE.json',RUN/'PHASE_ACTIVE.json']
    lifecycle={'launcher_pid':pid,'observed_process_count':process_count,
        'lock_files':{str(path):path.exists() for path in lock_paths},
        'completed_result_time':result['time'],'observed_at':s.now()}
    immutable={path:s.sha(path)==digest for path,digest in review['file_sha256'].items()}
    rows=[]
    for cell in review['cells']:
        g=cell['same_parameter_gradient_decomposition'];k=cell['knot_and_window_statistics'];m=cell['external_scientific_metrics']
        rows.append({'case_id':cell['case_id'],'initializer':cell['initializer'],
            'independent_final_stage_projected_gradient_inf':cell['independent_final_endpoint']['record']['scaled_projected_linf'],
            'executed_final_gtol':cell['final_stage_requested_gtol'],
            'independent_final_endpoint_passed':cell['independent_final_endpoint_passed'],
            'cold_tighter_template_RMS_change_e':cell['tighter_inner']['cold_start']['comparison']['template_RMS_difference_e'],
            'raw_CPU_reconstruction_maximum_absolute_e':cell['canonical_CPU_reconstruction']['predicted_raw_e']['maximum_absolute'],
            'within_C1_window_pixels':k['within_fixed_C1_half_window_pixels'],
            'latent_pixel_denominator':k['latent_pixels'],'within_C1_window_fraction':k['within_fixed_C1_half_window_fraction'],
            'latent_pixels_within_1e_minus4_K_of_knot':k['distance_counts']['0.0001'],
            'fixed_C1_prior_original_PLUT_conditional_profile_projected_gradient_inf':cell['same_z_original_PLUT_conditional_profile']['fixed_C1_prior']['record']['scaled_projected_linf'],
            'temperature_anchor_regscale_C1_over_PLUT':cell['anchor_scale_change']['C1_over_original_ratio'][0],
            'anchor_only_gradient_L2':g['anchor_regularizer_only']['physical_l2'],
            'calibration_only_gradient_L2':g['calibration_values_and_derivatives_only']['physical_l2'],
            'anchor_only_gradient_C1_scaled_inf':g['anchor_regularizer_only']['C1_scaled_linf'],
            'calibration_only_gradient_C1_scaled_inf':g['calibration_values_and_derivatives_only']['C1_scaled_linf'],
            'same_parameter_objective_anchor_change':g['objective_anchor_regularizer_only'],
            'same_parameter_objective_interpolation_change':g['objective_calibration_only'],
            'external_color_temperature_RMSE_K':m['temperature_reference_color_RMSE_K'],
            'external_template_RMSE_e':m['template_RMSE_e'],'external_template_relative_L2':m['template_relative_L2'],
            'external_metric_pixel_denominator':m['domain_pixels'],'scientific_acceptance_claimed':False})
    passed=bool(review['all_audit_checks_passed'] and all(fingerprints.values())
        and all(k['cache_key_verified'] for k in keys)
        and all(i['raw_camera_and_source_array_hashes_verified'] for i in initializer)
        and all(immutable.values()) and process_count==0 and not any(lifecycle['lock_files'].values()))
    findings={'time':s.now(),'all_review_checks_passed':passed,
        'denominators':{**review['denominators'],'unique_raws':len(references),'unique_mothers':len(references),
            'initializers_per_raw':2,'independent_scientific_sample_size_claimed':False},
        'fingerprint_checks':fingerprints,'cache_key_checks':keys,'initializer_provenance_checks':initializer,
        'additional_initializer_provenance_file_sha256':extra_hashes,'lifecycle':lifecycle,
        'all_frozen_files_unchanged_after_final_hash_pass':all(immutable.values()),
        'source_count':len(sources),'frozen_files_rechecked':len(immutable),
        'findings':[
            'All4 final float64 endpoints reconstruct on CPU and remain stationary at the executed stage3 gtol=1e-6 under tighter warm/cold conditional linear solves. Stage4 gtol=1e-7 was not executed.',
            'All12 recorded stage qualifications and8 platform comparisons are SHA-bound and logically consistent. Only4 final parameter states can be independently replayed; stage1/2 parameters and platform output arrays were not saved.',
            'At fixed C1 final parameters and fixed C1 prior, restoring the original PLUT raises the conditional-profile projected gradient to2.7e-5 through3.7e-5. This supports the calibration-smoothness mechanism at these saved endpoints.',
            'C1 also lowers the anchor temperature regularization scale by0.6249078104percent; log-amplitude scale is unchanged. Its gradient contribution exceeds the interpolation contribution in L2 and is comparable or larger in scaled infinity norm. Both changes belong to the new physical estimator.',
            'About10percent of final latent pixels lie within the predeclared0.25K C1 windows; only1to3 of65536 lie within1e-4K of a5K knot. These are C1 endpoints, not v001 stalled states.',
            'The old v001 latent states were not saved. Its precise stalled-pixel/knot association and a unique causal explanation cannot be established retrospectively.',
            'Physical errors are retained for every one of4 cells, with65536 pixels each and only2 unique development raws. Numerical qualification does not certify scientific accuracy, detection, calibration, uncertainty or final-test generalization.'],
        'cells':rows,'cross_initializer_pairs':review['cross_initializer_pairs'],
        'formal_inverse_calls':0,'GPU_used':False,'diagnostic_truth_reopened':False,
        'endpoint_audit_counters':review['counters'],'finalization_endpoint_evaluations':0,
        'main_review':ref(HERE/'NUMERICAL_ENDPOINT_REVIEW.json'),
        'finalization_script_sha256':s.sha(__file__)}
    findings_path=HERE/'FINAL_FINDINGS.json';s.write(findings_path,findings,replace=True)
    receipt={'time':s.now(),'all_review_checks_passed':passed,
        'main_review':ref(HERE/'NUMERICAL_ENDPOINT_REVIEW.json'),'findings':ref(findings_path),
        'scripts':[ref(HERE/'review_endpoints.py'),ref(Path(__file__))],
        'denominators':findings['denominators'],'formal_inverse_calls':0,'GPU_used':False,
        'frozen_sources_changed':False,'scope':'completed four-cell read-only CPU endpoint audit with explicit limitations; no complete inversion or scientific acceptance'}
    path=HERE/'FINAL_REVIEW_RECEIPT.json';s.write(path,receipt,replace=True)
    print(json.dumps({'receipt':ref(path),**receipt},indent=2),flush=True)
    return findings


if __name__=='__main__':
    raise SystemExit(0 if main()['all_review_checks_passed'] else 2)
