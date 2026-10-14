"""One predeclared input/method plan, call ledger, and final-seal handshake."""
from . import storage as s
import itertools

SOURCE_MOTHERS=('v4_validation_mother_0000','v4_validation_mother_0013','v4_validation_mother_0026',
    'v4_validation_mother_0009','v4_validation_mother_0022','v4_validation_mother_0035')
FIRST_DEVELOPMENT_INDICES=(0,6,7,13,14,20,21,27,28,34,35,5)

def source_hashes():
    return {p.relative_to(s.ROOT).as_posix():s.sha(p) for p in sorted((s.ROOT/'research_closeout').rglob('*.py'))}

def plan():
    inventory=s.read(s.ROOT/'research_v6/input_inventory.json')
    old=s.read(s.ROOT/'research_v5/protocol.json')
    first={f'v5_development_selection_mother_{i:04d}' for i in FIRST_DEVELOPMENT_INDICES}
    all_dev=sorted({r['mother_id'] for r in inventory['rows'] if r['cohort']=='development'})
    assert first.issubset(all_dev) and len(all_dev)==36
    source=[r['case_id'] for r in inventory['rows'] if r['mother_id'] in SOURCE_MOTHERS and r['cohort']=='oracle']
    assert len(source)==90
    return {'schema':'thermometry-closeout-001','authored_from_user_text':True,'base_commit':s.BASE,
        'user_request_sha256':s.sha(s.REPORTS/'USER_EXECUTION_REQUEST.txt'),
        'external_execution_spec_and_check_script_attached':False,
        'source_evidence_root':str(s.LEGACY),'existing_repair_is_separate_evidence':True,
        'scientific_completion_requires':['raw-only reproducibility','quantified numerical contribution',
            'independent tests and complete denominators','Chinese paper and seven figure groups','explicit operating range'],
        'global_stop':'integrity/truth leakage/wrong units/unhandled nonfinite/corruption',
        'module_failure':'quarantine only the uncertified numerical module; continue qualified baselines',
        'candidate_failure':'retain negative result and continue with frozen v3',
        'source_cases':source,'source_mothers':list(SOURCE_MOTHERS),'source_order':['noise','quantization','continuous'],
        'source_error_terms':['continuum_leakage','shared_measurement_noise','regularization_bias','actual_numerical_error'],
        'source_roundoff_terms':['component_solver_corrections','normal_algebra_roundoff','stored_label_roundoff'],
        'source_objective':'unchanged original v5 alpha1 K/W/lambda; no postblur-prior substitution',
        'source_repair_rtol':1e-8,'source_residual_reference_rtol':1e-11,'source_residual_reference_maxiter':1000,
        'development':{'selection_mothers':sorted(first),'confirmation_mothers':[m for m in all_dev if m not in first],
            'candidates':[{'id':f'joint_c{c:g}_s{t:g}','continuum_multiplier':c,'template_multiplier':t}
                          for c,t in itertools.product((.1,1.,10.),repeat=2)],
            'max_candidates':9,'retain_after_first_stage':2,'freeze_after_remaining':1,
            'primary_inputs':'all five signed conditions, noise; remaining input forms reused or evaluated after selection',
            'initializers':['v3','smooth_demod'],'representation_grids':[32,64,128,256],
            'representation_color_budget_K':.2,'representation_raw_budget_e':.17,
            'representation_rule':'smallest single global grid passing all selected development checks; native256 if necessary',
            'joint_per_frame_cost_limit_s':180.,'candidate_recovery':'one declared numerical strategy switch, no new architecture',
            'traditional_grid':[10.,100.,1000.],'historical_traditional_lambda':.1,
            'v4_1_preselected_traditional_lambda':100.,'old_B4_retrained':False,
            'nominal_color_engineering_RMSE_K':20.,'normal_signed_template_noninferiority_ratio':1.05,
            'key_dimension_improvement':.10,'amplitude_projection_absolute_error_tolerance':.10,
            'old_projection_gates_reinterpreted':False,'no_favorable_exclusion':True},
        'targets':{'primary':'original optical postblur, reference20us electron template, full fixed truth domain',
            'extra_Gaussian_sigma_px':[0.,2.,4.],'color_smoothing':'smooth continuum bands then compute ratio',
            'latent_temperature_superresolution_claimed':False},
        'budget_study':{'relative_photon_ratios':[.25,1.,2.],'fixed_aperture_and_routing':True,
            'scale_continuum_template_background_together':True,'ADC_read_noise_not_renormalized':True},
        'structure_cases':['hotspot_small','hotspot_large','double_hotspot','edge','filament'],
        'strength_scan':{'reference_e':[0.,15.,30.,45.,90.,180.],'normalizations':['fixed_peak','fixed_total_signal']},
        'uncertainty':{'representative_mothers':12,'calibration_mothers':6,'evaluation_mothers':6,
            'initial_MC_per_case':64,'stability_MC_counts':[32,64,128],
            'stability_case_count':2,'MC_full_inverse':True,'truth_in_bootstrap_inference':False,
            'statistics':['five predeclared pixel points','predeclared ROI mean'],
            'simultaneous_full_image_intervals_claimed':False,'shared_raw_correlation_required':True,
            'regularized_Hessian_inverse_is_not_frequentist_covariance':True},
        'detection':{'task':'maximum absolute matched-filter dictionary statistic per frame',
            'dictionary_centers_xy':[[64,64],[128,64],[192,64],[64,128],[128,128],[192,128],[64,192],[128,192],[192,192]],
            'dictionary_sigma_px':[8.,16.,32.],'target_frame_FPR':.05,
            'calibration_independent_mothers':200,'evaluation_independent_mothers':200,
            'new_seed_base':8100000,'noise_replicates_are_not_new_mothers':True,'compatibility_threshold_e':60.},
        'independent_renderer':{'mothers':24,'seed_base':8200000,'wavelength_samples_per_band':321,
            'object_oversampling':2,'boundary':'finite nonperiodic convolution and crop',
            'forward_core_reused':False,'target_labels_from_reconstruction':False,
            'effects_separate':['nominal independent integration','relative_gain','response_shape','PSF','registration','emissivity_ratio']},
        'dynamic':{'sequences':2,'frames_per_sequence':6,'seed_base':8300000,'each_frame_independent_inverse':True},
        'sealed_final':{'main_mothers':old['sealed_final_mothers'],'signed_mothers':old['sealed_signed_confirmation_mothers'],
            'main_noisy_frames':512,'signed_noisy_frames':180,'supplement_for_signed_geometry_and_negative_spectra_required':True,
            'opening_rule':'source/method/optical targets/thresholds/statistics frozen; frozen v3 may trigger final test',
            'after_final_no_reselection_or_threshold_changes':True,'raw_payload_opened_at_registration':False},
        'new_complete_inverse_call_cap':12000,'all_new_inverse_calls_including_initializers_accounted':True,
        'call_cap_adjustment_before_first_formal_run':'12000 allows the complete final matrix, 12-mother MC, both initializers and strong traditional comparison; cached same-method same-raw results are reused',
        'prospective_phase_call_caps':{'source':540,'development':3400,'noise_structure_strength_calibration_dynamic':950,
            'uncertainty':2100,'null':1600,'independent':480,'sealed':2768},
        'phase_caps_are_maxima_global_cap_also_applies':True,'new_output_budget_bytes':4*2**30,'free_floor_bytes':2*2**30,
        'training':False,'real_experimental_validation':'not available; data interface and experimental plan required',
        'hardware_throughput':'not verified','final_methods':['selected joint or frozen v3','frozen v3','old B4','strong B1'],
        'reporting_baseline_configs':['historical B1 lambda0.1','v4.1 selected lambda100','new finite development selection'],
        'paper_language':'Chinese','figure_groups':7,'release_requires_success_receipt_no_redownload':True}

def register():
    value=plan();p=s.ROOT/'research_closeout/execution_spec.yaml'
    if p.exists(): assert s.read(p)==value
    else: s.write(p,value)  # JSON is a lossless YAML 1.2 subset.
    if not (s.REPORTS/'STUDY_PROTOCOL.yaml').exists(): s.write(s.REPORTS/'STUDY_PROTOCOL.yaml',value)
    checks=s.read(s.REPORTS/'NUMERICAL_CONTRACTS.json');assert checks['all_passed']
    receipt={'time':s.now(),'base_commit':s.BASE,'spec_sha256':s.sha(p),
        'numeric_check_sha256':s.sha(s.REPORTS/'NUMERICAL_CONTRACTS.json'),
        'source_sha256':source_hashes(),'raw_manifest_sha256':s.sha(s.ROOT/'research_v6/input_inventory.json'),
        'status':'FOUNDATION_REGISTERED','final_method_frozen':False,'sealed_payloads_opened':False}
    s.write(s.REPORTS/'FOUNDATION_REGISTRATION.json',receipt)
    return receipt

def count_call(phase,method,case_id,*,reused=False):
    path=s.REPORTS/'inverse_calls.jsonl'
    if not hasattr(count_call,'number'):
        previous=list(s.rows(path)) if path.exists() else []
        count_call.number=sum(not r['reused'] for r in previous)
    number=count_call.number
    cap=s.read(s.ROOT/'research_closeout/execution_spec.yaml')['new_complete_inverse_call_cap']
    if not reused and number>=cap: raise RuntimeError('Preregistered complete inverse call cap reached')
    s.append(path,{'time':s.now(),'phase':phase,'method':method,'case_id':case_id,'reused':reused,
        'new_call_number':number+(not reused)})
    count_call.number=number+(not reused)

def assert_final_frozen():
    f=s.read(s.REPORTS/'FINAL_METHOD_FREEZE.json')
    for name,digest in f['source_sha256'].items(): assert s.sha(s.ROOT/name)==digest
    assert f['detector_thresholds_frozen'] and f['statistics_frozen'] and f['targets_frozen']
    assert f['method'] in ('joint_profile','v3_physics')
    return f
