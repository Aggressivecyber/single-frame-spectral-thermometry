"""Root evidence closure, keeping delivery pending until successful upload."""
from research_v5 import common
common.configure_runtime()
import argparse

R=common.REPORTS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--delivery-complete',action='store_true')
    a=p.parse_args()
    root=common.read_json(R/'root_review_001/ROOT_DATA_AUDIT.json')
    end=common.read_json(R/'execution_recovery_001/END.json')
    assert root['all_audit_checks_passed'] and end['all_planned_diagnostics_accounted_for']
    plans=common.read_json(R/'ROOT_ACCEPTANCE_REVIEW_PLAN_001.json')
    entries={
      'S0_IMMUTABILITY':('COMPLETED',None,['root_review_001/ROOT_DATA_AUDIT.json','baseline_readonly_inventory.json','old_array_readonly_inventory.json'],'1452 old tracked files byte hashes and33 source hashes match;12869 old arrays metadata/readonly match without opening sealed payloads.'),
      'S1_ORACLE':('COMPLETED',True,['ROOT_ORACLE_EVIDENCE_REVIEW.json','ORACLE_SUMMARY.json'],'180 unique raw cases x4=720;84x4=336 baselines; O0 clean normal gate passes.'),
      'S1_NUMERICS':('COMPLETED_WITH_PRECISION_ADDENDUM',None,['ROOT_INFORMATION_EVIDENCE_REVIEW.json','numerical_precision/summary_002.json'],'Original6/2016 float32 failures preserved; same2016 float64 probes pass3e-11 and192 FD pass;24 finite-mode Fisher/optical cases. Not full-image CRLB.'),
      'S2_GRID':('COMPLETED_WITH_FAILURES',False,['root_review_001/ROOT_DATA_AUDIT.json','development/records.jsonl','execution_recovery_001/END.json'],'Exact4320keys,4318successful and2nonselected numerical failures retained; original wall timeout preserved.'),
      'S2_SELECTION':('COMPLETED_NEGATIVE_RESULT',False,['development/selection.json'],'All four registered alphas infeasible;alpha1 diagnosticfallback;traditional v3 selected by original rule; no reselection.'),
      'S2_COLOR':('COMPLETED_NEGATIVE_RESULT',False,['root_review_001/RESULT_SUMMARY.json','root_review_001/PER_MOTHER_NOMINAL.csv'],'Nominal noisy180frames/36mothers RMSE28.5398826K>20K;mother95%CI[26.8342,30.4089]K; latent/reference targets separate.'),
      'S2_SIGNED_RMSE':('COMPLETED_NEGATIVE_RESULT',False,['development/paired_statistics.json','development/development_gate.json'],'Normal positive/negative RMSE ratios3.75867/3.70212>1.05; all36matchedmothers retained eachsign.'),
      'S2_SIGNED_PROJECTION':('COMPLETED_NEGATIVE_RESULT',False,['execution_recovery_001/scientific_stop.json'],'Both signs fail original Task4B and final-rule noninferiority. Frozen development boolean omission is explicitly supplemented at root review, not by source modification.'),
      'S2_CG':('COMPLETED_SELECTED_CONFIGURATION_PASSED',True,['development/development_gate.json','root_review_001/RESULT_SUMMARY.json'],'Selected noisy180 allconverged, true residual max9.84244e-5,valid1; failures on other alphas retained.'),
      'S2_NULL':('COMPLETED_EMPIRICAL_GATE_PASSED',True,['development/null_threshold.json'],'96calibration/96disjoint evaluation,threshold48.1619529724 frozenbeforeevaluation;FP2/96 withmotherCI[0,.0520833], no instrument guarantee.'),
      'S2_MC':('COMPLETED_NEGATIVE_RESULT_WITH_LOG_LIMITATION',False,['development/mc_coverage.json','root_review_001/MC_ARRAY_REOPEN_REVIEW.json'],'12normalcases x8repeats,6mothermean signedcoverage.903968811 andcolor.897291819<.90. Original3repeatlogs unavailable; weakcoverage unmeasured.'),
      'S2_HESSIAN':('COMPLETED_NUMERICAL_FAILURE',False,['development/hessian_diagnostics.json','execution_recovery_001/END.json'],'All8actualprobeIDs0..7 attempted200iterations;0converged,0validcovariancecoordinates. Failure diagnostic complete, covariance unavailable.'),
      'S2_STRESS':('COMPLETED_NEGATIVE_RESULT',False,['development/calibration_stress_records.jsonl','root_review_001/RESULT_SUMMARY.json'],'36x5=180rows;gain/emissivity nominalcolorRMSE110.464/176.521K;valid1,abstention0. Successful mismatch rejection was not established.'),
      'S2_STATISTICS':('COMPLETED',None,['development/paired_statistics.json','root_review_001/RESULT_SUMMARY.json','root_review_001/PER_MOTHER_NOMINAL.csv'],'Six paired normal/weak/sign comparisons reproduced exactly; colorCI added withsame2000/53130motherbootstrap;72weakarraydetectionsrescored,3thresholdpixelsallofftruthsupport.'),
      'S2_DEPLOYMENT_BOUNDARY':('COMPLETED_IMPLEMENTATION_AND_EXECUTION_REVIEW',None,['SOURCE_REGISTRATION_CHECKS.json','ROOT_FINAL_REPORT.md','root_review_001/ROOT_DATA_AUDIT.json'],'Frozen raw-only API/contracts plus actual source-bound runs; no duplicated likelihood or continuum feedback. Implementation acceptance is separate from failed scientific performance.'),
      'S3_NEURAL':('NOT_APPLICABLE_REGISTERED_SCIENTIFIC_STOP',None,['CONDITIONAL_STAGE_CLOSURE.json','TRAIN_V5_REPORT.md'],'Physical prerequisite failed;0optimizerupdates,noCNNcheckpoint,benefitunknown.'),
      'S4_CANDIDATE':('NOT_APPLICABLE_NO_ELIGIBLE_CANDIDATE',None,['CONDITIONAL_STAGE_CLOSURE.json'],'No scientifically eligible candidate or final_protocol created; no candidate-review run claimed.'),
      'S4_FINAL':('NOT_APPLICABLE_SEALED_BY_SCIENTIFIC_STOP',None,['CONDITIONAL_STAGE_CLOSURE.json','root_review_001/ROOT_DATA_AUDIT.json'],'64original+36signedmothers remain sealed;0finalcaseevaluations;accuracyunknown,notafailedfinaltest.'),
      'FINAL_DECISION':('COMPLETED',None,['V5_SCIENTIFIC_DECISION.md','ROOT_FINAL_REPORT.md','root_review_001/INDEPENDENT_FINAL_REVIEW.json'],'Allsixquestionsanswered;rawproxy/Fisher/MC/calibration/failureboundariesexplicit;figurevisuallyreviewed.'),
      'DELIVERY':('COMPLETED' if a.delivery_complete else 'PENDING_GITHUB_UPLOAD',None,['delivery_001/END.json'] if a.delivery_complete else [],'GitHub successful push and all release asset upload/publish exit codes required. No re-download verification requested.')}
    if a.delivery_complete:
        delivery=common.read_json(R/'delivery_001/END.json')
        assert delivery['status']=='COMPLETED_GITHUB_RELEASE' and delivery['all_assets_uploaded'] and delivery['published']
    requirements=[]
    for plan in plans['expected_evidence']:
        status,science,paths,finding=entries[plan['id']]
        requirements.append({'id':plan['id'],'requirement':plan['requirement'],'status':status,'scientific_acceptance_passed':science,'finding':finding,'evidence':[{'path':common.relative(R/path),'sha256':common.sha(R/path)} for path in paths]})
    value={'schema':'v5-root-task-completion-audit-v2','time':common.now(),'origin':'Root review supersedes automatic file-presence draft archived in root_review_001/automatic_drafts','task_spec_sha256':common.sha(common.CODE/'TASK_SPEC.md'),'protocol_sha256':common.sha(common.CODE/'protocol.json'),'requirements':requirements,'all_applicable_scientific_work_accounted_for':True,'all_applicable_delivery_complete':a.delivery_complete,'user_objective_complete':a.delivery_complete,'source_frozen_and_unchanged':True,'research_route_fully_audited':True,'estimator_scientific_acceptance_passed':False,'numerical_execution_all_passed':False,'scientific_final_acceptance_passed':None,'final_performance':'UNKNOWN_SEALED_NOT_RUN','sealed_evaluation_started':False,'sealed_arrays_generated':False,'sealed_evaluation_completed':False,'formal_training_optimizer_updates':0,'negative_results_preserved':True,'root_actual_audit_check_count':len(root['checks']),'visual_QA':{'path':'root_review_001/V5_DECISION_EVIDENCE.png','actually_viewed_by_root':True,'result':'Four panels readable; correct color/template errors, MC95% nominal vs90% gate, stressreferencevsnominal andzeroabstention; no clipping or label overlap'},'limitations':['Original21600srun timedout; separately registered5400ssupplement used2299.515s.','OriginalthreeMC repeatlogs unavailable; arrays andcoveragechecked; weakMCcoverage unmeasured.','Originalsixfloat32 adjointfailures retained with highprecisionreferencepass.','Two nonselectedgrid solves andall8Hessianprobes failed.','Unknown calibrationstress not automatically rejected.','Finite-modeFisher andrawproxy do not provefull-imageCRLB orweakdetectability.','Syntheticcolor/templateonly; noexperimentalgastemperature/concentration/10kHzclaim.','Old array byte-level identity was not newly rehashed; frozenmetadata/readonly andpriorgenerationhashesretained toprotectsealedpayloads.']}
    common.write_json(R/'COMPLETION_AUDIT.json',value,replace=True)
    print({'requirements':len(requirements),'research_accounted_for':True,'delivery_complete':a.delivery_complete})


if __name__=='__main__':main()
