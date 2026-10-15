"""Update human continuation indexes from a verified startup snapshot only."""
from pathlib import Path
import datetime
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research_closeout import storage as s

folder = Path(__file__).resolve().parent
process = s.read(folder/'BLANK_SHARD_PROCESS.json')
startup = s.read(folder/'BLANK_SHARD_STARTUP_CONTINUED_VERIFICATION.json')
qualification = s.read(folder/'ROOT_QUALIFICATION_REVIEW.json')
if not startup['startup_verified'] or not qualification['all_passed']:
    raise RuntimeError('Only a verified startup may update the continuation indexes')
wake = '2026-10-15T23:00:00+08:00'
snapshot = {'phase': 'C1v3_selection_first_stage_v001_one_blank_raw', 'root': str(folder),
    'launcher_pid': process['launcher_pid'], 'actual_python_pid': startup['actual_python']['pid'],
    'owner_token': startup['owner_locks']['owner_token'], 'start_utc': process['start_utc'],
    'startup_utc': startup['time'], 'status_scope': 'Verified running at startup; do not monitor continuously',
    'raw_ids': process['raw_ids'], 'planned_cells': 18, 'reused_cells': 2, 'maximum_new_inverse_calls': 16,
    'full_first_stage_denominator': 1080, 'full_stage_prior_reuses': 28, 'full_stage_maximum_new': 1052,
    'maximum_inverse_wall_s_excluding_overhead': 2880, 'charged_before_run': 5115,
    'charged_at_startup': startup['global_charged_calls_at_observation'], 'charged_maximum_after_run': 5131,
    'registered_sources': 42, 'read_dependencies_checked': 661, 'CPU_contracts': 45,
    'native_namespace_qualification_passed': 4, 'native_namespace_new_inverse_calls': 0,
    'registration_sha256': process['registration_sha256'], 'registered_source_commit': process['source_commit'],
    'human_approved_cumulative_cap': 16000, 'current_stricter_code_guard': 12000,
    'semantic_development_cap': 3400, 'next_wake_local': wake,
    'automatic_next_shard_or_confirmation': False, 'final_payloads_opened': False}
status_path = s.REPORTS/'repair_001/REPAIR_STATUS.json'
status = s.read(status_path)
status.setdefault('historical_job_snapshots', []).append(status.get('current_job', {}))
status.update(time=s.now(), scientific_repair_acceptance='Original nine C1 candidates passed one-raw prescreen; no scientific selection. First development shard explicitly running at startup snapshot.',
    quick_phase='C1_first_stage_one_blank_raw_shard_running_at_startup_snapshot',
    current_job=snapshot, launcher_pid=snapshot['launcher_pid'], actual_python_pid=snapshot['actual_python_pid'],
    startup_verified_utc=startup['time'], next_wake_local=wake,
    registered_source_commit=process['source_commit'], registered_source_commit_remote_verified=True,
    global_inverse_budget={'human_authorized_cumulative':16000,'charged_before_current_shard':5115,
       'charged_at_startup':startup['global_charged_calls_at_observation'],'legacy_registered_guard_unchanged':12000,
       'extension_receipt':'repair_001/INVERSE_BUDGET_EXTENSION_16000.json'},
    whole_closeout_complete=False, final_payloads_opened=False,
    prescreen_completed={'cells':18,'finite':18,'numerically_certified':18,'reused':2,'new_calls':16,'Q':9,
       'parent_mothers':1,'distinct_raws':1,'terminal_sha256':'7a529ed4d699db9adc9113fb774e6974c9d93783422ce1c0a046b2e9e8c74514',
       'native_analysis_sha256':'043042a8cedff6182c73b9794425a79e571404215d7c8191b1099282b49b30dd'},
    main_figure_group2={'PNG':'paper/figures/F02_source_mechanism/SOURCE_MECHANISM_002.png',
       'PNG_visual_QA_passed':True,'PDF_visual_render_QA_performed':False,'seven_groups_complete':False})
s.write(status_path,status,replace=True)
policy_path=s.REPORTS/'repair_001/EXECUTION_POLICY.json'
policy=s.read(policy_path)
policy.update(time=s.now(),future_execution_model='gpt-6.1-sol',reasoning_effort='max',active_followup_agents=[],
    current_scope='One root-owned native first-stage blank-raw shard,18cells=2reused+16new maximum. All Sol61 followups stopped; one23:00 Singapore wake, no continuous monitoring.',
    human_authorized_cumulative_inverse_cap=16000,current_stricter_registered_code_guard=12000,
    storage_global_limit_bytes=4*2**30,next_wake_local=wake)
s.write(policy_path,policy,replace=True)
next_path=s.REPORTS/'NEXT_ACTION.md'
old=next_path.read_text(encoding='utf-8-sig')
header=f'''# Closeout 001 current continuation: verified blank-shard startup

Latest direct human approval raises the **cumulative** new complete-inverse cap from12000 to16000; it does not add16000 on top. See repair_001/INVERSE_BUDGET_EXTENSION_16000.json. Original study/spec/registrations remain byte-unchanged. This first-stage controller deliberately keeps the stricter12000 guard; it fits that guard. Future formal science needs a new explicit cap/phase adapter, not a silent edit to old registered sources. Storage remains4GiB across the four E/D data/report roots, E floor2GiB; D project256MiB and D floor128MiB independently.

Workspace E:/CodexWork/thermometry/closeout_001, branch research/closeout-001. Start with this section, repair_001/REPAIR_STATUS.json, EXECUTION_POLICY.json and c1_selection_001/SELECTION_REGISTRATION.json, ROOT_REGISTRATION_REVIEW.json, ROOT_QUALIFICATION_REVIEW.json, BLANK_SHARD_PROCESS.json and BLANK_SHARD_STARTUP_VERIFICATION.json. Then read USER_EXECUTION_REQUEST.txt, research_closeout/AGENTS.md, execution_spec.yaml, STUDY_PROTOCOL.yaml and root-cause reports. Legacy public_repository_001 inputs/weights/results are read-only hash references; all new files/temp/cache E/D. No CNN training. All future execution agents gpt-6.1-sol/max; older Astra reviews retain their meaning.

## Current only GPU job: one blank raw; wake23:00 Singapore

Root launched at{process['start_utc']}, launcher{snapshot['launcher_pid']}, actualPython{snapshot['actual_python_pid']}. Real command: `python -u -B -m research_closeout.runtime_entry --module research_closeout.c1_selection run-shard --device cuda --raw v5_development_selection_mother_0000_blank_noise0`. NT startup verified at{startup['time']}; E-global/E-local locks match owner{snapshot['owner_token']}, nativecuda:0/float64 and42 registered sources checked in the first ticket context, stderr0. Source/registration commit{process['source_commit']} was pushed and remoteSHA exactly verified before this job. These are startup snapshots, not continuous current counts.

This explicit shard has18 cells=2 oldc1s1 reuses+16 new complete inverses maximum, native256,180s each/2880s inverse work plus overhead. Before run charged5115; at startup{snapshot['charged_at_startup']}; maximum after this shard5131. Ledger5655 rows before run includes540 reused and is not the charged count. Full first-stage matrix remains12 mothers x5conditions x9candidates x2initializers=1080 cells,28 prior reuses+1052 new maximum. No automatic all-run, next shard, top-two choice or24-mother confirmation. Ordinary numerical failure retains each candidate's full120-cell denominator; resource stops are incomplete execution, not scientific rejection. Routes/initializers are not independent scenes.

At wake first read real NT PID/commands/parent-child and locks plus c1_selection_001/shards/*/RUN_*.json orSTOP_*.json; also SUMMARY_CURRENT.json, case_states/*/STATE_*.json and inference/cells/* receipts and stdout/stderr blank_shard_001.*.txt. FIRST_STAGE_RESULT.json is the terminal only for the entire1080-cell stage; absence after this one shard is expected. If process still running, schedule one later wake and end: no monitoring loop, observed-time restart, source edits or second GPU. Use the existing v3_tangent_probe_002/startup_observer.py process() NT reader, not WMI/CIM. For the current shard verify all18 attempted/reused outcomes, parameters/readback and numerical stages/platforms/full65536pixel scores, costs and full unexecuted matrix denominator. Same method/raw success, finite uncertified, failed or interrupted attempts always reuse. Scoring failure permits score-only, never a new inverse or renamed registration rerun.

## Just completed evidence

Original nine-pair prescreen ended12:17:11Z,18/18 finite/certified,2 old+16 new, Q9. Root native18 endpoint/latent reconstruction and full-domain rescoring all byte/matrix matched,0inverse/optimizer/inner; analysisSHA043042a8cedff6182c73b9794425a79e571404215d7c8191b1099282b49b30dd. Independent Sol61 CPU review checked54stages/36platforms and364 immutable objects; tighter cold/warm linear changes<=8.458e-9e for two fixed extreme-ratio endpoints. Actual stagesgtol1e-4/1e-5/1e-6, no1e-7. Earlystage parameters were not saved. This remains one raw/one parent, not18 independent cases or a scientific selection. Different regularizers trade temperature/template error; all nine continue the original12-mother stage without a new grid.

New first-stage controller has45/45 CPU contracts and real180raw/v3-state provenance audit,342 protected prior objects unchanged. RegistrationSHAf218cad4346e8f5fff2a6f176c87b728a5e4fe3fa47598597b606caacb1abf74;42 source AST/worktree/Git-index exact,661 read dependencies checked. Separate new namespace qualification4/4 saved endpoints passed native read-only forward reconstruction and lossless persistence,0 inverse/optimizer/inner/truth. Original saved canonical and latent are compared exactly; unsavedv001 latent is not directly compared. Parameterfloat64 first then readback/forward hashes before scoring; per-frame successbound1385504B/initialpeak7873568B. Whole-phase2GiB success plan is conditional on actual fallback/storage; runtime reserves fail closed rather than deleting science or pretending all1052 fallbacks fit. Prior E/D evidence/old failures unchanged.

Budget review SELECTION_STORAGE_AND_BUDGET_001.json reconciles semanticdevelopment474 existing +1052firststage+480two-candidate confirmation=2006/3400, global6647 after both. New-primary mandatoryUQ1820 plusv3 independent896MC branch totals13030 or provisionally14757 with missing initializer states; v3-primary identical-route reuse totals10615. Full1820 forboth is optional. REMAINING_CORRECTED_MATRIX_SCOPE_REVIEW.json flags that matching24 nominal scenes alone may leave original146 independent effects and45calibration/12dynamic cases uncorrected; full matched-effect reruns would add716 calls (conditional15473 worst v3-init branch) but require fresh source/seed/call/storage registration. This is planning, not already executed science or proven4GiB fit. Resolve old missingv3-state materialization against the no-rerun rule before choosing that branch; old canonical outputs are not complete state. Do not select/prune candidates for budget.

Main figure group2 SOURCE_MECHANISM_002.png is generated from immutable36-mother source comparison and6-mother finite decomposition, visually reviewed; v1 annotation-overlap draft is preserved, v2 fixes only layout. Pairedparent10000 bootstrap is a development descriptor. Four-error signed spatial cross terms are not independent variances. PDF header/EOF only, no PDF render QA. Nine-pair supplementary PNG also passed visualQA. Chinese paper/manuscript_zh.md andclaims_to_evidence.md include these, still work drafts; seven main groups are not complete. Primary literature matrix now distinguishes original2017FRAME/2018FRAME/2024passive/2026thermometry and authors'LSMR, with access levels; temperature/emissivityfulltext remains unavailable, only abstract read.

After shard review continue explicit small scheduling batches toward all1080 cells, then scientific top-two/24-mother confirmation under original rule; no new grid or truth-driven pruning. Primary may be frozenv3 research baseline if no reliable benefit. Corrected independent/null/UQ studies remain: new balanced12-mother6cal/6eval allsix actual geometries, primary12+896MC+768conditionalbootstrap+144probes=1820, v3 independent896MC; bootstrap is only conditional scale. New-method200nullcal+200nulleval with method/domain thresholds, matched independent/3resolution/3photon/structure/strength/calibration/dynamic matrices and all failures. C1 exposure must scale newbuffer slopes via known_exposure. Real calibration absent, simulated distributions assumed.

Final source/dependencies/methods/params/optics/thresholds/UQ/baselines/statistics must all freeze and FINAL_METHOD_FREEZE.json verify before64main+36signed payload generation/open. Payloads remain unopened. Human explicitly permitsv3 final evaluation; final failure still goes in paper. Never use final outputs to tune/reselect/checkpoints. Complete seven figure groups/visualQA, Chinese manuscript/data-method-limitations/claims/raw+calibration interface/clean reproduction package. Final GitHub release preserves old releases and upload receipts only, no re-download. No realgas/AlOconcentration/10kHzhardware/journalguarantee claims. A successful shard/contract/probe/working draft is not full closeout.

## Historical continuation snapshots

The following text is retained historical context. The latest section above supersedes earlier active-job and12000-budget instructions; original scientific results retain their original meaning.

'''
next_path.write_text(header+old,encoding='utf-8',newline='\n')
print(json.dumps({'updated':str(next_path),'current_job':snapshot,'whole_closeout_complete':False},indent=2),flush=True)
