$ErrorActionPreference = 'Stop'
$workspace = 'E:\CodexWork\thermometry\closeout_001'
$folder = Join-Path $workspace 'reports\closeout_001\repair_001\c1_selection_001'
$receiptPath = Join-Path $folder 'PWEAK_SHARD_PROCESS.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Preserve existing launch receipt; no duplicate shard.' }
foreach ($lockPath in @((Join-Path $workspace 'reports\closeout_001\PHASE_ACTIVE.json'),(Join-Path $folder 'PHASE_ACTIVE.json'))) {
 if (Test-Path -LiteralPath $lockPath) { throw 'An active phase lock exists.' }
}
$completion = Get-Content -Raw -LiteralPath (Join-Path $folder 'BLANK_SHARD_COMPLETION_VERIFICATION.json') | ConvertFrom-Json
$plan = Get-Content -Raw -LiteralPath (Join-Path $folder 'PWEAK_SHARD_PLAN_REVIEW.json') | ConvertFrom-Json
if (-not $completion.all_passed -or -not $plan.all_passed) { throw 'Completed blank review and next-shard budget gate are required.' }
$registration = Join-Path $folder 'SELECTION_REGISTRATION.json'
$registrationSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $registration).Hash.ToLowerInvariant()
if ($registrationSha -ne 'f218cad4346e8f5fff2a6f176c87b728a5e4fe3fa47598597b606caacb1abf74') { throw 'Frozen registration changed.' }
$ledger = Join-Path $workspace 'reports\closeout_001\inverse_calls.jsonl'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $ledger).Hash.ToLowerInvariant() -ne $plan.ledger_sha256) { throw 'Ledger changed after the root plan check.' }
$cacheRoot = Join-Path $workspace 'outputs\closeout_001\_runtime'
$env:TEMP = Join-Path $cacheRoot 'tmp'; $env:TMP = $env:TEMP
$env:CUDA_CACHE_PATH = Join-Path $cacheRoot 'cuda'; $env:TORCH_HOME = Join-Path $cacheRoot 'torch'
$env:MPLCONFIGDIR = Join-Path $cacheRoot 'mpl'; $env:PYTHONDONTWRITEBYTECODE = '1'
$executable = 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe'
$caseId = 'v5_development_selection_mother_0000_positive_weak_noise0'
$arguments = @('-u','-B','-m','research_closeout.runtime_entry','--module','research_closeout.c1_selection','run-shard','--device','cuda','--raw',$caseId)
$stdout = Join-Path $folder 'positive_weak_shard_002.stdout.txt'
$stderr = Join-Path $folder 'positive_weak_shard_002.stderr.txt'
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'Preserve existing logs.' }
$commit = (& git -C $workspace rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot read source commit.' }
$requested = [DateTime]::UtcNow.ToString('o')
$process = Start-Process -FilePath $executable -ArgumentList $arguments -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$value = [ordered]@{
 schema='C1-first-stage-explicit-positive-weak-shard-launch-v001'; requested_utc=$requested
 start_utc=$process.StartTime.ToUniversalTime().ToString('o'); launcher_pid=$process.Id
 executable=$executable; arguments=$arguments; cwd=$workspace; source_commit=$commit
 registration_sha256=$registrationSha; raw_ids=@($caseId); planned_shard_cells=18; reused_cells=0
 maximum_new_inverse_calls=18; maximum_inverse_wall_s_excluding_overhead=3240
 global_charged_inverse_count_before_launch=5131; global_charged_inverse_count_maximum=5149
 human_approved_global_cap=16000; current_stricter_code_guard=12000; semantic_development_cap=3400
 full_first_stage_denominator=1080; full_stage_finite_before_launch=44; maximum_full_stage_finite_after_shard=62
 runtime_cache_root=$cacheRoot; stdout=$stdout; stderr=$stderr
 shard_receipts=(Join-Path $folder 'shards'); full_matrix_terminal=(Join-Path $folder 'FIRST_STAGE_RESULT.json')
 automatic_rerun_permitted=$false; automatic_next_shard_or_confirmation_permitted=$false
}
[IO.File]::WriteAllText($receiptPath,(($value | ConvertTo-Json -Depth 8)+"`n"),[Text.UTF8Encoding]::new($false))
$value | ConvertTo-Json -Depth 8
