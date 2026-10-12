param(
    [Parameter(Mandatory=$true)][string]$Worktree,
    [string]$Python = 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe',
    [string]$ReportPython = 'C:\Users\10629\AppData\Local\Programs\Python\Python313\python.exe'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$replayRoot = [IO.Path]::GetFullPath($Worktree)
if ([IO.Path]::GetPathRoot($replayRoot) -notin @('E:\','D:\')) { throw 'Worktree must be on E or D' }
if (Test-Path -LiteralPath $replayRoot) { throw 'Choose a new directory to preserve existing results' }
git -C $repo worktree add --detach $replayRoot 9a7ac7e86339ba9b9d3006765fef731f10941029
if ($LASTEXITCODE -ne 0) { throw 'Could not create baseline worktree' }
Copy-Item -LiteralPath $PSScriptRoot -Destination (Join-Path $replayRoot 'training_v4') -Recurse
$reports = Join-Path $replayRoot 'reports\training_v4\run_001'
New-Item -ItemType Directory -Path $reports -Force | Out-Null
# Historical rejected experiment is evidence only, never an inference backend.
Copy-Item -LiteralPath (Join-Path $repo 'reports\training_v4\run_001\batched_engine_rejected.json') -Destination $reports
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TMP = Join-Path $replayRoot '_runtime\tmp'
$env:TEMP = $env:TMP
$env:CUDA_CACHE_PATH = Join-Path $replayRoot '_runtime\cuda'
$env:PYTHONPATH = Join-Path $repo '_runtime\audit_deps'
New-Item -ItemType Directory -Path $env:TMP -Force | Out-Null
Push-Location -LiteralPath $replayRoot
try {
    & $Python -B -m training_v4.prepare_replay
    if ($LASTEXITCODE -ne 0) { throw 'Replay preparation failed' }
    foreach ($phase in @('data','select','overfit','train','compare')) {
        & $Python -B -m training_v4.run --phase $phase --seed 101
        if ($LASTEXITCODE -ne 0) { throw "Phase failed: $phase" }
    }
    $decision = Get-Content -LiteralPath (Join-Path $reports 'seed_101\benefit_decision.json') -Raw | ConvertFrom-Json
    if ($decision.clear_validation_gain) {
        foreach ($seed in @(202,303)) {
            & $Python -B -m training_v4.run --phase train --seed $seed
            if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $seed" }
            & $Python -B -m training_v4.run --phase compare --seed $seed
            if ($LASTEXITCODE -ne 0) { throw "Comparison failed for seed $seed" }
        }
    }
    & $Python -B -m pytest training_v4/tests.py --junitxml=reports/training_v4/run_001/tests_final.xml
    if ($LASTEXITCODE -ne 0) { throw 'Tensor/cache checks failed' }
    & $Python -B -m training_v4.verify_delivery
    if ($LASTEXITCODE -ne 0) { throw 'Native checkpoint reopening failed' }
    & $ReportPython -B -m training_v4.run --phase report
    if ($LASTEXITCODE -ne 0) { throw 'Report verification failed' }
} finally { Pop-Location }
