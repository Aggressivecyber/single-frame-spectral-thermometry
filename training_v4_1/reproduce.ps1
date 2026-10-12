param(
    [Parameter(Mandatory=$true)][string]$Destination,
    [string]$DataSource = (Split-Path -Parent $PSScriptRoot),
    [string]$GpuPython = 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe',
    [string]$ReportPython = 'C:\Users\10629\AppData\Local\Programs\Python\Python313\python.exe',
    [switch]$Run
)
$ErrorActionPreference = 'Stop'
$sourceRepo = Split-Path -Parent $PSScriptRoot
$targetRoot = [IO.Path]::GetFullPath($Destination)
$inputRoot = [IO.Path]::GetFullPath($DataSource)
if ([IO.Path]::GetPathRoot($targetRoot) -notin @('E:\','D:\')) { throw 'Destination must be on E/D' }
if ([IO.Path]::GetPathRoot($inputRoot) -notin @('E:\','D:\')) { throw 'DataSource must be on E/D' }
if (Test-Path -LiteralPath $targetRoot) { throw 'Use a new destination; existing experiments cannot be overwritten' }
foreach ($relative in @('outputs\training_v4\run_001\raw', 'outputs\training_v4\run_001\labels',
                        'outputs\training_v4\run_001\cache', 'outputs\training_v4\run_001\gate_cache',
                        'outputs\training_v4\run_001\validation_predictions', 'outputs\repair_v3\run_001\fixtures')) {
    if (-not (Test-Path -LiteralPath (Join-Path $inputRoot $relative))) { throw "Missing frozen artifact: $relative" }
}
if (Get-ChildItem -LiteralPath (Join-Path $inputRoot 'outputs\training_v4\run_001\raw') -Filter 'v4_final_unopened_*') {
    throw 'DataSource contains final-test raw; use the development-only frozen archive'
}
& git clone --local --no-hardlinks $sourceRepo $targetRoot
if ($LASTEXITCODE -ne 0) { throw 'Local clone failed' }
& git -C $targetRoot checkout --detach 62538a8bb4a0014c764bd395cfa8cb9d915e9c97
if ($LASTEXITCODE -ne 0) { throw 'Required base commit is missing' }
Copy-Item -LiteralPath (Join-Path $sourceRepo 'training_v4_1') -Destination $targetRoot -Recurse
# Only archived v4/repair data are copied. New protection data and all trials
# are generated afresh; no final-test raw is generated or copied.
foreach ($relative in @('outputs\training_v4\run_001\raw', 'outputs\training_v4\run_001\labels',
                        'outputs\training_v4\run_001\cache', 'outputs\training_v4\run_001\gate_cache',
                        'outputs\training_v4\run_001\validation_predictions', 'outputs\repair_v3\run_001\fixtures')) {
    $target = Join-Path $targetRoot $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $inputRoot $relative) -Destination $target -Recurse
}
$newReports = Join-Path $targetRoot 'reports\training_v4_1\run_001'
New-Item -ItemType Directory -Path $newReports -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $sourceRepo 'reports\training_v4_1\run_001\preserved_baseline.json') -Destination $newReports
$runtimeRoot = Join-Path $targetRoot '_runtime'
New-Item -ItemType Directory -Path (Join-Path $runtimeRoot 'tmp') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $runtimeRoot 'cuda') -Force | Out-Null
if (Test-Path -LiteralPath (Join-Path $sourceRepo '_runtime\audit_deps')) {
    Copy-Item -LiteralPath (Join-Path $sourceRepo '_runtime\audit_deps') -Destination $runtimeRoot -Recurse
}
if ($Run) {
    Set-Location -LiteralPath $targetRoot
    $env:TMP=Join-Path $runtimeRoot 'tmp'; $env:TEMP=$env:TMP
    $env:CUDA_CACHE_PATH=Join-Path $runtimeRoot 'cuda'
    $env:MPLCONFIGDIR=Join-Path $runtimeRoot 'matplotlib'
    $env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'
    $env:PYTHONPATH=Join-Path $runtimeRoot 'audit_deps'
    & $GpuPython -B -c 'from training_v4_1.data import generate; generate()'
    if ($LASTEXITCODE -ne 0) { throw 'Data generation failed' }
    & $GpuPython -B -m pytest training_v4_1/tests.py
    if ($LASTEXITCODE -ne 0) { throw 'Pre-training checks failed' }
    & $GpuPython -B -m training_v4_1.prepare_cache
    if ($LASTEXITCODE -ne 0) { throw 'Cache preparation failed' }
    & $GpuPython -B -m training_v4_1.train --name safe1 --safe-weight 1
    if ($LASTEXITCODE -ne 0) { throw 'safe1 training failed' }
    & (Join-Path $targetRoot 'training_v4_1\finish_pipeline.ps1') -GpuPython $GpuPython -ReportPython $ReportPython
}
Write-Output "Fresh E/D reproduction checkout: $targetRoot"
