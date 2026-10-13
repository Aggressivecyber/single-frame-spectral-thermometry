param([string]$GpuPython='E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
if ([IO.Path]::GetPathRoot($taskRoot) -notin @('E:\','D:\')) { throw 'Use an E/D checkout' }
Set-Location -LiteralPath $taskRoot
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'
$env:TMP=Join-Path $taskRoot '_runtime\tmp'; $env:TEMP=$env:TMP
$env:CUDA_CACHE_PATH=Join-Path $taskRoot '_runtime\cuda'
$env:MPLCONFIGDIR=Join-Path $taskRoot '_runtime\matplotlib'
$env:PYTHONPATH=Join-Path $taskRoot '_runtime\audit_deps'
& $GpuPython -B -m training_v4_2.experiment
if ($LASTEXITCODE -ne 0) { throw 'Identity restart failed; inspect the retained failure record' }
