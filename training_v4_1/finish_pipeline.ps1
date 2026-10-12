param(
    [string]$GpuPython = 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe',
    [string]$ReportPython = 'C:\Users\10629\AppData\Local\Programs\Python\Python313\python.exe'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
if ([IO.Path]::GetPathRoot($taskRoot) -notin @('E:\', 'D:\')) { throw 'Use an E/D checkout' }
Set-Location -LiteralPath $taskRoot
$taskReports = Join-Path $taskRoot 'reports\training_v4_1\run_001'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TMP = Join-Path $taskRoot '_runtime\tmp'
$env:TEMP = $env:TMP
$env:CUDA_CACHE_PATH = Join-Path $taskRoot '_runtime\cuda'
$env:MPLCONFIGDIR = Join-Path $taskRoot '_runtime\matplotlib'
$env:PYTHONPATH = Join-Path $taskRoot '_runtime\audit_deps'

function Write-PipelineStatus([string]$status, [string]$stage, [string]$message = '') {
    @{status=$status; stage=$stage; message=$message; timestamp=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskReports 'pipeline_status.json') -Encoding utf8
}
function Invoke-Stage([string]$name, [string]$python, [string[]]$arguments) {
    Write-PipelineStatus 'RUNNING' $name
    $stageLog = Join-Path $taskReports ($name+'.log')
    & $python @arguments *> $stageLog
    if ($LASTEXITCODE -ne 0) { throw "$name failed with exit $LASTEXITCODE; see $stageLog" }
}
try {
    Write-PipelineStatus 'WAITING' 'safe1_training_receipt'
    while (-not (Test-Path -LiteralPath (Join-Path $taskReports 'safe1\training_receipt.json'))) {
        if (Test-Path -LiteralPath (Join-Path $taskReports 'safe1\failure_record.json')) { throw 'safe1 failed; inspect failure_record.json' }
        Start-Sleep -Seconds 1
    }
    Invoke-Stage 'controls' $GpuPython @('-B','-m','training_v4_1.controls')
    Write-PipelineStatus 'WAITING' 'complete_physical_cache'
    do {
        $producerState = Get-Content -LiteralPath (Join-Path $taskReports 'cache_producer_status.json') -Raw | ConvertFrom-Json
        if ($producerState.status -eq 'FAILED') { throw $producerState.message }
        if ($producerState.status -ne 'COMPLETE') { Start-Sleep -Seconds 1 }
    } until ($producerState.status -eq 'COMPLETE')
    Invoke-Stage 'tikhonov' $GpuPython @('-B','-m','training_v4_1.tikhonov')
    Invoke-Stage 'compare' $GpuPython @('-B','-m','training_v4_1.compare')
    Invoke-Stage 'tests_final' $GpuPython @('-B','-m','pytest','training_v4_1/tests.py','--junitxml=reports/training_v4_1/run_001/tests_final.xml')
    Invoke-Stage 'verify' $GpuPython @('-B','-m','training_v4_1.verify')
    Invoke-Stage 'report' $ReportPython @('-B','-m','training_v4_1.report')
    Write-PipelineStatus 'COMPLETE' 'report'
} catch {
    Write-PipelineStatus 'FAILED' 'pipeline' $_.Exception.Message
    throw
}
