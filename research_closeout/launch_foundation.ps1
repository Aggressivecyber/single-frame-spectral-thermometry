$ErrorActionPreference = 'Stop'
$taskRoot = 'E:\CodexWork\thermometry\closeout_001'
$taskPython = 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe'
$taskReports = Join-Path $taskRoot 'reports\closeout_001'
$taskDependencyPid = 31104
$taskDependency = Get-CimInstance Win32_Process -Filter "ProcessId = $taskDependencyPid"
if ($taskDependency) {
    if ($taskDependency.CommandLine -notmatch 'research_v6\.repair_001\.run --phase stage-a') {
        throw 'Dependency PID was reused; inspect terminal evidence before running GPU work'
    }
    [ordered]@{status='WAITING_FOR_EXISTING_GPU_JOB';dependency_pid=$taskDependencyPid;at_utc=[DateTimeOffset]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskReports 'foundation_dependency.json') -Encoding UTF8
    Wait-Process -Id $taskDependencyPid -ErrorAction SilentlyContinue
}
$taskOtherGpuJob = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.ProcessId -ne $PID -and $_.CommandLine -match 'research_v6\.repair_001\.run --phase stage-a --device cuda'
}
if ($taskOtherGpuJob) { throw 'Another registered GPU job is active; do not overlap' }
Set-Location -LiteralPath $taskRoot
[ordered]@{status='DEPENDENCY_RELEASED_STARTING_FOUNDATION';at_utc=[DateTimeOffset]::UtcNow.ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskReports 'foundation_dependency.json') -Encoding UTF8
& $taskPython -u -B -m research_closeout.evaluate --phase foundation --device cuda
exit $LASTEXITCODE
