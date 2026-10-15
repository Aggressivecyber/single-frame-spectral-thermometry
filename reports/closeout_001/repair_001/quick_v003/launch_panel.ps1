$ErrorActionPreference = 'Stop'
$taskWorkspace = 'E:\CodexWork\thermometry\closeout_001'
$taskReport = Join-Path $taskWorkspace 'reports\closeout_001\repair_001\quick_v003'
$taskDReport = 'D:\CodexWork\thermometry\closeout_001\reports\repair_001\quick_v003'
$taskPython = 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe'
$taskStdout = Join-Path $taskReport 'panel_001.stdout.txt'
$taskStderr = Join-Path $taskReport 'panel_001.stderr.txt'
$taskPlan = Join-Path $taskReport 'PANEL_LAUNCH_PLAN.json'
$taskProcessReceipt = Join-Path $taskReport 'PANEL_PROCESS.json'
$taskArgs = @('-u', '-B', '-m', 'research_closeout.runtime_entry', '--module', 'research_closeout.c1_small_panel', 'run', '--device', 'cuda')

foreach ($taskForbidden in @(
    $taskPlan, $taskProcessReceipt, $taskStdout, $taskStderr,
    (Join-Path $taskWorkspace 'reports\closeout_001\PHASE_ACTIVE.json'),
    (Join-Path $taskDReport 'PHASE_ACTIVE.json'),
    (Join-Path $taskDReport 'PANEL_RESULT.json')
)) {
    if (Test-Path -LiteralPath $taskForbidden) { throw "Inspect existing launch or run state; never auto-repeat: $taskForbidden" }
}
if (Test-Path -LiteralPath (Join-Path $taskDReport 'inference')) {
    throw 'An inference subtree already exists; inspect and reuse its state before any launch.'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskDReport 'NATIVE_COMPACT_QUALIFICATION.json.sha256.json'))) {
    throw 'Sealed native qualification is required.'
}
if ((Get-FileHash -LiteralPath (Join-Path $taskDReport 'PANEL_REGISTRATION.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'f9b94df9e49e0755d3f0befb00dc0675a81df3d48f4f2ea9955e3a203b2aac9d') {
    throw 'Panel registration differs.'
}
$taskCache = Join-Path $taskWorkspace 'outputs\closeout_001\repair_001\quick_v003\runtime_cache'
foreach ($taskCacheSub in @('tmp', 'cuda', 'torch', 'inductor', 'triton', 'hf', 'mpl', 'xdg', 'numba')) {
    $null = New-Item -ItemType Directory -Force -Path (Join-Path $taskCache $taskCacheSub)
}
$env:TEMP = Join-Path $taskCache 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_CACHE_PATH = Join-Path $taskCache 'cuda'
$env:TORCH_HOME = Join-Path $taskCache 'torch'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $taskCache 'inductor'
$env:TRITON_CACHE_DIR = Join-Path $taskCache 'triton'
$env:HF_HOME = Join-Path $taskCache 'hf'
$env:MPLCONFIGDIR = Join-Path $taskCache 'mpl'
$env:XDG_CACHE_HOME = Join-Path $taskCache 'xdg'
$env:NUMBA_CACHE_DIR = Join-Path $taskCache 'numba'
Set-Location -LiteralPath $taskWorkspace
$taskCommit = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot read source commit.' }
$taskValue = [ordered]@{
    schema = 'C1-small-panel-eight-new-calls-launch-v001'
    requested_utc = [DateTime]::UtcNow.ToString('o')
    executable = $taskPython
    arguments = $taskArgs
    cwd = $taskWorkspace
    source_commit = $taskCommit
    registration_sha256 = 'f9b94df9e49e0755d3f0befb00dc0675a81df3d48f4f2ea9955e3a203b2aac9d'
    planned_cells = 12
    reused_cells = 4
    maximum_new_inverse_calls = 8
    inverse_wall_budget_s_excluding_overhead = 1440
    zero_reserves_remain_charged_until_verified_terminal_cleanup = $true
    runtime_cache_root = $taskCache
    stdout = $taskStdout
    stderr = $taskStderr
    terminal = (Join-Path $taskDReport 'PANEL_RESULT.json')
    automatic_rerun_permitted = $false
}
$taskValue | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $taskPlan -Encoding UTF8
$taskJob = Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WorkingDirectory $taskWorkspace -WindowStyle Hidden -PassThru -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr
$taskValue.launcher_pid = $taskJob.Id
$taskValue.start_utc = [DateTime]::UtcNow.ToString('o')
$taskValue | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $taskProcessReceipt -Encoding UTF8
$taskValue | ConvertTo-Json -Depth 20
