$ErrorActionPreference = 'Stop'
$workspace = 'E:\CodexWork\thermometry\closeout_001'
$folder = Join-Path $workspace 'reports\closeout_001\repair_001\c1_development_001'
$receiptPath = Join-Path $folder 'PRESCREEN_PROCESS.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Launch receipt already exists; do not repeat.' }
$globalLock = Join-Path $workspace 'reports\closeout_001\PHASE_ACTIVE.json'
$localLock = Join-Path $folder 'PHASE_ACTIVE.json'
if ((Test-Path -LiteralPath $globalLock) -or (Test-Path -LiteralPath $localLock)) { throw 'An active phase lock exists.' }
$qualification = Get-Content -Raw -LiteralPath (Join-Path $folder 'ROOT_QUALIFICATION_REVIEW.json') | ConvertFrom-Json
if (-not $qualification.all_passed -or $qualification.passed_existing_endpoints -ne 4 -or $qualification.complete_inverse_calls -ne 0) { throw 'Native E-cache qualification has not passed.' }
$registration = Join-Path $folder 'PRESCREEN_REGISTRATION.json'
$registrationSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $registration).Hash.ToLowerInvariant()
if ($registrationSha -ne 'a03f326859fd46de4b1cb1b47d6683d9b8e271d35faa7eef9a2093ab1dbf1032') { throw 'Registered prescreen bytes changed.' }
$cacheRoot = Join-Path $workspace 'outputs\closeout_001\repair_001\quick_v003\runtime_cache'
$env:TEMP = Join-Path $cacheRoot 'tmp'
$env:TMP = $env:TEMP
$env:CUDA_CACHE_PATH = Join-Path $cacheRoot 'cuda'
$env:TORCH_HOME = Join-Path $cacheRoot 'torch'
$env:MPLCONFIGDIR = Join-Path $cacheRoot 'mpl'
$env:PYTHONDONTWRITEBYTECODE = '1'
$executable = 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe'
$arguments = @('-u','-B','-m','research_closeout.runtime_entry','--module','research_closeout.c1_development','prescreen','--device','cuda')
$stdout = Join-Path $folder 'prescreen_001.stdout.txt'
$stderr = Join-Path $folder 'prescreen_001.stderr.txt'
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'Preserve existing launch logs.' }
$requested = [DateTime]::UtcNow.ToString('o')
$process = Start-Process -FilePath $executable -ArgumentList $arguments -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$value = [ordered]@{
  schema = 'C1-nine-pair-prescreen-launch-v001'
  requested_utc = $requested
  start_utc = $process.StartTime.ToUniversalTime().ToString('o')
  launcher_pid = $process.Id
  executable = $executable
  arguments = $arguments
  cwd = $workspace
  source_commit = '978a29fc53654057dda61a9c3c92f3f36bbc776c'
  registration_sha256 = $registrationSha
  planned_cells = 18
  reused_cells = 2
  maximum_new_inverse_calls = 16
  inverse_wall_budget_s_excluding_overhead = 2880
  global_charged_inverse_count_before_launch = 5099
  global_charged_inverse_count_maximum = 5115
  runtime_cache_root = $cacheRoot
  stdout = $stdout
  stderr = $stderr
  terminal = (Join-Path $folder 'PRESCREEN_RESULT.json')
  automatic_rerun_permitted = $false
  automatic_selection12_confirmation24_permitted = $false
}
[IO.File]::WriteAllText($receiptPath, (($value | ConvertTo-Json -Depth 8) + "`n"), [Text.UTF8Encoding]::new($false))
$value | ConvertTo-Json -Depth 8
