$ErrorActionPreference = 'Stop'
$repairRepo = 'E:\CodexWork\thermometry\public_repository_001'
$repairReports = Join-Path $repairRepo 'reports\research_v6\repair_001\processes'
$repairPython = 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe'
New-Item -ItemType Directory -Path $repairReports -Force | Out-Null
$repairStdout = Join-Path $repairReports 'stage_a_001.stdout.txt'
$repairStderr = Join-Path $repairReports 'stage_a_001.stderr.txt'
$repairReceipt = Join-Path $repairReports 'stage_a_001.process.json'
if ((Test-Path -LiteralPath $repairStdout) -or (Test-Path -LiteralPath $repairReceipt)) { throw 'Stage A launch evidence already exists; do not restart blindly.' }
$repairProcess = Start-Process -FilePath $repairPython -ArgumentList @('-u','-B','-m','research_v6.repair_001.run','--phase','stage-a','--device','cuda') -WorkingDirectory $repairRepo -WindowStyle Hidden -RedirectStandardOutput $repairStdout -RedirectStandardError $repairStderr -PassThru
$repairMetadata = [ordered]@{ pid=$repairProcess.Id; started_at=[DateTimeOffset]::UtcNow.ToString('o'); executable=$repairPython; arguments='-u -B -m research_v6.repair_001.run --phase stage-a --device cuda'; stdout=$repairStdout; stderr=$repairStderr; one_root_GPU_process=$true }
$repairMetadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $repairReceipt -Encoding utf8
$repairMetadata | ConvertTo-Json -Depth 4
