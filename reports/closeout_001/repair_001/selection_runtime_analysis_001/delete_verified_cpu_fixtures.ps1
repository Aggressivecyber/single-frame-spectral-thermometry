$ErrorActionPreference='Stop'
$workspace=(Resolve-Path -LiteralPath 'E:\CodexWork\thermometry\closeout_001').ProviderPath
$folder=Join-Path $workspace 'reports\closeout_001\repair_001\selection_runtime_analysis_001'
$manifestPath=Join-Path $folder 'CPU_FIXTURE_CLEANUP_MANIFEST.json'
$beforePath=Join-Path $folder 'ROOT_FIXTURE_CLEANUP_BEFORE.json'
$receiptPath=Join-Path $folder 'ROOT_FIXTURE_DELETE_RECEIPT.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Deletion already has a receipt; never repeat.' }
if (Test-Path -LiteralPath (Join-Path $workspace 'reports\closeout_001\PHASE_ACTIVE.json')) { throw 'A scientific phase is active.' }
if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne '77c1ea6b402ca667583a7e8ac8666eb30b3284e0503e3bb8accf0b81a392e172') { throw 'Audited cleanup manifest changed.' }
$manifest=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$before=Get-Content -LiteralPath $beforePath -Raw | ConvertFrom-Json
if (-not $manifest.all_six_runs_have_passed_CPU_zero_inverse_receipts -or $manifest.current_registration_direct_fixture_read_refs -ne 0 -or $manifest.runs.Count -ne 6) { throw 'CPU-only completed scope required.' }
$allowedRoots=@(
 (Join-Path $workspace 'reports\closeout_001\repair_001\c1_development_001\cpu_contracts\fixtures\'),
 (Join-Path $workspace 'reports\closeout_001\repair_001\c1_selection_001\cpu_contracts\fixtures\')
)
$targets=@();$files=0;$bytes=0L
foreach ($run in $manifest.runs) {
 $resolved=(Resolve-Path -LiteralPath $run.path).ProviderPath
 if ([IO.Path]::GetPathRoot($resolved) -ne 'E:\' -or [IO.Path]::GetFileName($resolved) -notmatch '^[0-9a-f]{32}$') { throw 'Unexpected deletion target.' }
 if (-not @($allowedRoots | Where-Object {$resolved.StartsWith($_,[StringComparison]::OrdinalIgnoreCase)}).Count) { throw 'Deletion target escapes the two explicit fixture roots.' }
 if ($before.targets -notcontains $resolved) { throw 'Target is outside the root-protected plan.' }
 $entries=@((Get-Item -LiteralPath $resolved -Force)) + @(Get-ChildItem -LiteralPath $resolved -Force -Recurse)
 if (@($entries | Where-Object {($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0}).Count) { throw 'Do not recursively delete reparse points.' }
 $actualFiles=@($entries | Where-Object {-not $_.PSIsContainer})
 if ($actualFiles.Count -ne $run.file_count) { throw 'Fixture file count changed.' }
 foreach ($file in $run.files) {
  $resolvedFile=(Resolve-Path -LiteralPath $file.path).ProviderPath
  if (-not $resolvedFile.StartsWith($resolved+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'File escapes verified fixture target.' }
  if ((Get-Item -LiteralPath $resolvedFile).Length -ne $file.bytes -or (Get-FileHash -LiteralPath $resolvedFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) { throw 'Fixture file bytes changed.' }
  $files++;$bytes+=$file.bytes
 }
 $targets+=$resolved
}
if ($files -ne 476 -or $bytes -ne 4464998) { throw 'The six-target cleanup scope changed.' }
$deleted=@();$failure=$null
try {
 foreach ($target in $targets) { Remove-Item -LiteralPath $target -Recurse -Force; $deleted+=$target }
} catch { $failure=$_.Exception.Message }
$receipt=[ordered]@{schema='root-verified-CPU-fixture-delete-v001'; time=[DateTime]::UtcNow.ToString('o');
 user_authority='Delete unused E-drive data and retain a brief'; deleted_paths=$deleted;
 deleted_directories=$deleted.Count; all_deleted=($deleted.Count -eq 6); planned_files=$files; planned_logical_bytes=$bytes;
 recursive_paths_validated_within_explicit_workspace_roots=$true; every_target_file_sha_verified=$true;
 failure=$failure; unique_scientific_outputs_deleted=$false; protected_after_verification_pending=$true}
[IO.File]::WriteAllText($receiptPath,(($receipt | ConvertTo-Json -Depth 6)+"`n"),[Text.UTF8Encoding]::new($false))
$receipt | ConvertTo-Json -Depth 6
if ($failure) { throw $failure }
