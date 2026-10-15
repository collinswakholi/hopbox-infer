# Run after guest.ps1 completes, against its actual read-only EXE installation.
$ErrorActionPreference='Stop'
$results='C:\HopBox-QA\Results'
$exe='C:\Program Files\HopBox Infer QA\HopBox-Infer.exe'
$record=[ordered]@{pass=$false;phase='integrity';started_utc=[DateTime]::UtcNow.ToString('o')}
try {
    $report=Join-Path $results 'installed-integrity.json'
    $diagnostic=Start-Process -FilePath $exe -ArgumentList ('--diagnose "'+$report+'"') -WindowStyle Hidden -PassThru -Wait
    $integrity=Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
    if($diagnostic.ExitCode -ne 0 -or -not $integrity.integrity_pass){throw 'Installed EXE integrity check failed'}
    $record.integrity_pass=$true
    $record.phase='explicit CPU startup'
    $env:PATH=Join-Path $env:SystemRoot 'System32'
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    $env:HOPBOX_PORT='8779'
    $env:HOPBOX_STATE_ROOT=Join-Path $env:LOCALAPPDATA 'HopBox-VM-CPU-CLI'
    $env:HOPBOX_NO_BROWSER='1'
    # The explicit CLI argument must override a pre-existing GPU preference.
    $env:HOPBOX_DEVICE='dml:0'
    $launcher=Start-Process -FilePath $exe -ArgumentList '--cpu' -WindowStyle Hidden -PassThru
    $deadline=[DateTime]::UtcNow.AddMinutes(3)
    do {
        Start-Sleep -Seconds 1
        try {$ready=Invoke-RestMethod -Uri 'http://127.0.0.1:8779/api/ready' -UseBasicParsing -TimeoutSec 5} catch {$ready=$null}
    } while(-not $ready.ok -and [DateTime]::UtcNow -lt $deadline)
    if(-not $ready.ok -or $ready.detail -notmatch '^CPU'){throw 'The --cpu launch option failed to select CPU'}
    $server=Get-Process -Id $ready.pid
    $null=$server.Handle
    $record.ready=$ready
    $record.private_interpreter=$server.Path
    if(-not $server.Path.StartsWith('C:\Program Files\HopBox Infer QA\python\',[StringComparison]::OrdinalIgnoreCase)){throw 'External interpreter was used'}
    if(-not $launcher.WaitForExit(30000) -or $launcher.ExitCode -ne 0){throw 'CPU splash did not finish successfully'}
    Invoke-RestMethod -Uri 'http://127.0.0.1:8779/api/exit' -Method Post -Body '{}' -ContentType 'application/json' -UseBasicParsing -TimeoutSec 15 | Out-Null
    if(-not $server.WaitForExit(30000) -or $server.ExitCode -ne 0){throw 'CPU server did not shut down cleanly'}
    $record.shutdown_exit=$server.ExitCode
    $record.pass=$true
    $record.phase='complete'
} catch {
    $record.error=$_.Exception.ToString()
    try {Invoke-RestMethod -Uri 'http://127.0.0.1:8779/api/exit' -Method Post -Body '{}' -ContentType 'application/json' -UseBasicParsing -TimeoutSec 5 | Out-Null} catch {}
} finally {
    $record.updated_utc=[DateTime]::UtcNow.ToString('o')
    $record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $results 'startup-options.json') -Encoding UTF8
}
if(-not $record.pass){exit 1}
