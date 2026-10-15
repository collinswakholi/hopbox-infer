# Attach to the EXE opened through the freshly installed desktop shortcut.
param([switch]$StartupOnly, [string]$InstallerName = 'HopBox-Infer-Setup-UI-QA.exe', [string]$RecordName = 'interactive-result.json')
$ErrorActionPreference = 'Stop'
$target = 'C:\Approved Apps\HopBox Infer GUI'
$results = 'C:\HopBox-QA\Results'
$record = [ordered]@{ pass = $false; phase = 'checking interactive installation'; target = $target }
function Save-Result { $record | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $results $RecordName) -Encoding UTF8 }
function Api([string]$path, $body = $null) {
    $options = @{ Uri = ('http://127.0.0.1:8765' + $path); TimeoutSec = 120; UseBasicParsing = $true }
    if ($null -ne $body) { $options.Method = 'Post'; $options.ContentType = 'application/json'; $options.Body = ($body | ConvertTo-Json -Depth 8 -Compress) }
    Invoke-RestMethod @options
}
try {
    Save-Result
    $record.installation = Get-Content (Join-Path $target 'installation.json') -Raw | ConvertFrom-Json
    if ($record.installation.destination -ne $target) { throw 'The wizard ignored the custom destination.' }
    $record.installer_sha256 = (Get-FileHash (Join-Path 'C:\HopBox-QA\Package' $InstallerName) -Algorithm SHA256).Hash.ToLowerInvariant()
    $shell = New-Object -ComObject WScript.Shell
    $record.shortcuts = @()
    foreach ($folder in @([Environment]::GetFolderPath('Desktop'),(Join-Path ([Environment]::GetFolderPath('Programs')) 'HopBox Infer'))) {
        $link = $shell.CreateShortcut((Join-Path $folder 'HopBox Infer.lnk'))
        if ($link.TargetPath -ne (Join-Path $target 'HopBox-Infer.exe')) { throw 'Shortcut points to another installation.' }
        $record.shortcuts += [string]$link.TargetPath
    }
    $record.launcher_sha256 = (Get-FileHash (Join-Path $target 'HopBox-Infer.exe')).Hash.ToLowerInvariant()
    $record.app_archive_sha256 = (Get-FileHash (Join-Path $target 'app\hopbox-code.zip')).Hash.ToLowerInvariant()
    if (@(Get-ChildItem (Join-Path $target 'app') -File -Recurse | Where-Object { $_.Extension -in '.py','.cs','.ps1' }).Count) { throw 'Editable application source was installed.' }
    $readyDeadline = [DateTime]::UtcNow.AddMinutes(5)
    do {
        try { $record.ready = Api '/api/ready' } catch { $record.ready = $null }
        if ($record.ready.ok) { break }
        Start-Sleep -Seconds 1
    } while ([DateTime]::UtcNow -lt $readyDeadline)
    if (-not $record.ready.ok) { throw 'The shortcut EXE is not ready.' }
    $server = Get-Process -Id $record.ready.pid
    $null = $server.Handle
    if (-not $server.Path.StartsWith($target,[StringComparison]::OrdinalIgnoreCase)) { throw 'Another installation answered the readiness request.' }
    $record.server_executable = $server.Path
    $record.about = Api '/api/about'
    if (-not $StartupOnly) {
    $record.phase = 'folder run through installed EXE'; Save-Result
    $job = Api '/api/run' @{ folders = @('C:\HopBox-QA\119CANON'); output = (Join-Path $results 'interactive-folder'); device = 'dml:1'; imgsz = 2048; model = 'ccm'; qr_mode = 'qr_ocr'; batch_size = 1 }
    $deadline = [DateTime]::UtcNow.AddMinutes(15)
    do {
        Start-Sleep -Seconds 2
        $run = Api ('/api/jobs/' + $job.id)
        if ($run.status -in 'error','stopped') { throw ($run | ConvertTo-Json -Depth 5) }
    } while ($run.status -ne 'done' -and [DateTime]::UtcNow -lt $deadline)
    if ($run.status -ne 'done') { throw 'Folder processing timed out.' }
    $record.run_log = $run.log
    if (@($run.log | Where-Object { $_ -match '(?i)finding hops on the CPU|GPU startup failed' }).Count) { throw 'GPU processing fell back to CPU.' }
    $csv = Join-Path $results 'interactive-folder\119CANON\Features.csv'
    $record.rows = @(Import-Csv $csv).Count
    if ($record.rows -ne 110) { throw 'Detection count changed.' }
    $record.csv_sha256 = (Get-FileHash $csv).Hash.ToLowerInvariant()
    }
    $record.forbidden_loaded_dlls = @($server.Modules | Where-Object { $_.FileName -match '(?i)(torch|c10|ultralytics)' } | ForEach-Object { $_.FileName })
    if ($record.forbidden_loaded_dlls.Count) { throw 'A forbidden runtime was loaded.' }
    # Closing the browser/server may close a pooled HTTP connection first.
    # Only the observed process exit code establishes successful shutdown.
    try { Api '/api/exit' @{} | Out-Null } catch [System.Net.WebException] { $record.exit_http_disconnect = $_.Exception.Message }
    if (-not $server.WaitForExit(30000) -or $server.ExitCode -ne 0) { throw 'Installed EXE did not exit cleanly.' }
    $record.shutdown_exit = $server.ExitCode
    $record.phase = 'complete'; $record.pass = $true
} catch { $record.error = $_.Exception.ToString() } finally { Save-Result }
