param(
    [Parameter(Mandatory=$true)][string]$Installer,
    [Parameter(Mandatory=$true)][string]$Destination,
    [Parameter(Mandatory=$true)][string]$Results,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InstallerSha256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$InventorySha256,
    [string]$InputFolder,
    [int]$Port=8794,
    [switch]$RunApp
)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$started=Get-Date
$destinationPath=[IO.Path]::GetFullPath($Destination).TrimEnd('\')
$original='C:\Program Files\HopBox-Infer'
if($destinationPath -eq $original){throw 'Use a separate test installation; preserve the original app.'}
if(Test-Path -LiteralPath $destinationPath){throw 'Fresh-test destination already exists; no files changed.'}
New-Item -ItemType Directory -Path $Results -Force | Out-Null
$report=[ordered]@{pass=$false;install_pass=$false;runtime_pass=$false;phase='preflight';started_utc=[DateTime]::UtcNow.ToString('o');hostname=$env:COMPUTERNAME;destination=$destinationPath;security_settings_changed=$false}
$server=$null
function Save-Report {
    $report.updated_utc=[DateTime]::UtcNow.ToString('o')
    $path=Join-Path $Results 'result.json'
    $temporary=Join-Path $Results ('report-'+[Guid]::NewGuid().ToString('N')+'.tmp')
    [IO.File]::WriteAllText($temporary,($report | ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    try {
        for($attempt=0;$attempt -lt 50;$attempt++) {
            try {
                if([IO.File]::Exists($path)){[IO.File]::Replace($temporary,$path,[NullString]::Value)}
                else {[IO.File]::Move($temporary,$path)}
                return
            } catch [IO.IOException] {if($attempt -eq 49){throw};Start-Sleep -Milliseconds 100}
        }
    } finally {if([IO.File]::Exists($temporary)){[IO.File]::Delete($temporary)}}
}
function Current-Prevention {
    @(Get-WinEvent -FilterHashtable @{LogName='Palo Alto Networks';Id=85;StartTime=$started} -ErrorAction SilentlyContinue | Where-Object {
        $xml=$_.ToXml()
        $xml.IndexOf($InstallerSha256,[StringComparison]::OrdinalIgnoreCase) -ge 0 -or
        $xml.IndexOf($destinationPath,[StringComparison]::OrdinalIgnoreCase) -ge 0
    } | ForEach-Object { @{utc=$_.TimeCreated.ToUniversalTime().ToString('o');id=$_.Id;message=$_.Message;xml=$_.ToXml()} })
}
function Check-Prevention {
    $events=@(Current-Prevention)
    if($events.Count){$report.security_blocked=$true;$report.prevention_events=$events;throw 'Endpoint protection blocked this test. No alternate execution or policy changes attempted.'}
}
function Verify-Files {
    $manifest=Join-Path $destinationPath 'package-files.json'
    if((Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne $InventorySha256){throw 'Installed inventory differs from the verified release'}
    $inventory=Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
    $failures=@();$count=0
    foreach($file in $inventory.files) {
        $path=[IO.Path]::GetFullPath((Join-Path $destinationPath $file.path))
        if(-not $path.StartsWith($destinationPath+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Unsafe inventory path'}
        if(-not(Test-Path -LiteralPath $path)){$failures+=('Missing: '+$file.path)}
        elseif((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256){$failures+=('Changed: '+$file.path)}
        $count++
    }
    if($failures.Count){$report.integrity_errors=$failures;Check-Prevention;throw 'Installed-file integrity failed'}
    return $count
}
function Api([string]$path,$body=$null) {
    $request=[Net.HttpWebRequest]::Create('http://127.0.0.1:'+$Port+$path)
    $request.Proxy=$null;$request.Timeout=15000
    if($null -ne $body){$request.Method='POST';$request.ContentType='application/json';$writer=[IO.StreamWriter]::new($request.GetRequestStream());try{$writer.Write(($body | ConvertTo-Json -Depth 8 -Compress))}finally{$writer.Dispose()}}
    $response=$request.GetResponse();$reader=[IO.StreamReader]::new($response.GetResponseStream())
    try {return ($reader.ReadToEnd() | ConvertFrom-Json)} finally {$reader.Dispose();$response.Dispose()}
}
try {
    $originalExe=Join-Path $original 'HopBox-Infer.exe'
    $before=if(Test-Path -LiteralPath $originalExe){(Get-FileHash -LiteralPath $originalExe -Algorithm SHA256).Hash}else{$null}
    $report.original_launcher_sha256=$before
    $actual=(Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
    if($actual -ne $InstallerSha256){throw 'Installer checksum mismatch; nothing executed'}
    $report.installer_sha256=$actual
    $report.installer_signature=(Get-AuthenticodeSignature -LiteralPath $Installer).Status.ToString()
    $report.installations=@()
    foreach($phase in @('fresh','upgrade')) {
        $report.phase=$phase+' installation';Save-Report;Check-Prevention
        $watch=[Diagnostics.Stopwatch]::StartNew()
        $process=Start-Process -FilePath $Installer -ArgumentList ('--quiet --no-launch --components core --install-dir "'+$destinationPath+'"') -WindowStyle Hidden -PassThru
        if(-not $process.WaitForExit(240000)){throw 'Installer did not finish within four minutes'}
        if($process.ExitCode -ne 0){$report.installer_exit=$process.ExitCode;Check-Prevention;throw ('Installer exited with '+$process.ExitCode)}
        $report.phase=$phase+' file verification';Save-Report
        $count=Verify-Files
        $report.installations+=@{phase=$phase;exit=0;verified_files=$count;every_sha256_matches=$true;seconds=$watch.Elapsed.TotalSeconds}
        Check-Prevention;Save-Report
    }
    $report.install_pass=$true
    $installation=Get-Content -LiteralPath (Join-Path $destinationPath 'installation.json') -Raw | ConvertFrom-Json
    if($installation.destination -ne $destinationPath -or $installation.components -ne 'core'){throw 'Installation destination/components mismatch'}
    $report.installation=$installation
    $exe=Join-Path $destinationPath 'HopBox-Infer.exe'
    $report.launcher_sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    if($RunApp) {
        $report.phase='installed EXE diagnostics';Save-Report
        $integrity=Join-Path $Results 'installed-integrity.json'
        $process=Start-Process -FilePath $exe -ArgumentList ('--diagnose "'+$integrity+'"') -WindowStyle Hidden -PassThru
        if(-not $process.WaitForExit(90000)){throw 'Installed EXE diagnostic timed out'}
        $report.diagnostic_exit=$process.ExitCode
        Check-Prevention
        if($process.ExitCode -ne 0 -or -not(Test-Path -LiteralPath $integrity)){throw 'Installed EXE did not produce a successful diagnostic report'}
        if(-not(Get-Content -LiteralPath $integrity -Raw | ConvertFrom-Json).integrity_pass){throw 'Installed EXE diagnostic found damaged files'}
        $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$Port)
        try {$listener.Start()}finally{$listener.Stop()}
        $env:HOPBOX_PORT=[string]$Port;$env:HOPBOX_STATE_ROOT=Join-Path $Results 'state';$env:HOPBOX_NO_BROWSER='1'
        $env:HOPBOX_DEVICE='auto';$env:HOPBOX_STARTUP_TRACE=Join-Path $Results 'startup.txt'
        Remove-Item Env:HOPBOX_DML_GENERIC,Env:PYTHONPATH -ErrorAction SilentlyContinue
        $report.phase='installed EXE startup';Save-Report
        $process=Start-Process -FilePath $exe -WindowStyle Hidden -PassThru
        $deadline=[DateTime]::UtcNow.AddMinutes(3)
        do {
            Start-Sleep -Seconds 1
            try {$ready=Api '/api/ready'}catch{$ready=$null}
            if($process.HasExited -and $process.ExitCode -ne 0){$report.launcher_exit=$process.ExitCode;Check-Prevention;throw 'Installed launcher stopped before readiness'}
        } while(-not $ready.ok -and [DateTime]::UtcNow -lt $deadline)
        Check-Prevention
        if(-not $ready.ok){throw 'Installed EXE failed to become ready'}
        $server=Get-Process -Id $ready.pid;$null=$server.Handle
        if($server.Path -ne (Join-Path $destinationPath 'python\pythonw.exe')){throw 'The installed EXE used an unexpected interpreter'}
        $report.ready=$ready;$report.private_interpreter=$server.Path
        $report.startup_trace=[IO.File]::ReadAllLines($env:HOPBOX_STARTUP_TRACE)
        if($InputFolder) {
            $report.phase='installed EXE folder processing';Save-Report
            Api '/api/import' @{path=$InputFolder} | Out-Null
            $token=(Api '/api/run' @{folders=@($InputFolder);output=(Join-Path $Results 'output');device='auto';imgsz=2048;model='ccm';qr_mode='qr_ocr';batch_size=1}).id
            $deadline=[DateTime]::UtcNow.AddMinutes(15)
            do {
                Start-Sleep -Seconds 2
                $job=Api ('/api/jobs/'+$token)
                if($job.status -in @('error','stopped')){throw ($job | ConvertTo-Json -Depth 6)}
                $report.current_job=@{status=$job.status;progress=$job.progress;recent_log=@($job.log | Select-Object -Last 3)};Save-Report
            } while($job.status -ne 'done' -and [DateTime]::UtcNow -lt $deadline)
            if($job.status -ne 'done'){throw 'Folder processing timed out'}
            $csv=Join-Path (Join-Path (Join-Path $Results 'output') (Split-Path $InputFolder -Leaf)) 'Features.csv'
            $rows=@(Import-Csv -LiteralPath $csv)
            if(-not $rows.Count){throw 'Folder run produced no measurements'}
            $report.folder=@{path=$InputFolder;rows=$rows.Count;images=@($rows.Image_name | Select-Object -Unique).Count;csv_sha256=(Get-FileHash -LiteralPath $csv -Algorithm SHA256).Hash.ToLowerInvariant();accuracy_against_manual_ground_truth='not assessed'}
        }
        Api '/api/exit' @{} | Out-Null
        if(-not $server.WaitForExit(30000) -or $server.ExitCode -ne 0){throw 'Installed EXE did not shut down cleanly'}
        $report.shutdown_exit=$server.ExitCode;$report.runtime_pass=$true
    }
    $report.pass=$true;$report.phase='complete'
} catch {
    $report.error=$_.Exception.ToString();$report.error_location=$_.InvocationInfo.PositionMessage;$report.phase='failed'
    $report.prevention_events=@(Current-Prevention)
    if($report.prevention_events.Count){$report.security_blocked=$true}
} finally {
    if($server -and -not $server.HasExited){try {Api '/api/exit' @{} | Out-Null;$null=$server.WaitForExit(30000)}catch{}}
    $originalExe=Join-Path $original 'HopBox-Infer.exe'
    $after=if(Test-Path -LiteralPath $originalExe){(Get-FileHash -LiteralPath $originalExe -Algorithm SHA256).Hash}else{$null}
    $report.original_launcher_unchanged=($before -eq $after)
    Save-Report
}
Write-Output ('HB_INSTALL_RESULT '+([pscustomobject]$report | Select-Object pass,phase,install_pass,runtime_pass,security_blocked,diagnostic_exit | ConvertTo-Json -Compress))
if(-not $report.pass){exit 1}
