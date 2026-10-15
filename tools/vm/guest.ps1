# Runs inside a fresh offline Windows Sandbox, with no development environment.
param([ValidateSet('CPU','GPU')][string]$Mode = 'CPU', [switch]$Interactive, [ValidatePattern('^(1|2|4|8)(,(1|2|4|8))*$')][string]$Batches = '8,4,2,1')
$ErrorActionPreference = 'Stop'
$resultRoot = 'C:\HopBox-QA\Results'
$controlRoot = 'C:\HopBox-QA\Control'
$targetRoot = 'C:\Program Files\HopBox Infer QA'
$qaRecord = [ordered]@{ pass = $false; phase = 'starting'; mode = $Mode; started_utc = [DateTime]::UtcNow.ToString('o') }
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
function Save-Status {
    $qaRecord.updated_utc = [DateTime]::UtcNow.ToString('o')
    # A host reader can hold a mapped-file handle while the guest reports progress.
    # Publish complete JSON by replacement; retry short sharing conflicts instead
    # of treating a test-report write as an application failure.
    $statusPath = Join-Path $resultRoot 'vm-result.json'
    $temporary = Join-Path $resultRoot ('status-'+[Guid]::NewGuid().ToString('N')+'.tmp')
    [IO.File]::WriteAllText($temporary,($qaRecord | ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    try {
        for($attempt=0; $attempt -lt 50; $attempt++) {
            try {
                if([IO.File]::Exists($statusPath)) { [IO.File]::Replace($temporary,$statusPath,[NullString]::Value) }
                else { [IO.File]::Move($temporary,$statusPath) }
                return
            } catch [IO.IOException] {
                if($attempt -eq 49) { throw }
                Start-Sleep -Milliseconds 100
            }
        }
    } finally {
        if([IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) }
    }
}
function Request-Api([string]$path, $body = $null) {
    $arguments = @{ Uri = ('http://127.0.0.1:8765' + $path); TimeoutSec = 120; UseBasicParsing = $true }
    if ($null -ne $body) { $arguments.Method = 'Post'; $arguments.ContentType = 'application/json'; $arguments.Body = ($body | ConvertTo-Json -Depth 10 -Compress) }
    Invoke-RestMethod @arguments
}
function Wait-Job($body, [string]$endpoint) {
    $jobToken = (Request-Api $endpoint $body).id
    $jobDeadline = [DateTime]::UtcNow.AddMinutes(20)
    $nextProgressSave = [DateTime]::UtcNow
    do {
        Start-Sleep -Seconds 1
        $jobResult = Request-Api ('/api/jobs/' + $jobToken)
        if ($jobResult.status -eq 'done') { return $jobResult }
        if ($jobResult.status -in @('error','stopped')) { throw ($jobResult | ConvertTo-Json -Depth 10) }
        if ([DateTime]::UtcNow -ge $nextProgressSave) {
            $qaRecord.current_job = @{ id = $jobToken; status = $jobResult.status; progress = $jobResult.progress; recent_log = @($jobResult.log | Select-Object -Last 4) }
            $runningServer = Get-Process -Id $serverPid
            $qaRecord.peak_server_working_set_bytes = [Math]::Max($qaRecord.peak_server_working_set_bytes,$runningServer.PeakWorkingSet64)
            Save-Status
            $nextProgressSave = [DateTime]::UtcNow.AddSeconds(10)
        }
    } while ([DateTime]::UtcNow -lt $jobDeadline)
    throw 'Guest processing job timed out.'
}
try {
    $qaRecord.os = (Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture)
    $qaRecord.computer = $env:COMPUTERNAME
    $qaRecord.python_before_install = @(Get-Command python.exe,python3.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
    $qaRecord.phase = 'installing'; Save-Status
    $packageBranding = Get-Content -LiteralPath 'C:\HopBox-QA\Package\branding.json' -Raw | ConvertFrom-Json
    $setupPath = 'C:\HopBox-QA\Package\HopBox-Infer-Setup-' + $packageBranding.version + '.exe'
    # Exercise the actual wizard in the guest; host UI automation chooses the
    # destination/components and clicks Install/Finish. Quiet upgrade follows.
    $qaRecord.installer_exits = @()
    $expectedComponents = 'core'
    $qaRecord.phase = if ($Interactive) { 'waiting for interactive installer' } else { 'installing custom destination' }; Save-Status
    $firstArguments = if ($Interactive) { '--no-launch' } else { '--no-launch --quiet --install-dir "' + $targetRoot + '" --components ' + $expectedComponents }
    $setupProcess = Start-Process -FilePath $setupPath -ArgumentList $firstArguments -PassThru
    if (-not $setupProcess.WaitForExit(900000)) { throw 'Installer did not finish within fifteen minutes.' }
    $qaRecord.installer_exits += $setupProcess.ExitCode
    if ($setupProcess.ExitCode -ne 0) { throw 'Interactive installer failed or was cancelled.' }
    if (-not (Test-Path -LiteralPath (Join-Path $targetRoot 'installation.json'))) { throw 'The wizard did not use the selected custom destination.' }
    $installed = Get-Content -LiteralPath (Join-Path $targetRoot 'installation.json') -Raw | ConvertFrom-Json
    if ($installed.components -ne $expectedComponents) { throw 'The wizard installed different components from the UI selection.' }
    $qaRecord.wizard_installation = $installed
    $qaRecord.installer_mode = if ($Interactive) { 'interactive wizard' } else { 'quiet command line with explicit destination/components' }
    $qaRecord.phase = 'quiet upgrade'; Save-Status
    foreach ($attempt in 1..1) {
        $setupProcess = Start-Process -FilePath $setupPath -ArgumentList ('--no-launch --quiet --install-dir "' + $targetRoot + '" --components ' + $expectedComponents) -WindowStyle Hidden -Wait -PassThru
        $qaRecord.installer_exits += $setupProcess.ExitCode
        if ($setupProcess.ExitCode -ne 0) { throw (Get-Content -LiteralPath ($targetRoot + '.installation-error.txt') -Raw) }
    }
    $qaRecord.phase = 'launching'; Save-Status
    $startupTimer = [Diagnostics.Stopwatch]::StartNew()
    $env:PATH = Join-Path $env:SystemRoot 'System32'
    Remove-Item Env:PYTHONPATH,Env:HOPBOX_EXPORTER_PYTHON,Env:HOPBOX_EXPORTER_SCRIPT -ErrorAction SilentlyContinue
    $env:PYTHONNOUSERSITE = '1'
    Remove-Item Env:HOPBOX_NO_BROWSER -ErrorAction SilentlyContinue
    $env:HOPBOX_SKIP_INSTANCE_STOP = '1'
    $env:HOPBOX_STATE_ROOT = Join-Path $env:LOCALAPPDATA 'HopBox-VM-QA'
    $env:HOPBOX_STARTUP_TRACE = Join-Path $resultRoot 'startup.txt'
    New-Item -ItemType Directory -Force -Path $env:HOPBOX_STATE_ROOT | Out-Null
    # Deny installation writes even to the guest user. Running the EXE must
    # keep every mutable file in the data/output directory instead.
    # Apply inheritable rules to the root only. Removing inheritance with /T
    # would leave leaf files with empty ACLs and prevent even execution.
    & "$env:SystemRoot\System32\icacls.exe" $targetRoot /inheritance:r /grant:r ($env:USERNAME + ':(OI)(CI)RX') 'SYSTEM:(OI)(CI)F' /Q | Out-File (Join-Path $resultRoot 'readonly-acl.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Could not make the installation read-only.' }
    $writeDenied = $false
    try { [IO.File]::WriteAllText((Join-Path $targetRoot 'write-probe.txt'),'must fail') } catch [UnauthorizedAccessException] { $writeDenied = $true }
    if (-not $writeDenied) { throw 'Installation write protection was ineffective.' }
    $qaRecord.installation_readonly = $true
    $sourceFiles = @(Get-ChildItem -LiteralPath (Join-Path $targetRoot 'app') -Recurse -File | Where-Object { $_.Extension -in @('.py','.cs','.ps1') })
    if ($sourceFiles.Count) { throw 'Editable application/build source was deployed.' }
    $qaRecord.editable_application_sources = 0
    Start-Process -FilePath (Join-Path $targetRoot 'HopBox-Infer.exe') -WindowStyle Hidden
    $readyDeadline = [DateTime]::UtcNow.AddMinutes(5)
    do {
        Start-Sleep -Seconds 1
        try { $ready = Request-Api '/api/ready' } catch { $ready = $null }
    } while (-not $ready.ok -and [DateTime]::UtcNow -lt $readyDeadline)
    if (-not $ready.ok) { throw 'Guest EXE failed to become ready.' }
    $startupTimer.Stop()
    $qaRecord.first_ready_seconds = $startupTimer.Elapsed.TotalSeconds
    $qaRecord.ready = $ready
    $qaRecord.branding = Request-Api '/api/about'
    if ($qaRecord.branding.version -ne $packageBranding.version) { throw 'Application version mismatch.' }
    Start-Sleep -Seconds 2
    # Get-Content adds provider metadata to strings in Windows PowerShell.
    # Serializing those decorated strings can recurse through PSDrive objects.
    $qaRecord.startup_trace = [IO.File]::ReadAllLines($env:HOPBOX_STARTUP_TRACE)
    if (-not ($qaRecord.startup_trace -match '^splash-shown ') -or -not ($qaRecord.startup_trace -match '^splash-closed ')) { throw 'Native splash lifecycle did not complete.' }
    $serverPid = [int](@(Get-Content -LiteralPath (Join-Path $env:HOPBOX_STATE_ROOT 'servers.txt'))[-1])
    $serverProcess = Get-Process -Id $serverPid
    # Keep a query handle open so the actual exit code remains readable.
    $null = $serverProcess.Handle
    if (-not $serverProcess.Path.StartsWith($targetRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Guest used an external interpreter.' }
    $qaRecord.server_executable = $serverProcess.Path
    $qaRecord.launched_exe = Join-Path $targetRoot 'HopBox-Infer.exe'
    $qaRecord.peak_server_working_set_bytes = $serverProcess.PeakWorkingSet64
    $qaRecord.ui_http_status = (Invoke-WebRequest -Uri 'http://127.0.0.1:8765/' -UseBasicParsing -TimeoutSec 30).StatusCode
    if ($qaRecord.ui_http_status -ne 200) { throw 'Guest UI page did not load.' }
    $qaRecord.browser_processes = @(Get-Process msedge,chrome -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path -Unique)
    $photoFolder = 'C:\HopBox-QA\119CANON'
    Request-Api '/api/import' @{ path = $photoFolder } | Out-Null
    $testDevice = 'cpu'
    if ($Mode -eq 'GPU') {
        $qaRecord.phase = 'GPU adapter discovery and inference'; Save-Status
        $catalog = Request-Api '/api/vram?device=auto'
        $qaRecord.device_catalog = $catalog
        $qaRecord.gpu_probe_failures = @()
        $gpuCandidates = @($catalog.devices | Where-Object { $_.enabled -and $_.id -like 'dml:*' })
        $selectedGpu = $null
        foreach ($candidate in $gpuCandidates) {
            try {
                $selection = Request-Api '/api/backend' @{ device = $candidate.id; imgsz = 2048 }
                $probe = Wait-Job @{ image = (Join-Path $photoFolder 'IMG_0002.JPG'); mode = 'detections'; output = $resultRoot; device = $candidate.id; imgsz = 2048; model = 'ccm' } '/api/preview'
                if (-not $probe.previews -or @($probe.log | Where-Object { $_ -match '(?i)finding hops on the CPU|GPU startup failed' }).Count) { throw 'GPU preview did not complete on the selected accelerator.' }
                $selectedGpu = $candidate
                $qaRecord.gpu_probe = @{ selection = $selection; candidate = $candidate; log = $probe.log }
                break
            } catch {
                $qaRecord.gpu_probe_failures += @{ candidate = $candidate; error = $_.Exception.ToString() }
            }
        }
        if ($null -eq $selectedGpu) { throw 'No exposed DirectML GPU completed EXE inference. CPU fallback does not pass this GPU test.' }
        $testDevice = $selectedGpu.id
        $qaRecord.gpu_evidence = 'Explicit DirectML device requests; the deployed OnnxBackend rejects sessions whose actual first provider is not DmlExecutionProvider. Loaded DLLs and CPU fallback logs are checked separately.'
    }
    $qaRecord.processing_device = $testDevice
    $qaRecord.batches = @()
    # Exercise the memory-sensitive batches first, then verify every smaller size.
    foreach ($batchSize in @($Batches.Split(',') | ForEach-Object { [int]$_ })) {
        $qaRecord.phase = ($testDevice+' Run Folders batch '+$batchSize); Save-Status
        $batchOutput = Join-Path $resultRoot ('batch-'+$batchSize)
        New-Item -ItemType Directory -Force -Path $batchOutput | Out-Null
        $batchTimer = [Diagnostics.Stopwatch]::StartNew()
        $run = Wait-Job @{ folders = @($photoFolder); output = $batchOutput; device = $testDevice; imgsz = 2048; model = 'ccm'; qr_mode = 'qr_ocr'; batch_size = $batchSize } '/api/run'
        if ($Mode -eq 'GPU' -and @($run.log | Where-Object { $_ -match '(?i)finding hops on the CPU|GPU startup failed' }).Count) { throw 'GPU folder run fell back to CPU.' }
        $batchTimer.Stop()
        $csvPath = Join-Path $batchOutput '119CANON\Features.csv'
        $table = @(Import-Csv -LiteralPath $csvPath)
        if ($table.Count -ne 110 -or @($table.Image_name | Select-Object -Unique).Count -ne 11) { throw 'Guest CSV count changed.' }
        & (Join-Path $controlRoot 'audit_guest.ps1') -Target $targetRoot -ServerPid $serverPid -Csv $csvPath -Reference (Join-Path $controlRoot 'reference.csv') -Output $batchOutput -Device $testDevice
        $serverProcess.Refresh()
        $memory = @{ working_set_bytes = [long]$serverProcess.WorkingSet64; private_bytes = [long]$serverProcess.PrivateMemorySize64; peak_working_set_bytes = [long]$serverProcess.PeakWorkingSet64 }
        $qaRecord.batches += @{ batch = $batchSize; seconds = $batchTimer.Elapsed.TotalSeconds; rows = $table.Count; images = 11; memory_after = $memory; audit = Get-Content -LiteralPath (Join-Path $batchOutput 'guest-audit.json') -Raw | ConvertFrom-Json; log = $run.log }
        Save-Status
    }
    # One clearly scoped CPU smoke after a GPU run also proves the standard
    # component switches devices without replacing the installation.
    if ($Mode -eq 'GPU' -and $Batches -eq '1') {
        $qaRecord.phase = 'CPU smoke after DirectML'; Save-Status
        $cpuOutput = Join-Path $resultRoot 'cpu-smoke'
        $cpuTimer = [Diagnostics.Stopwatch]::StartNew()
        $cpuRun = Wait-Job @{ folders = @($photoFolder); output = $cpuOutput; device = 'cpu'; imgsz = 2048; model = 'ccm'; qr_mode = 'qr_ocr'; batch_size = 1 } '/api/run'
        $cpuTimer.Stop()
        $cpuCsv = Join-Path $cpuOutput '119CANON\Features.csv'
        if (@(Import-Csv -LiteralPath $cpuCsv).Count -ne 110) { throw 'Final CPU smoke lost detections.' }
        & (Join-Path $controlRoot 'audit_guest.ps1') -Target $targetRoot -ServerPid $serverPid -Csv $cpuCsv -Reference (Join-Path $controlRoot 'reference.csv') -Output $cpuOutput -Device 'cpu'
        $qaRecord.cpu_smoke = @{ pass = $true; rows = 110; images = 11; seconds = $cpuTimer.Elapsed.TotalSeconds }
        Save-Status
    }
    $qaRecord.phase = 'custom ONNX model and cancellation'; Save-Status
    $customRoot = Join-Path $env:HOPBOX_STATE_ROOT 'models\vm-tuber'
    New-Item -ItemType Directory -Force -Path $customRoot | Out-Null
    Copy-Item -LiteralPath (Join-Path $controlRoot 'best_s_1024_fp32.onnx'),(Join-Path $controlRoot 'best_s_1024_fp32.json') -Destination $customRoot
    $customGraph = Join-Path $customRoot 'best_s_1024_fp32.onnx'
    $model = Request-Api '/api/model' @{ path = $customGraph; device = $testDevice; imgsz = 1024 }
    if ($model.count -ne 1) { throw 'Guest custom class discovery failed.' }
    if ($model.needs_map) {
        $cancelled = Request-Api '/api/model/cancel' @{}
        if (-not $cancelled.default) { throw 'Guest mapping cancel failed.' }
        Request-Api '/api/model' @{ path = $customGraph; device = $testDevice; imgsz = 1024 } | Out-Null
    }
    $mapped = Request-Api '/api/model/map' @{ rows = @(@{ index = 0; source = 'tuber'; display = 'Tuber'; role = 'measure' }); device = $testDevice; imgsz = 1024 }
    if (-not $mapped.ready) { throw 'Guest model mapping failed.' }
    $sample = Join-Path $controlRoot 'frame_000038.jpg'
    Request-Api '/api/import' @{ path = $controlRoot } | Out-Null
    $preview = Wait-Job @{ image = $sample; mode = 'detections'; output = $resultRoot; device = $testDevice; imgsz = 1024; model = 'ccm' } '/api/preview'
    if (-not $preview.previews) { throw 'Guest custom-model preview failed.' }
    $qaRecord.custom_model_pass = $true
    if (Test-Path -LiteralPath (Join-Path $targetRoot 'gpu\rtx')) { throw 'Unvalidated RTX runtime was installed.' }
    $restored = Request-Api '/api/model' @{ path = (Join-Path $targetRoot 'app\models\portable\hops_fp32.onnx'); device = $testDevice; imgsz = 2048 }
    if (-not $restored.default) { throw 'Guest default-model restore failed.' }
    $qaRecord.phase = 'shutdown'; Save-Status
    Request-Api '/api/exit' @{} | Out-Null
    if (-not $serverProcess.WaitForExit(30000)) { throw 'Guest shutdown timed out.' }
    $qaRecord.shutdown_exit = $serverProcess.ExitCode
    if ($serverProcess.ExitCode -ne 0) { throw 'Guest shutdown failed.' }
    $qaRecord.pass = $true
    $qaRecord.Remove('current_job')
    $qaRecord.phase = 'complete'; Save-Status
} catch {
    $qaRecord.error = $_.Exception.ToString()
    $qaRecord.error_location = $_.InvocationInfo.PositionMessage
    $qaRecord.script_stack = $_.ScriptStackTrace
    $qaRecord.phase = 'failed'; Save-Status
    try { Request-Api '/api/exit' @{} | Out-Null } catch { }
    exit 1
}
