# Read-only endpoint evidence. Does not run HopBox or change security policy.
$ProgressPreference = 'SilentlyContinue'
$root = 'C:\Program Files\HopBox-Infer'
$events = @(Get-WinEvent -FilterHashtable @{ LogName='Application'; Id=1000,1001; StartTime=(Get-Date).AddHours(-10) } -ErrorAction SilentlyContinue |
    Where-Object { $_.Message -match '(?i)HopBox|pythonw|onnxruntime|DirectML|ccp-engine' } |
    Select-Object -First 16 TimeCreated,Id,Message)
$files = @()
foreach ($relative in @('HopBox-Infer.exe','app\hopbox-code.zip','python\pythonw.exe','python\python312.dll','python\python312._pth','python\msvcp140.dll','python\msvcp140_1.dll','python\vcruntime140.dll','python\vcruntime140_1.dll','python\Lib\site-packages\onnxruntime\capi\onnxruntime_pybind11_state.pyd','python\Lib\site-packages\onnxruntime\capi\DirectML.dll','app\native\ccp-engine.exe','app\native\science.dll')) {
    $path = Join-Path $root $relative
    $files += @{ path=$relative; exists=(Test-Path -LiteralPath $path); sha256=if(Test-Path -LiteralPath $path){(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash}else{$null} }
}
$os = Get-CimInstance Win32_OperatingSystem
$report = [ordered]@{
    hostname=$env:COMPUTERNAME; time=(Get-Date).ToString('o')
    os=@{ name=$os.Caption; version=$os.Version; architecture=$os.OSArchitecture }
    cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
    gpu=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)
    root_exists=(Test-Path -LiteralPath $root); files=$files; crash_events=$events
    cortex_services=@(Get-Service -Name Traps,CyveraService,cyserver -ErrorAction SilentlyContinue | Select-Object Name,Status)
    installation=if(Test-Path "$root\installation.json"){[IO.File]::ReadAllText("$root\installation.json")}else{$null}
    log=if(Test-Path "$env:LOCALAPPDATA\HopBox-Infer\HopBox.log"){[string]::Join("`n",([IO.File]::ReadAllLines("$env:LOCALAPPDATA\HopBox-Infer\HopBox.log") | Select-Object -Last 25))}else{$null}
}
$destination = Join-Path $env:USERPROFILE 'HopBoxSupport'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$report | ConvertTo-Json -Depth 8 -Compress | Set-Content (Join-Path $destination 'remote-diagnosis.json') -Encoding UTF8
Write-Output 'HB_REPORT_BEGIN'
$report | ConvertTo-Json -Depth 8 -Compress
Write-Output 'HB_REPORT_END'
