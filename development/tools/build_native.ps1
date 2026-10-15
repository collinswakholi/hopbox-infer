# Run from an x64 Visual Studio Developer PowerShell with Rust installed.
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$nativeOutput = Join-Path $projectRoot 'app\native'
New-Item -ItemType Directory -Force -Path $nativeOutput | Out-Null
foreach ($module in @('color_cpu','postprocess','science')) {
    & rustc --edition=2021 --crate-type cdylib -C opt-level=3 -C debuginfo=0 -C panic=abort -C target-cpu=x86-64 (Join-Path $projectRoot ('app\native-src\' + $module + '.rs')) -o (Join-Path $nativeOutput ($module + '.dll'))
    if ($LASTEXITCODE -ne 0) { throw ('Rust build failed: ' + $module) }
}
$workerRoot = Join-Path $projectRoot 'development\native\ccp-worker'
Push-Location -LiteralPath $workerRoot
try {
    & cargo build --release --locked --no-default-features -p ccp_bridge --bin ccp-engine
    if ($LASTEXITCODE -ne 0) { throw 'Rust color worker build failed' }
    Copy-Item -LiteralPath 'target\release\ccp-engine.exe' -Destination (Join-Path $nativeOutput 'ccp-engine.exe')
} finally { Pop-Location }
