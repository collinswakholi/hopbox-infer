# Build the private offline runtime, native kernels and Windows installer.
param()
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = Join-Path $repoRoot '.build-env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create .build-env and install app/requirements.txt first.' }
Set-Location -LiteralPath $repoRoot
$releasePath = [IO.Path]::GetFullPath((Join-Path $repoRoot 'release'))
if (Test-Path -LiteralPath $releasePath) {
    $backupPath = [IO.Path]::GetFullPath((Join-Path $repoRoot ('.archive\release-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))))
    if (-not $releasePath.StartsWith($repoRoot + '\',[StringComparison]::OrdinalIgnoreCase) -or -not $backupPath.StartsWith($repoRoot + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Release paths must stay inside the repository.' }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backupPath) | Out-Null
    Move-Item -LiteralPath $releasePath -Destination $backupPath
}
& (Join-Path $repoRoot 'tools\build_native.ps1')
& $pythonPath tools/build_release.py
exit $LASTEXITCODE
