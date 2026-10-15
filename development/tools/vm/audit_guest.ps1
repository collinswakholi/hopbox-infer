param([string]$Target,[int]$ServerPid,[string]$Csv,[string]$Reference,[string]$Output,[string]$Device = 'cpu')
$ErrorActionPreference = 'Stop'
$server = Get-Process -Id $ServerPid
if (-not $server.Path.StartsWith($Target,[StringComparison]::OrdinalIgnoreCase)) { throw 'Guest server used an external runtime.' }
$packages = @{}
Get-ChildItem -LiteralPath (Join-Path $Target 'python\Lib\site-packages') -Directory -Filter '*.dist-info' | ForEach-Object {
    $metadata = [IO.File]::ReadAllLines((Join-Path $_.FullName 'METADATA'))
    $name = (($metadata | Where-Object { $_.StartsWith('Name: ') } | Select-Object -First 1).Substring(6)).ToLower().Replace('-','_')
    $version = ($metadata | Where-Object { $_.StartsWith('Version: ') } | Select-Object -First 1).Substring(9)
    $packages[$name] = $version
}
foreach ($name in @('torch','torchvision','torchaudio','ultralytics','easyocr','scipy','scikit_image','numba','llvmlite','taichi')) {
    if ($packages.ContainsKey($name)) { throw ('Guest has forbidden package '+$name) }
}
$libraries = @($server.Modules | ForEach-Object { [string]$_.FileName })
$bad = @($libraries | Where-Object { $_ -match '(?i)torch|ultralytics|libtorch|c10\.dll|skimage|taichi|numba|llvmlite|[\\/]scipy([\\/]|\.libs[\\/])' })
if ($bad.Count) { throw ('Guest loaded forbidden libraries: '+($bad -join ', ')) }
if ($Device -like 'dml:*' -and -not @($libraries | Where-Object { [IO.Path]::GetFileName($_) -ieq 'DirectML.dll' }).Count) { throw 'The GPU EXE process did not load DirectML.dll.' }
$cppLibraries = @($libraries | Where-Object { [IO.Path]::GetFileName($_) -match '(?i)^msvcp140(_1)?\.dll$' })
if ($cppLibraries.Count -lt 2 -or @($cppLibraries | Where-Object { -not $_.StartsWith($Target,[StringComparison]::OrdinalIgnoreCase) }).Count) { throw 'Guest did not load its bundled C++ runtime.' }
$actual = @(Import-Csv -LiteralPath $Csv)
$baseline = @(Import-Csv -LiteralPath $Reference)
$columns = @($baseline[0].PSObject.Properties.Name)
if ($actual.Count -ne $baseline.Count -or ($columns -join '|') -ne (@($actual[0].PSObject.Properties.Name) -join '|')) { throw 'Guest CSV structure changed.' }
$traits = @()
foreach ($column in $columns) {
    if ($column -in @('Image_name','QR_info','CC_model')) {
        for ($row=0;$row -lt $actual.Count;$row++) { if ($actual[$row].$column -cne $baseline[$row].$column) { throw ('Guest identifier changed: '+$column) } }
        continue
    }
    $maxAbs = 0.0; $maxRelative = 0.0; $exact = $true
    for ($row=0;$row -lt $actual.Count;$row++) {
        $aText = [string]$baseline[$row].$column; $bText = [string]$actual[$row].$column
        if ([string]::IsNullOrWhiteSpace($aText) -or [string]::IsNullOrWhiteSpace($bText)) {
            if ($aText -ne $bText) { throw ('Guest missing values changed: '+$column) }
            continue
        }
        $a = [double]::Parse($aText,[Globalization.CultureInfo]::InvariantCulture)
        $b = [double]::Parse($bText,[Globalization.CultureInfo]::InvariantCulture)
        $difference = [Math]::Abs($a-$b)
        $maxAbs = [Math]::Max($maxAbs,$difference)
        $maxRelative = [Math]::Max($maxRelative,$difference/[Math]::Max([Math]::Abs($a),1e-12))
        if ($difference -ne 0) { $exact = $false }
    }
    $traits += @{ column = $column; max_absolute_difference = $maxAbs; max_relative_difference = $maxRelative; exact = $exact }
    if ($column -eq 'ID' -and -not $exact) { throw 'Guest object identities changed.' }
    if ($column -eq 'RP_Area' -and $maxRelative -gt .001) { throw 'Guest area parity gate failed.' }
}
@{ pass = $true; packages = $packages; server = $server.Path; libraries = $libraries; forbidden_libraries = $bad; bundled_cpp_libraries = $cppLibraries; identifiers_match = $true; traits = $traits; requested_device = $Device; audit_method = 'PowerShell inspection of the installed EXE process and package'; note = 'Guest versus NVIDIA host reference; every field reported, area gate 0.1%. A virtual GPU test does not certify other physical GPU models.' } | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $Output 'guest-audit.json') -Encoding UTF8
