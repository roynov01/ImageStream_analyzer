Param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = 'C:/Users/royno/.conda/envs/imagestream/python.exe'
$distDir = Join-Path $projectRoot 'dist'
$buildDir = Join-Path $projectRoot 'build'
$specFile = Join-Path $projectRoot 'image_stream_analysis.spec'

if (-not (Test-Path $python)) {
    throw "Python executable not found: $python"
}

& $python -m pip show pyinstaller | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing PyInstaller into the imagestream conda environment...'
    & $python -m pip install pyinstaller
}

if ($Clean) {
    if (Test-Path $distDir) { Remove-Item $distDir -Recurse -Force }
    if (Test-Path $buildDir) { Remove-Item $buildDir -Recurse -Force }
    if (Test-Path $specFile) { Remove-Item $specFile -Force }
}

New-Item -ItemType Directory -Path $distDir -Force | Out-Null

$pyArgs = @(
    '--noconfirm',
    '--clean',
    '--name', 'image_stream_analysis',
    '--windowed',
    '--onedir',
    '--distpath', $distDir,
    '--workpath', $buildDir,
    '--specpath', $projectRoot,
    '--add-data', 'data;data'
)

$envRoot = Split-Path -Parent $python
$libBin = Join-Path $envRoot 'Library\bin'
$dllPatterns = @(
    'ffi*.dll',
    'libexpat*.dll',
    'sqlite3.dll',
    'libcrypto*.dll',
    'libssl*.dll',
    'liblzma*.dll',
    'libbz2*.dll',
    'tcl*.dll',
    'tk*.dll'
)

if (Test-Path $libBin) {
    foreach ($pattern in $dllPatterns) {
        Get-ChildItem -Path $libBin -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
            $pyArgs += '--add-binary'
            $pyArgs += ("{0};." -f $_.FullName)
        }
    }
}

# Ensure Tcl/Tk data directories are collected and mapped to the names PyInstaller expects
$tclCandidates = @((Join-Path $envRoot 'Library\lib\tcl8.6'), (Join-Path $envRoot 'Library\lib\tcl8'))
$tkCandidates = @((Join-Path $envRoot 'Library\lib\tk8.6'), (Join-Path $envRoot 'Library\lib\tk8'))

foreach ($tclPath in $tclCandidates) {
    if (Test-Path $tclPath) {
        $pyArgs += '--add-data'
        $pyArgs += ("{0};_tcl_data" -f $tclPath)
        break
    }
}

foreach ($tkPath in $tkCandidates) {
    if (Test-Path $tkPath) {
        $pyArgs += '--add-data'
        $pyArgs += ("{0};_tk_data" -f $tkPath)
        break
    }
}

$pyArgs += 'run_gui.py'

# Ensure h5py package (compiled extensions) is included — map to _internal\h5py
$h5pySrc = Join-Path $envRoot 'Lib\site-packages\h5py'
if (Test-Path $h5pySrc) {
    $pyArgs += '--add-data'
    $pyArgs += ("{0};h5py" -f $h5pySrc)
}

& $python -m PyInstaller @pyArgs

if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller build failed.'
}

# Post-build: ensure Tcl/Tk data directories exist inside the bundle _internal folder
$bundleInternal = Join-Path $distDir 'image_stream_analysis\_internal'
if (Test-Path $bundleInternal) {
    foreach ($candidate in @((Join-Path $envRoot 'Library\lib\tcl8.6'), (Join-Path $envRoot 'Library\lib\tcl8'))) {
        if (Test-Path $candidate) {
            $dst = Join-Path $bundleInternal '_tcl_data'
            New-Item -ItemType Directory -Path $dst -Force | Out-Null
            Copy-Item -Path (Join-Path $candidate '*') -Destination $dst -Recurse -Force
            Write-Host "Copied _tcl_data from $candidate"
            break
        }
    }
    foreach ($candidate in @((Join-Path $envRoot 'Library\lib\tk8.6'), (Join-Path $envRoot 'Library\lib\tk8'))) {
        if (Test-Path $candidate) {
            $dst = Join-Path $bundleInternal '_tk_data'
            New-Item -ItemType Directory -Path $dst -Force | Out-Null
            Copy-Item -Path (Join-Path $candidate '*') -Destination $dst -Recurse -Force
            Write-Host "Copied _tk_data from $candidate"
            break
        }
    }
}

Write-Host "Build complete. EXE should be in: $distDir\image_stream_analysis\image_stream_analysis.exe"