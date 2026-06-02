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

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --name image_stream_analysis `
    --windowed `
    --onedir `
    --distpath $distDir `
    --workpath $buildDir `
    --specpath $projectRoot `
    --add-data "data;data" `
    run_gui.py

if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller build failed.'
}

Write-Host "Build complete. EXE should be in: $distDir\image_stream_analysis\image_stream_analysis.exe"