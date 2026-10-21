param([string]$Run = 'runs/smoke')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$ProjectPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $ProjectPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
if (!(Test-Path 'manifests/smoke/train.json')) {
    & $ProjectPython scripts/make_smoke_data.py
    if ($LASTEXITCODE -ne 0) { throw 'Fixture preparation failed' }
}
& $ProjectPython -m transparent_seg.train --config configs/smoke.json --output $Run
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
& $ProjectPython -m transparent_seg.evaluate --checkpoint "$Run/best.pt" --manifest manifests/smoke/test.json --output "$Run/evaluation" --robustness --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
& $ProjectPython -m transparent_seg.predict --checkpoint "$Run/best.pt" --image data/smoke/test/easy/images/000.jpg --output "$Run/prediction" --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Prediction failed' }
Write-Output "Passed. Synthetic smoke results in $Run are not benchmark evidence."
