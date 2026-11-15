param([string]$Run = 'runs/member2_full_A_20261114')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$ProjectPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = (Get-Location).Path.Replace('\','/')
$env:HF_HOME = Join-Path (Get-Location) '.cache\huggingface'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

& $ProjectPython -m transparent_seg.train --config configs/member2_full_A_20261114.json --output $Run
if ($LASTEXITCODE -ne 0) { throw 'Full A training failed; retain checkpoints and inspect the log.' }
foreach ($Split in @('validation','test')) {
    & $ProjectPython -m transparent_seg.evaluate --checkpoint "$Run/best.pt" --manifest "manifests/member1_final_20261029/v1/$Split.json" --output "$Run/${Split}_evaluation" --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Selected checkpoint evaluation failed: $Split" }
}
& $ProjectPython scripts/summarize_full_baseline.py --run $Run --output reports/member2_full_A_20261114
if ($LASTEXITCODE -ne 0) { throw 'Full-run integrity checks failed.' }
