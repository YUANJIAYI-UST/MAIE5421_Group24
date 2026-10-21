param([int]$Seed = 42)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$ProjectPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
# Sequential execution prevents four runs competing for the same GPU.
foreach ($Experiment in @('A_baseline', 'B_boundary', 'C_augmentation', 'D_combined')) {
    & $ProjectPython -m transparent_seg.train --config "configs/$Experiment.json" --seed $Seed --output "runs/${Experiment}_seed${Seed}"
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $Experiment" }
}
# Freeze hyperparameters on validation before invoking test evaluation separately.
