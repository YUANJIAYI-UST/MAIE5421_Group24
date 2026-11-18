param(
    [Parameter(Mandatory)] [ValidateSet('B03_D','B03_B','E03_A','E03_D')] [string]$Stage,
    [string]$RunRoot = 'runs/member2_b03_e03_20261118',
    [switch]$TrainOnly,
    [switch]$EvaluateOnly,
    [string]$Resume,
    [string]$EvalDeps
)
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
$env:PYTHONUNBUFFERED = '1'
$Isolated = $Stage.StartsWith('E03')
$Experiment = $Stage.Substring(0,3).ToLowerInvariant()
$Model = $Stage.Substring(4)
$Run = Join-Path $RunRoot $Stage
$ResumeArgs = @()
if ($Resume) { $ResumeArgs = @('--resume', $Resume) }
if ($TrainOnly -and $EvaluateOnly) { throw 'Choose only one stage restriction' }
if (-not $EvaluateOnly) {
    & $ProjectPython -m transparent_seg.train --config "configs/group24_${Experiment}_${Model}_seed42.json" --output $Run @ResumeArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $Stage. Preserve the checkpoints and inspect the log." }
}
if ($TrainOnly) { return }
$Backend = 'scipy'
if ($EvalDeps) {
    $env:PYTHONPATH = (Resolve-Path -LiteralPath $EvalDeps).Path
    $Backend = 'opencv'
}
$Output = Join-Path $Run 'evaluation'
$Checkpoint = Join-Path $Run 'best.pt'
if ($Isolated) {
    & $ProjectPython scripts/evaluate_group24_e03.py --checkpoint $Checkpoint --output $Output --boundary-backend $Backend
} else {
    $EvalResumeArgs = @()
    if (Test-Path (Join-Path $Output 'identity.json')) { $EvalResumeArgs = @('--resume') }
    & $ProjectPython -m transparent_seg.evaluate_resumable --checkpoint $Checkpoint --manifest manifests/member1_final_20261029/v1/test.json --output $Output --conditions clean --boundary-backend $Backend @EvalResumeArgs
}
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $Stage." }
