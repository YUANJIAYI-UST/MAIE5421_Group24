[CmdletBinding()]
param(
    [string]$Run = "runs/member4_full_C_seed42_20261115",
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw "Project Python environment is missing."
}
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$trainArguments = @(
    "-X", "utf8", "-B", "-m", "transparent_seg.train",
    "--config", "configs/member4_full_C_20261115.json",
    "--output", $Run
)
if ($Resume) {
    $trainArguments += @("--resume", $Resume)
}
& $pythonExecutable @trainArguments
if ($LASTEXITCODE -ne 0) {
    throw "C training stopped with exit code $LASTEXITCODE. Keep the run for diagnosis; do not change the frozen settings."
}
