param([ValidateSet('cpu', 'cuda')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (!(Test-Path '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create venv' }
}
$ProjectPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $ProjectPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
$TorchIndex = if ($Device -eq 'cuda') { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
$TorchVersion = if ($Device -eq 'cuda') { 'torch==2.7.1+cu128' } else { 'torch==2.7.1+cpu' }
& $ProjectPython -m pip install $TorchVersion --index-url $TorchIndex --extra-index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $ProjectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $ProjectPython -c "import torch; print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
if ($LASTEXITCODE -ne 0) { throw 'Import check failed' }
if ($Device -eq 'cuda') {
    & $ProjectPython -c "import torch; assert torch.cuda.is_available(); x=torch.randn(64,64,device='cuda',requires_grad=True); (x@x).sum().backward(); print('CUDA forward/backward passed')"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA computation failed; inspect driver/wheel compatibility' }
}
