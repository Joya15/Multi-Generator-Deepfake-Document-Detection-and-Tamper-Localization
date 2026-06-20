param(
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Venv = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
if ([string]::IsNullOrWhiteSpace($Python) -or !(Test-Path $Python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { throw "No Python executable found. Pass -Python <path>." }
    $Python = $cmd.Source
}
if (!(Test-Path $VenvPython)) {
    & $Python -m venv $Venv
}
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-cuda.txt")
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
& $VenvPython -m pip install -e $ProjectRoot
& $VenvPython -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
