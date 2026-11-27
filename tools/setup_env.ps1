# One-shot environment setup for Windows + NVIDIA RTX 50-series (Blackwell, sm_120 -> needs CUDA >= 12.8 wheels).
# Creates .venv in this folder; does NOT touch your global Python packages. Log: setup_log.txt
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot   # project folder (this script lives in tools/)
Set-Location -LiteralPath $Root
$log = Join-Path $Root "setup_log.txt"
Start-Transcript -Path $log -Force | Out-Null
function Refresh-Path { $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User") }

# 1. Pick a Python 3.10-3.14 interpreter (prefer the one already installed)
$py = $null
foreach ($cand in @("py -3.13","py -3.12","py -3.14","py -3.11","python")) {
    $parts = $cand.Split(" ")
    try {
        $v = & $parts[0] $parts[1..9] -c "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(1[0-4])$') { $py = $parts; Write-Host "Using Python $v via '$cand'"; break }
    } catch {}
}
if (-not $py) {
    Write-Host "No compatible Python (3.10-3.14) found -> installing Python 3.12 side-by-side via winget"
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent
    Refresh-Path; $py = @("py","-3.12")
}

# 2. Virtual environment
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $py[0] $py[1..9] -m venv .venv }
$vp = Join-Path $Root ".venv\Scripts\python.exe"
& $vp -m pip install --upgrade pip setuptools wheel

# 3. PyTorch with CUDA >= 12.8 (required for RTX 5050). Try newest CUDA index first.
$ok = $false
foreach ($cu in @("cu130","cu129","cu128")) {
    Write-Host "Trying PyTorch index $cu ..."
    & $vp -m pip install --upgrade torch torchaudio --index-url "https://download.pytorch.org/whl/$cu"
    if ($LASTEXITCODE -eq 0) {
        $cuda = & $vp -c "import torch;print(torch.cuda.is_available())"
        if ($cuda -eq "True") { $ok = $true; Write-Host "PyTorch OK with $cu"; break }
    }
}
if (-not $ok) { Write-Host "WARNING: CUDA PyTorch not working. Update NVIDIA driver (>= 570) and re-run." }

# 4. Other dependencies (torch already satisfied)
& $vp -m pip install numpy scipy scikit-learn soundfile pyyaml pytest gradio

# 5. FFmpeg
Refresh-Path
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "Installing FFmpeg via winget"
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements --silent
    Refresh-Path
}

# 6. Verify
Write-Host "===== VERIFY ====="
& $vp -c "import torch,sys;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available());print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None);print('cc',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None);x=torch.randn(1024,1024,device='cuda') if torch.cuda.is_available() else None;print('gpu_matmul_ok', bool(x is not None and (x@x).sum().isfinite()))"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
ffmpeg -hide_banner -version | Select-Object -First 1
& $vp -m pytest -q -p no:cacheprovider tests
& $vp experiments\run_phase0_audit.py | Out-Null
Write-Host "SETUP_DONE"
Stop-Transcript | Out-Null
