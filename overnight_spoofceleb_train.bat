@echo off
title OVERNIGHT: SpoofCeleb full download, then train EXP-010-SC, then test
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
for /L %%i in (1,1,3) do (
  if not exist "data\raw\spoofceleb\.full_extracted" (
    echo === SpoofCeleb full download, attempt %%i ===
    ".venv\Scripts\python.exe" -u experiments\download_spoofceleb_eval.py --all
  )
)
if not exist "data\raw\spoofceleb\.full_extracted" (
  echo SpoofCeleb download did not complete after 3 attempts - training not started.
  goto end
)
echo === Training EXP-010-SC ===
".venv\Scripts\python.exe" -u experiments\train_exp010_sc.py
:end
pause
