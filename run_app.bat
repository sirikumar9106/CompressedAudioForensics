@echo off
title CompressedAudioForensics - local server
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo === Checking web server packages ===
".venv\Scripts\python.exe" -m pip install -q fastapi "uvicorn[standard]" python-multipart joblib
if not exist models\compressed_forensics_F.joblib (
  echo === Building deployment model from EXP-005 features ===
  ".venv\Scripts\python.exe" -m src.app.model_store
)
echo === Starting server: http://127.0.0.1:7860  (close this window to stop) ===
start "" http://127.0.0.1:7860
".venv\Scripts\python.exe" -m uvicorn src.app.server:app --host 127.0.0.1 --port 7860
pause
