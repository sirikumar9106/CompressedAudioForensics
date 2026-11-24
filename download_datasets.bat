@echo off
title Dataset downloads (live) - safe to close and re-run, it resumes
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -m pip install -q requests
".venv\Scripts\python.exe" -u experiments\download_datasets.py
echo.
pause
