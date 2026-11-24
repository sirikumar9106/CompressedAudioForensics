@echo off
title EXP-004b confidence stability (live)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\run_confidence_stability.py
echo.
echo === Finished. Results in results\confidence_stability.json ===
pause
