@echo off
title EXP-007a real-world diagnosis (live)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\run_exp007a_diagnose.py
echo.
echo === Finished. Results in results\exp007a.json ===
pause
