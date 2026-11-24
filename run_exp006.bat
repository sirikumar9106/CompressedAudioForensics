@echo off
title EXP-006 real-world robustness (live)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\run_exp006.py
echo.
echo === Finished. Results in results\exp006.json ===
pause
