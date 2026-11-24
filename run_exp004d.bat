@echo off
title EXP-004d tweaks A/B/C (live)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\run_exp004d.py
echo.
echo === Finished. Results in results\exp004d.json ===
pause
