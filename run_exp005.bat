@echo off
title EXP-005 challengers vs A (live)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\run_exp005.py
echo.
echo === Finished. Results in results\exp005.json ===
pause
