@echo off
title EXP-004c re-compression (live)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\run_double_compression.py
echo.
echo === Finished. Results in results\double_compression.json ===
pause
