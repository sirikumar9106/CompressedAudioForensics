@echo off
title EXP-009b selected features + combination with WavLM (live)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\run_exp009b_selected.py
echo.
pause
