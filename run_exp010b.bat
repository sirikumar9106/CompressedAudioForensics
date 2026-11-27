@echo off
title EXP-010b: backup clone for double compression (not deployed)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\run_exp010b_double.py
pause
