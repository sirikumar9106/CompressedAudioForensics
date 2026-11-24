@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" experiments\run_bitrate_sweep.py > results\bitrate_sweep_log.txt 2>&1
type results\bitrate_sweep_log.txt
pause
