@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo [1/2] Building manifests... > results\pilot_log.txt
if not exist data\manifests\asvspoof2019_dev.csv ".venv\Scripts\python.exe" experiments\prepare_data.py >> results\pilot_log.txt 2>&1
echo [2/2] Running Phase 2 pilot... >> results\pilot_log.txt
".venv\Scripts\python.exe" experiments\run_phase2_pilot.py >> results\pilot_log.txt 2>&1
echo PILOT_FINISHED >> results\pilot_log.txt
type results\pilot_log.txt
pause
