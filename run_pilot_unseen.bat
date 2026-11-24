@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set LOG=results\pilot_unseen_log.txt
echo [1/2] Building manifests... > %LOG%
if not exist data\manifests\asvspoof2019_eval.csv ".venv\Scripts\python.exe" experiments\prepare_data.py >> %LOG% 2>&1
echo [2/2] Running EXP-002 unseen-attack pilot... >> %LOG%
".venv\Scripts\python.exe" experiments\run_phase2_pilot.py --config config\pilot_unseen_attacks.yaml >> %LOG% 2>&1
echo PILOT_FINISHED >> %LOG%
type %LOG%
pause
