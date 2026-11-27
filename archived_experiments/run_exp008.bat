@echo off
title EXP-008 merge datasets + fine-tune WavLM (live)
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
echo === Step 1: merge + balance datasets, pre-encode codec copies ===
".venv\Scripts\python.exe" -u experiments\exp008_data.py
echo === Step 2: fine-tune WavLM and test on In-the-Wild + ASVspoof 2019 ===
".venv\Scripts\python.exe" -u experiments\run_exp008_train.py
echo.
pause
