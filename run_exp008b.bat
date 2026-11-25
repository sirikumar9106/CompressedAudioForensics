@echo off
title EXP-008b all datasets + balanced threshold (live)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo === Step 1: merge ASVspoof5 + Fake-or-Real + LibriSeVoc + ASVspoof2019 ===
".venv\Scripts\python.exe" -u experiments\exp008_data.py exp008b
echo === Step 2: fine-tune WavLM + test ===
".venv\Scripts\python.exe" -u experiments\run_exp008_train.py exp008b
echo.
pause
