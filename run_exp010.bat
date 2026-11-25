@echo off
title EXP-010: modern TTS (MLAAD) + 8 layers + texture branch
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\exp008_data.py exp010
".venv\Scripts\python.exe" -u experiments\run_exp008_train.py exp010
pause
