@echo off
title EXP-007 extract + In-the-Wild baseline (live)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo === Step 1: finish downloads / extract (already-downloaded files are skipped) ===
".venv\Scripts\python.exe" -u experiments\download_datasets.py
echo === Step 2: EXP-007 baseline on In-the-Wild ===
".venv\Scripts\python.exe" -u experiments\run_exp007_inthewild.py
echo.
pause
