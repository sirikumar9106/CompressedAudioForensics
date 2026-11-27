@echo off
title MLAAD download (live) - safe to close and re-run, it resumes
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
powershell -NoProfile -Command "& '.venv\Scripts\python.exe' -u experiments\download_mlaad.py 2>&1 | ForEach-Object { \"$_\" } | Tee-Object -FilePath results\mlaad_download_log.txt"
pause
