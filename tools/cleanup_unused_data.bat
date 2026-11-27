@echo off
title Cleanup: deleting unused datasets and caches (approved)
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\cleanup_unused_data.py
pause
