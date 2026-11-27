@echo off
title SpoofCeleb eval-only stream (134 GB download, few GB on disk)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\download_spoofceleb_eval.py
pause
