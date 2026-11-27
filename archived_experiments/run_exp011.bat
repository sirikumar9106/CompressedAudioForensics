@echo off
title EXP-011: SpoofCeleb test (no training)
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -u experiments\run_exp011_spoofceleb.py
pause
