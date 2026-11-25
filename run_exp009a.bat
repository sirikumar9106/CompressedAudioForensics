@echo off
title EXP-009a image CNN vs all methods (live)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -m pip install -q --no-deps torchvision --index-url https://download.pytorch.org/whl/cu130
".venv\Scripts\python.exe" -u experiments\run_exp009a_image.py
echo.
pause
