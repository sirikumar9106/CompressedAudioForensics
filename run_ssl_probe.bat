@echo off
title EXP-004 WavLM probe (live)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo === Checking transformers ===
".venv\Scripts\python.exe" -m pip install "transformers>=4.40"
echo === Running EXP-004 (WavLM download progress shows below) ===
".venv\Scripts\python.exe" -u experiments\run_ssl_probe.py
echo.
echo === Finished. Results saved in results\ssl_probe_log.txt ===
pause
