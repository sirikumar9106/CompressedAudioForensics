@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup_env.ps1"
echo.
echo Setup finished. See setup_log.txt. You can close this window.
pause
