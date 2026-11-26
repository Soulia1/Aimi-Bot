@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -u aimi_local.py %*
) else (
    py -3.12 -u aimi_local.py %*
)
if errorlevel 1 pause
