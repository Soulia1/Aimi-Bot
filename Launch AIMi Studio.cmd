@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" aimi_app.py
) else (
    python aimi_app.py
)
if errorlevel 1 pause
