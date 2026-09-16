@echo off
cd /d "%~dp0"
python ui.py
if errorlevel 1 (
  echo.
  echo Failed to start UI. Make sure Python and dependencies are installed.
  pause
)
