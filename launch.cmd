@echo off
cd /d "%~dp0"
set "CF_PYTHON=.venv\Scripts\python.exe"
if exist ".venv-clean\Scripts\python.exe" set "CF_PYTHON=.venv-clean\Scripts\python.exe"
if not exist "%CF_PYTHON%" (
  echo Please run setup.cmd first.
  pause
  exit /b 1
)
"%CF_PYTHON%" scripts\launch_platform.py %*
exit /b %errorlevel%
