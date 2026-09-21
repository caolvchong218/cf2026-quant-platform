@echo off
cd /d "%~dp0"
set "CF_PYTHON=.venv\Scripts\python.exe"
if exist ".venv-clean\Scripts\python.exe" set "CF_PYTHON=.venv-clean\Scripts\python.exe"
if not exist "%CF_PYTHON%" (
  echo Please run setup.cmd first.
  pause
  exit /b 1
)
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8501/_stcore/health' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1"
if not errorlevel 1 (
  start "" "http://127.0.0.1:8501"
  exit /b 0
)
start "" "http://127.0.0.1:8501"
"%CF_PYTHON%" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
pause
