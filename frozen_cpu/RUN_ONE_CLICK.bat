@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1

echo ============================================================
echo ContextBench v2.7 PSC-BS - FRESH CPU CONFIRMATION ONE-CLICK
echo Frozen protocol: DO NOT change thresholds/states/models/budgets.
echo ============================================================

where py >nul 2>nul
if %errorlevel%==0 (
  set "BOOTPY=py -3.12"
) else (
  where python >nul 2>nul
  if not %errorlevel%==0 goto NOPY
  set "BOOTPY=python"
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/7] Creating Python virtual environment...
  %BOOTPY% -m venv .venv
  if errorlevel 1 goto FAIL
) else (
  echo [1/7] Virtual environment already exists - reuse.
)
set "PY=.venv\Scripts\python.exe"

if not exist ".deps_v27_ok" (
  echo [2/7] Installing Python dependencies...
  "%PY%" -m pip install --upgrade pip
  if errorlevel 1 goto FAIL
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 goto FAIL
  echo ok> .deps_v27_ok
) else (
  echo [2/7] Dependencies already installed - skip.
)

echo [3/7] Verifying frozen v2.7 method...
"%PY%" VERIFY_FREEZE.py
if errorlevel 1 goto FAIL

echo [4/7] Downloading locked fresh datasets (cached files are skipped)...
"%PY%" DOWNLOAD_FRESH_DATA.py
if errorlevel 1 goto FAIL
"%PY%" PREPARE_FRESH_DATA.py
if errorlevel 1 goto FAIL

echo [5/7] Running fresh CPU confirmation. Safe to rerun after interruption; completed task JSONs are reused.
"%PY%" v27_fresh_cpu.py
if errorlevel 1 goto FAIL

echo [6/7] Applying PREDECLARED confirmation gates...
"%PY%" ANALYZE_FRESH_RESULTS.py
if errorlevel 1 goto FAIL

echo [7/7] Packaging results...
"%PY%" PACKAGE_RESULTS.py
if errorlevel 1 goto FAIL

echo.
echo ============================================================
echo DONE.
echo Upload this file back to ChatGPT:
echo   V27_FRESH_CPU_RESULTS.zip
echo Gate report:
echo   runs\v2_7_fresh_cpu\analysis\FRESH_CPU_GATE_REPORT.md
echo ============================================================
pause
exit /b 0

:NOPY
echo ERROR: Python 3.12 was not found. Install 64-bit Python 3.12 and rerun.
pause
exit /b 2
:FAIL
echo.
echo ERROR: pipeline stopped. Do NOT delete runs or incoming; fix the shown error and double-click RUN_ONE_CLICK.bat again to resume.
pause
exit /b 1
