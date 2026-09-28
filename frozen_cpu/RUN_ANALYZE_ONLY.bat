@echo off
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" ANALYZE_FRESH_RESULTS.py
"%PY%" PACKAGE_RESULTS.py
pause
