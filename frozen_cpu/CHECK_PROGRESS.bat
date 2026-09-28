@echo off
cd /d "%~dp0"
for /f %%N in ('dir /b /a-d "runs\v2_7_fresh_cpu\tasks\*.json" 2^>nul ^| find /c /v ""') do set COUNT=%%N
echo Completed task JSONs: %COUNT% / 5760
if exist "runs\v2_7_fresh_cpu\TASK_MANIFEST.json" (
  echo.
  type "runs\v2_7_fresh_cpu\TASK_MANIFEST.json"
)
pause
