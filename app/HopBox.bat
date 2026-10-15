@echo off
set "HERE=%~dp0"
if exist "%HERE%..\release\HopBox-Infer\HopBox-Infer.exe" (
  start "" "%HERE%..\release\HopBox-Infer\HopBox-Infer.exe"
  exit /b 0
)
"%HERE%..\.build-env\Scripts\python.exe" "%HERE%main.py"
