@echo off
setlocal enabledelayedexpansion
title Quality Gate Check
cd /d "%~dp0"

set "PY_EXE="
for /d %%D in ("%LOCALAPPDATA%\Python\pythoncore-*") do (
    if exist "%%~fD\python.exe" set "PY_EXE=%%~fD\python.exe"
)
if not defined PY_EXE (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%~fD\python.exe" set "PY_EXE=%%~fD\python.exe"
    )
)
if not defined PY_EXE set "PY_EXE=python"

"!PY_EXE!" -X utf8 quality_gate.py
pause
