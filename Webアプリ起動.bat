@echo off
setlocal enabledelayedexpansion
title E57 Web Point Cloud Viewer
cd /d "%~dp0"

echo ========================================================
echo  Starting Web Point Cloud Viewer ^(http://localhost:8080/^)...
echo ========================================================

start http://localhost:8080/

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

"!PY_EXE!" -m http.server 8080
pause
