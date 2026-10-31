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

rem Bind to localhost only: the server exposes this whole folder, so never listen on all interfaces.
start "" http://127.0.0.1:8080/
"!PY_EXE!" -m http.server 8080 --bind 127.0.0.1
pause
