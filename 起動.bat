@echo off
setlocal enabledelayedexpansion
title E57 Point Cloud Tool
cd /d "%~dp0"

echo ========================================================
echo  Starting E57 Point Cloud Tool...
echo ========================================================

if exist "%~dp0dist\E57_PointCloud_Tool\E57_PointCloud_Tool.exe" (
    start "" "%~dp0dist\E57_PointCloud_Tool\E57_PointCloud_Tool.exe"
    exit /b 0
)

set "PY_EXE="
set "PY_ARG="

if exist "%~dp0python\python.exe" (
    set "PY_EXE=%~dp0python\python.exe"
)
if not defined PY_EXE (
    for /d %%D in ("%LOCALAPPDATA%\Python\pythoncore-*") do (
        if exist "%%~fD\python.exe" set "PY_EXE=%%~fD\python.exe"
    )
)
if not defined PY_EXE (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%~fD\python.exe" set "PY_EXE=%%~fD\python.exe"
    )
)
if not defined PY_EXE (
    for /d %%D in ("C:\Program Files\Python3*") do (
        if exist "%%~fD\python.exe" set "PY_EXE=%%~fD\python.exe"
    )
)
if not defined PY_EXE (
    where py >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        py -3 -c "import sys" >nul 2>nul
        if !ERRORLEVEL! EQU 0 (
            set "PY_EXE=py"
            set "PY_ARG=-3"
        )
    )
)
if not defined PY_EXE (
    where python >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        python -c "import sys" >nul 2>nul
        if !ERRORLEVEL! EQU 0 set "PY_EXE=python"
    )
)

if not defined PY_EXE (
    echo.
    echo [ERROR] Python 64-bit is not installed on this PC.
    echo Please install Python 3.11 or 3.12 ^(64-bit^) from:
    echo   https://www.python.org/downloads/
    echo Make sure to check "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python: "!PY_EXE!" !PY_ARG!

"!PY_EXE!" !PY_ARG! -c "import numpy, scipy, pye57, PySide6, OpenGL" >nul 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [SETUP] Installing required packages ^(numpy, scipy, pye57, PySide6, PyOpenGL^)...
    "!PY_EXE!" !PY_ARG! -m pip install --upgrade pip
    "!PY_EXE!" !PY_ARG! -m pip install -r requirements-desktop.txt
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo [ERROR] Failed to install packages. Check internet connection.
        pause
        exit /b 1
    )
)

"!PY_EXE!" !PY_ARG! -X utf8 main.py

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
