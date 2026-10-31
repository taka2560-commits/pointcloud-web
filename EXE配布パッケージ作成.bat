@echo off
setlocal enabledelayedexpansion
title Build Standalone EXE Package
cd /d "%~dp0"

echo ========================================================
echo  Building Standalone EXE Package (dist\E57_PointCloud_Tool)
echo ========================================================

set "PY_EXE="
set "PY_ARG="

for /d %%D in ("%LOCALAPPDATA%\Python\pythoncore-*") do (
    if exist "%%~fD\python.exe" set "PY_EXE=%%~fD\python.exe"
)
if not defined PY_EXE (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%~fD\python.exe" set "PY_EXE=%%~fD\python.exe"
    )
)
if not defined PY_EXE (
    where py >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        set "PY_EXE=py"
        set "PY_ARG=-3"
    )
)
if not defined PY_EXE (
    where python >nul 2>nul
    if !ERRORLEVEL! EQU 0 set "PY_EXE=python"
)

if not defined PY_EXE (
    echo.
    echo ========================================================
    echo  [ERROR] Python is NOT installed on this PC!
    echo.
    echo  NOTICE:
    echo  - You can only BUILD the EXE on the PC with Python installed.
    echo  - Do NOT run this batch file on client/other PCs.
    echo  - On other PCs, simply unzip and run "E57_PointCloud_Tool.exe"!
    echo ========================================================
    echo.
    pause
    exit /b 1
)

echo [1/3] Ensuring PyInstaller is installed...
"!PY_EXE!" !PY_ARG! -m pip install pyinstaller

echo.
echo [2/3] Building EXE package (This takes about 2-4 minutes, please wait)...
"!PY_EXE!" !PY_ARG! -m PyInstaller --noconfirm --onedir --windowed --name "E57_PointCloud_Tool" --collect-all pye57 --collect-submodules OpenGL main.py

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Creating ZIP distribution package...
"!PY_EXE!" !PY_ARG! -c "import os, shutil; shutil.make_archive(os.path.join('dist', 'E57_PointCloud_Tool_Dist'), 'zip', root_dir='dist', base_dir='E57_PointCloud_Tool'); print('Created: dist\\E57_PointCloud_Tool_Dist.zip')"

echo.
echo ========================================================
echo  [SUCCESS] Standalone EXE Package is Ready!
echo.
echo  Output Location:
echo    1) Folder: dist\E57_PointCloud_Tool\E57_PointCloud_Tool.exe
echo    2) ZIP:    dist\E57_PointCloud_Tool_Dist.zip
echo.
echo  HOW TO SHARE WITH OTHER PCs:
echo    Copy "dist\E57_PointCloud_Tool_Dist.zip" to other PCs,
echo    extract it, and double-click "E57_PointCloud_Tool.exe".
echo    (No Python installation needed on target PCs!)
echo ========================================================
pause
