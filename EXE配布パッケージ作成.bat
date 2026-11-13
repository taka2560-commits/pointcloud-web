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
echo [1.5/3] Writing build stamp (build_info.json: build time and commit)...
rem The stamp is bundled by the spec and shown in the window title, the About dialog and the log.
rem Delete the old stamp first so that a failed write can never leave a stale one in the package.
if exist "build_info.json" del /q "build_info.json"
"!PY_EXE!" !PY_ARG! tools\write_build_info.py

echo.
echo [2/3] Building EXE package (This takes about 2-4 minutes, please wait)...
rem Build from the checked-in spec file (manual files, hidden imports and open3d data are declared there).
rem Do NOT pass main.py here: PyInstaller would regenerate and overwrite E57_PointCloud_Tool.spec.
"!PY_EXE!" !PY_ARG! -m PyInstaller --noconfirm E57_PointCloud_Tool.spec

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo [2.5/3] Copying manuals and README next to the EXE (PyInstaller 6 puts data files under _internal)...
rem File names contain Japanese, so the selection is done in Python to keep this .bat pure ASCII.
"!PY_EXE!" !PY_ARG! -c "import glob, shutil; dst = 'dist/E57_PointCloud_Tool/'; skip = ('index.html', 'AGENTS.md', 'CLAUDE.md'); skip_prefix = ('requirements', 'QA_REPORT', 'AI_HANDOVER', '00_AI'); files = [f for f in glob.glob('*') if f.endswith(('.html', '.md', '.txt')) and f not in skip and not f.startswith(skip_prefix)]; [shutil.copy(f, dst) for f in files]; print('  Copied:', len(files), 'document(s)')"

echo.
echo [3/3] Archiving previous ZIP and creating new ZIP distribution package...
if exist "dist\E57_PointCloud_Tool_Dist.zip" (
    if not exist "dist\archive" mkdir "dist\archive"
    for /f "tokens=1-3 delims=/ " %%a in ("%DATE%") do set "DSTAMP=%%a%%b%%c"
    for /f "tokens=1-2 delims=:." %%a in ("%TIME%") do set "TSTAMP=%%a%%b"
    set "TSTAMP=!TSTAMP: =0!"
    move /y "dist\E57_PointCloud_Tool_Dist.zip" "dist\archive\E57_PointCloud_Tool_Dist_!DSTAMP!_!TSTAMP!.zip" >nul
    echo   Previous package moved to dist\archive\E57_PointCloud_Tool_Dist_!DSTAMP!_!TSTAMP!.zip
)
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
