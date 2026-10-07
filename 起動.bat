@echo off
title E57点群ノイズ処理ツール
cd /d "%~dp0"

echo ========================================================
echo  E57点群ノイズ処理ツールを起動しています...
echo ========================================================

set "PY_PATH=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"

if exist "%PY_PATH%" (
    echo [OK] Pythonを検出しました: %PY_PATH%
    "%PY_PATH%" main.py
) else (
    echo [情報] 標準パスのPythonを試行します...
    python main.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo --------------------------------------------------------
    echo [エラー] 起動に失敗しました。
    echo --------------------------------------------------------
    echo 何かキーを押すとウィンドウを閉じます。
    pause
)
