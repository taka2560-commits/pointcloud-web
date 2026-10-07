@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ========================================================
echo   Web版 3D点群エディタ & ノイズ除去ツールを起動中...
echo ========================================================
echo.
echo ブラウザが自動的に開きます。
echo 終了するときはこの黒い画面を閉じてください。
echo.

rem ブラウザを自動で開く
start http://localhost:8080/

rem Python標準の内蔵Webサーバーを起動
python -m http.server 8080
if %ERRORLEVEL% NEQ 0 (
    "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" -m http.server 8080
)

pause
