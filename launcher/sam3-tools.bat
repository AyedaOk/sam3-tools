@echo off
set APP_DIR=%~dp0..
cd /d "%APP_DIR%" || exit /b 1
"%APP_DIR%\.venv\Scripts\python.exe" main.py %*
