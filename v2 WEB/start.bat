@echo off
chcp 65001 >nul
title DeepSeek AI Analysis Website
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (
    python server.py
    goto :end
)
where py >nul 2>nul
if %errorlevel%==0 (
    py server.py
    goto :end
)

echo.
echo [ERROR] Python not found.
echo Please install Python 3.9 or newer from https://www.python.org/downloads/
echo (check "Add Python to PATH" during installation).
echo.

:end
pause
