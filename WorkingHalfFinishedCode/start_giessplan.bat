@echo off
REM Gießplan Launcher Batch File
REM This batch file starts the Gießplan application launcher

title Gießplan Launcher

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"

REM Change to the script directory
cd /d "%SCRIPT_DIR%"

echo Starting Gießplan Launcher...
echo.

REM Try to run the launcher with Python
python launcher.py 2>nul
if %errorlevel% neq 0 (
    echo Python not found in PATH, trying py command...
    py launcher.py 2>nul
    if %errorlevel% neq 0 (
        echo Python is not installed or not found in PATH.
        echo.
        echo Please install Python from: https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
)

REM If we get here, the launcher should have started successfully
echo Launcher finished.
pause
