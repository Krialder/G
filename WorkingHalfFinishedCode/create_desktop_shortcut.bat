@echo off
chcp 1252 >nul
setlocal enabledelayedexpansion
REM Create Desktop Shortcut for Giessplan - Launcher.py Target
REM This script creates a desktop shortcut that specifically targets
REM launcher.py in the same directory, with full dynamic path detection

title Create Giessplan Desktop Shortcut

echo.
echo ========================================
echo   Giessplan Desktop Shortcut Creator
echo ========================================
echo.

REM Get the directory where THIS script is located
set "SCRIPT_DIR=%~dp0"

REM Construct path to launcher.py in the SAME directory
set "LAUNCHER_PATH=!SCRIPT_DIR!launcher.py"

echo Checking for launcher.py...
echo Script location: "!SCRIPT_DIR!"
echo Looking for: "!LAUNCHER_PATH!"

REM Check if launcher.py exists in the same directory
if not exist "!LAUNCHER_PATH!" (
    echo.
    echo ERROR: launcher.py not found!
    echo.
    echo Expected location: "!LAUNCHER_PATH!"
    echo Please make sure launcher.py is in the same folder as this script.
    echo.
    pause
    exit /b 1
)

echo [OK] Found launcher.py

REM Get user's desktop path dynamically
echo.
echo Detecting desktop path...

REM Try to get desktop path from registry
for /f "tokens=3*" %%i in ('reg query "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders" /v Desktop 2^>nul') do set "DESKTOP_PATH=%%i %%j"

REM Clean up any trailing spaces
if defined DESKTOP_PATH (
    for /l %%a in (1,1,10) do if "!DESKTOP_PATH:~-1!"==" " set "DESKTOP_PATH=!DESKTOP_PATH:~0,-1!"
)

REM Fallback to standard desktop if registry fails
if not defined DESKTOP_PATH (
    set "DESKTOP_PATH=%USERPROFILE%\Desktop"
)

REM Final fallback to OneDrive desktop
if not exist "!DESKTOP_PATH!" (
    if exist "%USERPROFILE%\OneDrive\Desktop" (
        set "DESKTOP_PATH=%USERPROFILE%\OneDrive\Desktop"
        echo Using OneDrive Desktop: "!DESKTOP_PATH!"
    )
)

echo Desktop path: "!DESKTOP_PATH!"

REM Define shortcut details
set "SHORTCUT_NAME=Giessplan.lnk"
set "SHORTCUT_PATH=!DESKTOP_PATH!\!SHORTCUT_NAME!"

echo.
echo Creating desktop shortcut...
echo Shortcut will be created at: "!SHORTCUT_PATH!"

REM Check if shortcut already exists
if exist "!SHORTCUT_PATH!" (
    echo.
    echo [!] Desktop shortcut already exists!
    set /p "OVERWRITE=Do you want to overwrite it? (y/n): "
    if /i not "!OVERWRITE!"=="y" if /i not "!OVERWRITE!"=="yes" (
        echo.
        echo Operation cancelled. Existing shortcut kept.
        pause
        exit /b 0
    )
    echo Overwriting existing shortcut...
)

REM Create temporary VBScript file
set "VBS_TEMP=%TEMP%\create_giessplan_shortcut.vbs"

echo.
echo Generating VBScript...

REM Generate VBScript with proper escaping and UNC path handling
(
echo Set objShell = WScript.CreateObject^("WScript.Shell"^)
echo Set objShortcut = objShell.CreateShortcut^("!SHORTCUT_PATH!"^)
echo objShortcut.TargetPath = "cmd.exe"
echo objShortcut.Arguments = "/k ""python """"!LAUNCHER_PATH!"""" ^&^& pause"""
echo objShortcut.WorkingDirectory = "!SCRIPT_DIR!"
echo objShortcut.Description = "Giessplan - Rotkreuz-Institut BBW (Python Launcher)"
echo objShortcut.IconLocation = "shell32.dll,76"
echo objShortcut.Save
) > "!VBS_TEMP!"

echo VBScript created. Executing...

REM Execute VBScript to create shortcut
cscript //nologo "!VBS_TEMP!" 2>nul
set "RESULT=%ERRORLEVEL%"

REM Clean up temporary VBScript
del "!VBS_TEMP!" 2>nul

REM Verify shortcut creation
echo.
if exist "!SHORTCUT_PATH!" (
    echo ✅ SUCCESS: Desktop shortcut created!
    echo.
    echo 📍 Shortcut Details:
    echo    Location: "!SHORTCUT_PATH!"
    echo    Target: cmd.exe with python launcher.py
    echo    Working Directory: "!SCRIPT_DIR!"
    echo.
    echo 🚀 You can now double-click the Giessplan icon on your desktop
    echo    to start the application with full console output.
) else (
    echo ❌ FAILED: Could not create desktop shortcut
    echo VBScript exit code: !RESULT!
    echo.
    echo 💡 Alternative method:
    echo    1. Right-click on launcher.py
    echo    2. Select "Send to > Desktop (create shortcut)"
    echo    3. Right-click the new shortcut > Properties
    echo    4. Change Target to: cmd.exe /k "cd /d "!SCRIPT_DIR!" && python "!LAUNCHER_PATH!" && pause"
)

echo.
echo ========================================
pause
