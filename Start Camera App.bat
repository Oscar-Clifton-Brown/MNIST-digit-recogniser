@echo off
setlocal
cd /d "%~dp0"

if not exist "Saved_paramaters\784_128_64_10.npz" (
    echo The saved model was not found. Make sure the repository download is complete.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python 3 is required. Install it from https://www.python.org/downloads/
        echo and select "Add python.exe to PATH", then run this file again.
        pause
        exit /b 1
    )
    echo Setting up the app environment for first use...
    py -3 -m venv .venv
    if errorlevel 1 goto setup_failed
)

".venv\Scripts\python.exe" -c "import cv2, matplotlib, numpy, PIL" >nul 2>nul
if errorlevel 1 (
    echo Installing app dependencies. This may take a few minutes...
    ".venv\Scripts\python.exe" -m pip install -r app-requirements.txt
    if errorlevel 1 goto setup_failed
)

echo Starting the digit recognizer...
".venv\Scripts\python.exe" camera_app.py
if errorlevel 1 goto app_failed
exit /b 0

:setup_failed
echo.
echo Setup failed. Check your internet connection and Python installation.
pause
exit /b 1

:app_failed
echo.
echo The app closed with an error. Review the message above.
pause
exit /b 1
