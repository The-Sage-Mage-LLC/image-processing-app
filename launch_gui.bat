@echo off
REM Launch Image Processing Application GUI
REM Optimized for 32" Samsung Smart Monitor
REM Project ID: Image Processing App 20251119

echo.
echo ========================================
echo Image Processing Application
echo Optimized for 32" Samsung Monitor
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist "venv_312_fixed\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv_312_fixed\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found
    echo Using system Python installation
)

REM Launch the GUI
echo Launching GUI in maximized mode...
echo Target Resolution: 1920x1080 (32" Monitor)
echo.

python gui_launcher.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo Error: GUI failed to launch
    echo Check the error messages above
    echo.
    pause
) else (
    echo.
    echo GUI closed successfully
)

REM Deactivate virtual environment if it was activated
if defined VIRTUAL_ENV (
    deactivate
)

echo.
pause