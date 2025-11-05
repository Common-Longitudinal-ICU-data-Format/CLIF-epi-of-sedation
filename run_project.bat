@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM ── CLIF Epidemiology of Sedation Project - Windows Execution Script ──

REM ── Step 0: Go to script directory ──
cd /d %~dp0

REM ── Display banner ──
cls
echo.
echo ============================================================================
echo                           CLIF SEDATION PROJECT
echo             Epidemiology of Sedation in Mechanically Ventilated Patients
echo ============================================================================
echo.

REM ── Step 1: Create virtual environment if missing ──
echo [Step 1/5] Creating Virtual Environment...
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment
        echo Please ensure Python 3 is installed and available in PATH
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)
echo [OK] Virtual environment ready
echo.

REM ── Step 2: Activate virtual environment ──
echo [Step 2/5] Activating Virtual Environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM ── Step 3: Install required packages ──
echo [Step 3/5] Installing Dependencies...
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo Installing requirements...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some packages may have failed to install
    echo Check requirements.txt for compatibility issues
)
echo Installing Jupyter...
pip install jupyter ipykernel --quiet
echo [OK] Dependencies installed
echo.

REM ── Step 4: Register kernel ──
echo [Step 4/5] Registering Jupyter Kernel...
python -m ipykernel install --user --name=.venv --display-name="Python (sedation)"
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Kernel registration may have failed
    echo You may need to select the kernel manually in Jupyter
)
echo [OK] Jupyter kernel registered
echo.

REM ── Step 5: Check configuration ──
echo [Step 5/5] Checking Configuration...
if exist "config\config.json" (
    echo [OK] Configuration file found
) else (
    echo [WARNING] Configuration file not found!
    echo Please copy config\config_template.json to config\config.json
    echo and update it with your site-specific settings
    echo.
    pause
)

REM ── Final message ──
echo.
echo ============================================================================
echo                          SETUP COMPLETE
echo ============================================================================
echo.
echo [OK] Environment setup completed successfully!
echo.
echo ============================================================================
echo                          NEXT STEPS
echo ============================================================================
echo.
echo To run the analysis:
echo.
echo   1. Open your IDE (VS Code, PyCharm, etc.)
echo   2. Select the .venv Python interpreter
echo   3. Open: code\sedation_sbt.ipynb
echo   4. Run the notebook interactively
echo.
echo ============================================================================
echo.
echo Thank you for using the CLIF Epidemiology of Sedation Project!
echo.
pause

exit /b 0