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
echo [Step 1/6] Creating Virtual Environment...
if not exist ".sedation\" (
    echo Creating virtual environment...
    python -m venv .sedation
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
echo [Step 2/6] Activating Virtual Environment...
call .sedation\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM ── Step 3: Install required packages ──
echo [Step 3/6] Installing Dependencies...
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo Installing requirements...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some packages may have failed to install
    echo Check requirements.txt for compatibility issues
)
echo Installing Jupyter...
pip install jupyter ipykernel papermill --quiet
echo [OK] Dependencies installed
echo.

REM ── Step 4: Register kernel ──
echo [Step 4/6] Registering Jupyter Kernel...
python -m ipykernel install --user --name=.sedation --display-name="Python (sedation)"
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Kernel registration may have failed
    echo You may need to select the kernel manually in Jupyter
)
echo [OK] Jupyter kernel registered
echo.

REM ── Step 5: Set environment variables ──
echo [Step 5/6] Setting Environment Variables...
set PYTHONWARNINGS=ignore
set PYTHONPATH=%cd%\code;%cd%\utils;%PYTHONPATH%
echo [OK] Environment variables set
echo.

REM ── Step 6: Check configuration ──
echo [Step 6/6] Checking Configuration...
if exist "config\config.json" (
    echo [OK] Configuration file found
) else (
    echo [WARNING] Configuration file not found!
    echo Please copy config\config_template.json to config\config.json
    echo and update it with your site-specific settings
    echo.
    pause
)

REM ── Create logs folder ──
if not exist logs (
    mkdir logs
)

REM ── Execution options ──
echo.
echo ============================================================================
echo                          EXECUTION OPTIONS
echo ============================================================================
echo.
echo Please choose how to run the analysis:
echo.
echo 1) Run analysis automatically (using papermill)
echo 2) Open Jupyter notebook for interactive analysis
echo 3) Exit (run manually later)
echo.

:choice_loop
set /p "choice=Enter your choice (1-3): "

if "!choice!"=="1" (
    echo.
    echo Running analysis notebook...
    cd code
    if exist 01_cohort_id.ipynb (
        echo Executing 01_cohort_id.ipynb...
        papermill 01_cohort_id.ipynb 01_cohort_id_output.ipynb > ..\logs\01_cohort_id.log 2>&1
        if %ERRORLEVEL% NEQ 0 (
            echo [ERROR] Notebook execution failed. Check logs\01_cohort_id.log for details.
            cd ..
            pause
            exit /b 1
        )
        echo [OK] Analysis completed successfully!
        cd ..
    ) else (
        echo [ERROR] Notebook not found: code\01_cohort_id.ipynb
        cd ..
        pause
        exit /b 1
    )
    goto :done
) else if "!choice!"=="2" (
    echo.
    echo Starting Jupyter notebook...
    cd code
    if exist 01_cohort_id.ipynb (
        start "" jupyter notebook 01_cohort_id.ipynb
        echo [OK] Jupyter notebook launched in your browser
    ) else (
        start "" jupyter notebook
        echo [OK] Jupyter launched. Please open 01_cohort_id.ipynb
    )
    cd ..
    goto :done
) else if "!choice!"=="3" (
    echo.
    echo Setup completed. You can run the analysis later with:
    echo    cd code ^&^& jupyter notebook 01_cohort_id.ipynb
    goto :done
) else (
    echo Invalid choice. Please enter 1, 2, or 3.
    goto :choice_loop
)

:done
REM ── Final message ──
echo.
echo ============================================================================
echo                          SETUP COMPLETE
echo ============================================================================
echo.
echo Results will be saved to: output\final\
echo Logs are available in: logs\
echo.
echo Thank you for using the CLIF Epidemiology of Sedation Project!
echo.
pause

exit /b 0