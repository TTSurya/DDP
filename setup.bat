@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Could not switch to the project folder.
    exit /b 1
)

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Creating a project-local Python environment...
    python -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo ERROR: Could not create .venv. Install Python and ensure it is on PATH.
        exit /b 1
    )
)

echo Installing project dependencies in .venv...
"%PYTHON%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo ERROR: Dependency installation failed. See the message above.
    exit /b 1
)

set "OUTPUT_DIR=%~dp0outputs\bsd500_deblurring"

echo [1/2] Preparing the BSD500 dataset...
"%PYTHON%" "%~dp0prepare_bsds500.py"
if errorlevel 1 (
    echo ERROR: Dataset preparation failed. See the message above.
    exit /b 1
)

echo [2/2] Training and evaluating the model...
"%PYTHON%" "%~dp0run_bsd500_restoration.py" --output-dir "%OUTPUT_DIR%"
if errorlevel 1 (
    echo ERROR: Training or evaluation failed. See the message above.
    exit /b 1
)

echo.
echo Completed successfully.
echo Metrics:    "%OUTPUT_DIR%\test_metrics.json"
echo History:    "%OUTPUT_DIR%\training_history.json"
echo Checkpoint: "%OUTPUT_DIR%\best_model.pth"
exit /b 0