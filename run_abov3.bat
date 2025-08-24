@echo off
REM ABOV3 Genesis - Windows Batch Runner
REM Run ABOV3 Genesis without installation

echo 🚀 Starting ABOV3 Genesis...
python run_abov3.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ ABOV3 Genesis encountered an error
    echo 🔧 Make sure you have installed dependencies:
    echo    pip install -r requirements.txt
    echo.
    pause
)