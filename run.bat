@echo off
echo ============================================
echo   Stock Price Prediction - AI Powered App
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.9+ from python.org
    pause
    exit /b 1
)

:: Install dependencies if not installed
echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo [2/3] Starting Flask server...
echo.
echo ============================================
echo   Open your browser at: http://localhost:5000
echo   Press Ctrl+C to stop the server
echo ============================================
echo.

:: Start the app
python app.py

pause
