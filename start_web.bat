@echo off
echo ====================================
echo  Eco-Sort AI Web Interface Launcher
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if we're in the correct directory
if not exist "start_web_interface.py" (
    echo ❌ start_web_interface.py not found
    echo Please run this script from the Eco-Sort AI directory
    pause
    exit /b 1
)

echo 🚀 Starting Eco-Sort AI Web Interface...
echo.
echo 📊 This will:
echo    - Start the Flask backend server
echo    - Load YOLO and LLM models
echo    - Open your web browser automatically
echo.
echo ⏱️  Initial model loading may take 1-2 minutes...
echo.

REM Start the web interface
python start_web_interface.py --install-deps

if errorlevel 1 (
    echo.
    echo ❌ Failed to start the web interface
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo 👋 Web interface stopped
pause
