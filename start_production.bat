@echo off
echo 🚀 Starting Face Finder API with Uvicorn (Windows-compatible)...

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install requirements if needed
echo 📥 Checking dependencies...
pip install -r requirements.txt

REM Create necessary directories
if not exist "cache\embeddings" mkdir cache\embeddings
if not exist "data" mkdir data

REM Start Uvicorn server
echo 🌟 Starting Uvicorn server...
python uvicorn_config.py

pause
