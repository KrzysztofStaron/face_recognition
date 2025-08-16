  GNU nano 7.2                                      ./start_production.sh                                               #!/bin/bash

# Face Finder API - Production Startup Script
echo "🚀 Starting Face Finder API with FastAPI..."

# Activate virtual environment
if [ -d "/root/face_recognition/flask_test_env" ]; then
    echo "📦 Activating virtual environment..."
    source /root/flask_test_env/bin/activate
fi

# Create necessary directories
mkdir -p cache/embeddings
mkdir -p data

# Start Uvicorn server (cross-platform)
echo "🌟 Starting Uvicorn server..."
python uvicorn_config.py
