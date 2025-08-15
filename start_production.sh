  GNU nano 7.2                                      ./start_production.sh                                               #!/bin/bash

# Face Finder API - Production Startup Script
echo "🚀 Starting Face Finder API..."

# Activate virtual environment
if [ -d "/root/face_recognition/flask_test_env" ]; then
    echo "📦 Activating virtual environment..."
    source /root/flask_test_env/bin/activate
fi

# Create necessary directories
mkdir -p cache/embeddings
mkdir -p data

# Detect platform and start appropriate server
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OS" == "Windows_NT" ]]; then
    echo "🌟 Starting Waitress server (Windows)..."
    python waitress_config.py
else
    echo "🌟 Starting Gunicorn server (Unix)..."
    /root/flask_test_env/bin/gunicorn -c gunicorn.conf.py wsgi:app
fi
