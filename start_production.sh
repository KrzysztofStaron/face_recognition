#!/bin/bash

# Face Finder API - Production Startup Script
echo "🚀 Starting Face Finder API..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Install requirements if needed
echo "📥 Checking dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p cache/embeddings
mkdir -p data

# Detect platform and start appropriate server
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OS" == "Windows_NT" ]]; then
    echo "🌟 Starting Waitress server (Windows)..."
    python waitress_config.py
else
    echo "🌟 Starting Gunicorn server (Unix)..."
    gunicorn -c gunicorn.conf.py wsgi:app
fi
