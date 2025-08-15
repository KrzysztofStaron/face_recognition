# Face Finder API - Deployment Guide

## Production Deployment (Cross-Platform)

### Prerequisites

- Python 3.8+
- Required dependencies (see requirements.txt)

### Quick Start

#### Windows

```batch
# Install dependencies
pip install -r requirements.txt

# Start production server with Waitress
start_production.bat
```

#### Linux/macOS (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Start production server (auto-detects platform)
./start_production.sh
```

### Manual Server Startup

#### Windows - Waitress (Recommended)

```bash
python waitress_config.py
```

#### Linux/macOS - Gunicorn

```bash
# Basic startup
gunicorn -c gunicorn.conf.py wsgi:app

# Or with custom settings
gunicorn --bind 0.0.0.0:5003 --workers 4 --timeout 120 wsgi:app
```

#### Cross-Platform - Waitress

```bash
# Works on all platforms
python waitress_config.py
```

### Configuration

#### Waitress (Windows/Cross-Platform)

Configuration is in `waitress_config.py`. Environment variables:

- **HOST**: Server host (default: 0.0.0.0)
- **PORT**: Server port (default: 5003)
- **THREADS**: Number of threads (default: 8)

#### Gunicorn (Linux/macOS)

Configuration is in `gunicorn.conf.py`. Key settings:

- **Port**: 5003 (configurable in gunicorn.conf.py)
- **Workers**: CPU cores \* 2 + 1 (auto-detected)
- **Timeout**: 120 seconds (for face processing operations)
- **Memory optimization**: Uses /dev/shm for temporary files

### Environment Setup

1. **Cache Directory**: `cache/embeddings/` will be created automatically
2. **Permissions**: Make sure the app has read/write access to cache directory

### Health Check

Once running, test the API:

```bash
curl http://localhost:5003/api/health
```

### API Endpoints

- `POST /api/v0/embed` - Pre-warm cache with image URLs
- `POST /api/v0/findIn` - Find target in scope of images

Special:

- `GET /api/health` - Health check
- `GET /api/cache/stats` - Cache statistics
- `POST /api/cache/clear` - Clear cache
- `POST /api/cache/cleanup` - Remove invalid cache entries

### Performance Notes

- First requests may be slower as face detection models initialize
- Embedding cache improves performance for repeated image analysis
- Consider using a reverse proxy (nginx) for production deployments
- Monitor memory usage as face detection can be memory-intensive

### Troubleshooting

1. **Port already in use**: Change port in `gunicorn.conf.py`
2. **Memory issues**: Reduce number of workers in configuration
3. **Timeout errors**: Increase timeout value for large images
4. **Permission errors**: Check file permissions for cache directory
