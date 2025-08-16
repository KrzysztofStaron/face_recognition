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
- Single worker is recommended for ML workloads to avoid model loading overhead

### Troubleshooting

1. **Port already in use**: Change port in `uvicorn_config.py` or set PORT environment variable
2. **Memory issues**: Use single worker configuration (default)
3. **Timeout errors**: Increase timeout values in uvicorn configuration
4. **Permission errors**: Check file permissions for cache directory

## Project files and purpose

- `main.py`: FastAPI application.

  - Exposes endpoints: `POST /api/v0/embed`, `POST /api/v0/findIn`, `POST /api/v0/inspect`, `GET /api/health`, and cache utilities `GET /api/cache/stats`, `POST /api/cache/clear`, `POST /api/cache/cleanup`.
  - Contains helpers like `download_image_from_url`, target face selection, and cosine similarity.
  - Prints available endpoints on startup via `init_app()`.

- `embedding_cache.py`: Embedding/cache management.

  - Class `EmbeddingCache` handles computing and caching face embeddings and face metadata for both local files and URLs.
  - Supports metadata persistence, stats, clear/cleanup, and migration from an old cache format.

- `findAll.py`: Legacy/local dataset utilities and CLI.

  - Functions to search a local `data/` folder for matches: `find_matching_photos`, `load_reference_face`.
  - CLI commands: `clear-cache`, `cache-stats`, `cleanup-cache`, `migrate-cache`.

- `example_v0_usage.py`: Example client usage of the v0 API.

  - Demonstrates pre-warming cache, single-target search, and batch searches via HTTP.

- `uvicorn_config.py`: Production entrypoint for cross-platform (Uvicorn).

  - Reads `HOST`, `PORT`, `WORKERS` env vars and serves `main.app` with Uvicorn.
  - Ensures `cache/embeddings/` and `data/` directories exist.
  - Optimized for ML workloads with appropriate timeouts.

- `wsgi.py`: Renamed to ASGI entry point that exposes `app` for ASGI servers (Uvicorn).

- `start_production.bat`: Windows startup script.

  - Activates `venv` (if present), installs `requirements.txt`, creates dirs, and runs `uvicorn_config.py`.

- `start_production.sh`: Unix-like startup script.

  - Activates `venv` (if present), installs dependencies, creates dirs, and starts Uvicorn.

- `API_v0_DOCS.md`: Detailed API v0 usage guide (Polish) with request/response examples.

- `API_v0_DOCS.pdf`: PDF version of the API v0 documentation.

- `doc.txt`: Example payload for the legacy `POST /api/findAll` (local files) flow.

- `requirements.txt`: Python dependencies.

- `eval/`: Sample images for quick evaluation/demos.

- `out.json`: Sample output captured from a previous run.

- `README.md`: This deployment and operations guide.
