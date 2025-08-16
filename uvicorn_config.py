# Uvicorn configuration for Windows/cross-platform deployment
import uvicorn
from main import app
import os

def run_server():
    """Run the Face Finder API with Uvicorn ASGI server"""
    
    # Configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5003))
    workers = int(os.getenv('WORKERS', 1))  # FastAPI works better with single worker for ML workloads
    
    print(f"🚀 Starting Face Finder API with Uvicorn...")
    print(f"📡 Server: http://{host}:{port}")
    print(f"👥 Workers: {workers}")
    print(f"💾 Cache directory: cache/embeddings/")
    print()
    print("Available endpoints:")
    print("🔄 POST /api/v0/embed - Pre-warm cache with image URLs")
    print("🔍 POST /api/v0/findIn - Find target in scope of images") 
    print("💚 GET /api/health - Health check")
    print("📊 GET /api/cache/stats - Cache statistics")
    print("🧹 POST /api/cache/clear - Clear all cached embeddings")
    print("🧹 POST /api/cache/cleanup - Remove invalid cache entries")
    print()
    
    # Ensure directories exist
    os.makedirs('cache/embeddings', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # SSL Configuration
    ssl_keyfile = os.getenv('SSL_KEYFILE', '/etc/letsencrypt/live/fotoklaser-facedetection.duckdns.org/privkey.pem')
    ssl_certfile = os.getenv('SSL_CERTFILE', '/etc/letsencrypt/live/fotoklaser-facedetection.duckdns.org/fullchain.pem')
    
    # Check if SSL certificates exist
    use_ssl = os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile)
    
    if use_ssl:
        print(f"🔒 SSL enabled with certificates:")
        print(f"   Key: {ssl_keyfile}")
        print(f"   Cert: {ssl_certfile}")
        protocol = "https"
    else:
        print("⚠️  SSL certificates not found, running HTTP only")
        print(f"   Looking for key: {ssl_keyfile}")
        print(f"   Looking for cert: {ssl_certfile}")
        protocol = "http"
    
    print(f"📡 Server: {protocol}://{host}:{port}")
    
    # Start server
    if use_ssl:
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            # SSL Configuration
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            # Performance settings for ML workloads
            timeout_keep_alive=120,
            timeout_graceful_shutdown=60,
            # Logging
            log_level="info",
            access_log=True
        )
    else:
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            # Performance settings for ML workloads
            timeout_keep_alive=120,
            timeout_graceful_shutdown=60,
            # Logging
            log_level="info",
            access_log=True
        )

if __name__ == '__main__':
    run_server()
