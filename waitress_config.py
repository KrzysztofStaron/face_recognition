# Waitress configuration for Windows/cross-platform deployment
from waitress import serve
from main import app
import os

def run_server():
    """Run the Face Finder API with Waitress WSGI server"""
    
    # Configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5003))
    threads = int(os.getenv('THREADS', 8))
    
    print(f"🚀 Starting Face Finder API with Waitress...")
    print(f"📡 Server: http://{host}:{port}")
    print(f"🧵 Threads: {threads}")
    print(f"💾 Cache directory: cache/embeddings/")
    print(f"📁 Data directory: data/")
    print()
    print("Available endpoints:")
    print("🔄 POST /api/v0/embed - Pre-warm cache with image URLs")
    print("🔍 POST /api/v0/findIn - Find target in scope of images") 
    print("🔍 POST /api/findIn - Find matches in data directory (legacy)")
    print("💚 GET /api/health - Health check")
    print("📊 GET /api/cache/stats - Cache statistics")
    print("🧹 POST /api/cache/clear - Clear all cached embeddings")
    print("🧹 POST /api/cache/cleanup - Remove invalid cache entries")
    print()
    
    # Ensure directories exist
    os.makedirs('cache/embeddings', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Start server
    serve(
        app,
        host=host,
        port=port,
        threads=threads,
        url_scheme='http',
        # Performance settings
        cleanup_interval=30,
        channel_timeout=120,
        # Adjust for face processing workloads
        recv_bytes=65536,
        send_bytes=65536,
        # Connection settings
        backlog=1024,
        # Logging
        _quiet=False
    )

if __name__ == '__main__':
    run_server()
