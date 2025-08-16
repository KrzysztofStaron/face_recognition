# ASGI entry point for FastAPI
from main import app

# For uvicorn/gunicorn ASGI servers
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
