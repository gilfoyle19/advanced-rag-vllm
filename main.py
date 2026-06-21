# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:api",
        host="0.0.0.0",
        port=8000,
        reload=True,        # auto-reload on code changes during development
    )
