"""FastAPI application for cam2body-calib web UI."""

import webbrowser
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def create_app() -> FastAPI:
    app = FastAPI(title="cam2body-calib", version="0.1.0")

    from .routes.api import router

    app.include_router(router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


def main(host: str = "127.0.0.1", port: int = 8765):
    import uvicorn

    url = f"http://{host}:{port}"
    webbrowser.open(url)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
