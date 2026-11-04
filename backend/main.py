from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from analysis.api import router


app = FastAPI(title="sklearn-map-lab")
app.include_router(router)

frontend = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend.is_dir():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
