"""Main module for the FastAPI app."""

from pathlib import Path

import marimo
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from optuna.storages import RDBStorage
from optuna_dashboard import wsgi

server = (
    marimo.create_asgi_app()
    .with_app(path="/watershed_nucleus", root="app/pages/watershed_nucleus.py")
    .with_app(path="/svm", root="app/pages/svm.py")
)

storage = RDBStorage("sqlite:///params.db")
optuna_app = wsgi(storage=storage)

app = FastAPI(
    redoc_url=None,
    docs_url=None,
)

@app.get("/")
def read_root() -> HTMLResponse:
    """Return landing page.

    Returns:
        HTMLResponse: The landing page.

    """
    file_content = Path("app/land.html").read_text()
    return HTMLResponse(
        content=file_content,
        status_code=200,
    )


app.mount("/", server.build())
app.mount("/optuna", optuna_app)
