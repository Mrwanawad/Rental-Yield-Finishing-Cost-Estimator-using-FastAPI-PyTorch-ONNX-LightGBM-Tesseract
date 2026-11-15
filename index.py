"""Vercel entrypoint: the FastAPI service under /api; the web UI is served statically from public/."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("RYFC_ROOT", str(ROOT))
os.environ.setdefault("RYFC_WORK_DIR", "/tmp/ryfc")  # only /tmp is writable on Vercel

from fastapi import FastAPI  # noqa: E402

from ryfc.api import app as api  # noqa: E402

app = FastAPI(title="ryfc-vercel", docs_url=None, redoc_url=None)
app.mount("/api", api)
if not os.getenv("VERCEL"):  # on Vercel the CDN serves public/; locally FastAPI does
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=ROOT / "public", html=True), name="web")
