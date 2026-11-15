"""FastAPI service.

POST /analyze   multipart: file (image/PDF) + unit (JSON string)   -> rooms, annotated plan, financials
POST /evaluate  JSON: {rooms, unit}                                 -> financials for corrected rooms
POST /report    JSON: analyze result + unit                         -> PDF
GET  /catalog, /health, /monitoring
"""

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ryfc import __version__, monitoring, pipeline, report
from ryfc.config import load_yaml

app = FastAPI(title="Rental Yield & Finishing Cost Estimator", version=__version__)
MAX_BYTES = 15 * 1024 * 1024


class Room(BaseModel):
    type: str
    area_m2: float
    perimeter_m: float | None = None


class EvaluateIn(BaseModel):
    rooms: list[Room]
    unit: pipeline.UnitInfo


class ReportIn(BaseModel):
    result: dict
    unit: pipeline.UnitInfo


@app.get("/health")
def health():
    pipeline.segmenter()
    pipeline.rent_model()
    return {"status": "ok", "version": __version__}


@app.get("/catalog")
def catalog():
    return {"catalog": load_yaml("catalog.yaml"), "market": load_yaml("market.yaml")}


@app.get("/monitoring")
def monitoring_summary():
    return monitoring.summary()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), unit: str = Form("{}"), use_ocr: bool = Form(True)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (15 MB max)")
    try:
        info = pipeline.UnitInfo(**json.loads(unit))
        return pipeline.analyze(data, file.filename or "", info, use_ocr)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/evaluate")
def evaluate(body: EvaluateIn):
    if not body.rooms:
        raise HTTPException(422, "rooms is empty")
    return pipeline.evaluate([r.model_dump() for r in body.rooms], body.unit)


@app.post("/report")
def make_report(body: ReportIn):
    pdf = report.build(body.result, body.unit.model_dump())
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=finishing-yield-report.pdf"})
