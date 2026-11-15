# Rental Yield & Finishing Cost Estimator

Upload the floor plan of an unfinished unit. The system finds the rooms and their areas, prices three
finishing packages (basic / standard / premium) with a line-item breakdown, predicts the monthly rent
range for each package, and returns gross/net yield, payback and the package with the best 10-year value,
plus a PDF report. Built for the "Nawy Unlocked" use case (finishing + rentals). Design: [architecture.md](architecture.md).

```
plan (PNG/JPG/PDF) -> OpenCV pre-process (PDF render, deskew, crop, dark-mode invert)
  -> U-Net (MobileNetV2) segmentation, ONNX Runtime CPU: wall + 6 room classes
  -> room instances (walled free-space components, typed by probability vote, open-plan split)
  -> Tesseract OCR: room names override the model, printed "4.20 x 3.60" dimensions give a scale
  -> scale: entered BUA/net area (cross-checked against OCR dimensions) -> m2 per room, perimeters
  -> cost engine: catalog rules per room/item (EGP) x LightGBM adjustment (region, floor, size,
     contractor tier) x catalog inflation since its date, 80% range
  -> rent: LightGBM quantile q10/q50/q90 + split-conformal calibration, per finishing package
  -> yield: gross/net yield, finishing & total payback, PV of net rent, NPV of finishing
  -> FastAPI + Streamlit UI (editable room table, recomputes instantly), PDF report
```

## Results

All training data is synthetic for now (see Limitations). Full tables: [reports/](reports/).

| Stage | Metric | Value |
|---|---|---|
| Segmentation (300 held-out plans) | mIoU, wall + 6 room classes (v1 -> v2) | 0.953 -> 0.991 |
| Rooms | precision / recall at IoU 0.5 | 99.8% / 99.1% |
| Rooms | type accuracy | 100% |
| Area | per-room error, scale from entered net area | 0.6% (p90 1.2%) |
| Area | total area from OCR'd room dimensions (no area entered), 52% of plans | 2.2% |
| Cost | MAPE, catalog rules only -> rules + ML adjustment | 9.0% -> 5.7% |
| Cost | 80% range coverage | 80.6% |
| Rent | MAPE (baseline area rent/m2 x size: 24.2%) | 10.8% |
| Rent | 80% interval coverage after conformal (74.0% before) | 78.0% |
| Latency | ONNX segmentation on CPU, 512x512 | ~0.25 s |

Segmentation training: v1 = 2,000 steps on line-drawing plans (best of a 6,000-step run, ImageNet MobileNetV2
encoder); v2 = v1 fine-tuned 1,000 steps on fresh seeds that add brochure renders (room floor tones, tile grids,
wood strips). Same 300 test plans for both: kitchen IoU 0.876 -> 0.992, mIoU 0.953 -> 0.991.

## Run

Training needs PyTorch (CUDA optional); serving only needs ONNX Runtime.

```bash
pip install -e ".[ui,train]" pytest ruff      # Tesseract binary: apt install tesseract-ocr / choco install tesseract
python scripts/train_tabular.py               # rent quantile models + cost adjuster -> models/, reports/tabular.md
python scripts/train_seg.py --steps 6000      # U-Net on on-the-fly synthetic plans (MLflow: .work/mlflow.db)
python scripts/export_onnx.py                 # -> models/seg.onnx, checks parity with PyTorch
python scripts/eval_seg.py                    # -> reports/segmentation.md
python scripts/make_samples.py                # demo plans + ground truth -> samples/
python scripts/monitor_report.py              # latency, low-confidence rate, catalog age, rent drift
pytest -q

uvicorn ryfc.api:app --port 8000              # API (docs at /docs)
streamlit run app/streamlit_app.py            # UI; set RYFC_API_URL to call the API instead of in-process
```

Shareable demo link: `http://localhost:8501/?demo=plan_2.png` runs a sample plan on open.
`RYFC_WORK_DIR` moves training data, MLflow and logs off the repo drive.

### API

| Endpoint | Input | Output |
|---|---|---|
| `POST /analyze` | multipart `file` + `unit` JSON | rooms (type, m2, confidence, flag, source), annotated PNG (base64), scale, warnings, financials for all packages |
| `POST /evaluate` | `{rooms, unit}` | financials for a corrected room list |
| `POST /report` | `{result, unit}` | PDF |
| `GET /catalog`, `/monitoring`, `/health` | | price catalog + market priors, latency/low-confidence/catalog age |

`unit`: `area`, `compound_tier` (A/B/C), `unit_type`, `bua_m2` or `net_area_m2`, `floor`, `view`, `furnished`,
`amenities` (0-5), `delivery` (semi_finished / core_shell), `contractor_tier`, `purchase_price` (optional,
estimated from area prices), yield assumptions (vacancy, management fee, maintenance, discount rate, horizon).

## Layout

```
configs/catalog.yaml     finishing prices (EGP, installed, Q4 2026) per package; catalog date + monthly inflation
configs/market.yaml      area rent/sale priors, tier/type/view/finishing multipliers, yield defaults
src/ryfc/synth.py        synthetic plan generator with pixel-exact masks and room areas
src/ryfc/vision.py       load/PDF, deskew, crop, ONNX segmentation, room instances, scale, annotation
src/ryfc/ocr.py          Tesseract with wall-line removal; room-name vocabulary (EN/AR), dimension parsing
src/ryfc/cost.py         catalog rules (flooring, plaster/paint/ceiling, kitchen, bathrooms, electrical, AC, doors)
                         + CostAdjuster (LightGBM) + inflation drift
src/ryfc/rent.py         quantile LightGBM + conformal; per-prediction drivers (TreeSHAP via pred_contrib)
src/ryfc/yieldcalc.py    yield, payback, PV/NPV
src/ryfc/pipeline.py     detect() / evaluate() / analyze()
src/ryfc/api.py          FastAPI
src/ryfc/report.py       PDF (black and white)
src/ryfc/monitoring.py   JSONL events, latency summary, rent drift by area, catalog staleness
app/streamlit_app.py     UI
scripts/                 training, export, evaluation, samples, monitoring
models/                  seg.onnx, rent_q{10,50,90}.txt, cost_adj.txt + metadata
```

## Choices against architecture.md

- **U-Net over YOLOv8-seg.** Rooms are a semantic labelling problem with closed walls; per-pixel classes plus
  "door opening = wall" in the labels give closed room regions directly, and the 26 MB ONNX model runs in
  about a third of a second on CPU.
- **Tesseract over PaddleOCR.** Plan text is short printed Latin words and numbers; Tesseract plus wall-line
  removal reads it, installs in one apt line, and keeps the serving image small. PaddleOCR stays an option
  for Arabic-heavy plans.
- **LightGBM only (no CatBoost).** Categorical support is native and the error sits at the noise floor of the
  data; a second library would add weight without a measurable gain. Optuna tunes the median model.
- **SHAP via LightGBM `pred_contrib`.** Same TreeSHAP values, no extra dependency.
- **Synthetic plans instead of CubiCasa5k for now.** The 5 GB dataset could not be pulled on this machine;
  the generator gives exact areas, which CubiCasa does not (no scale), so area error is measurable.
- **LLM vision** is not used for areas (precision), as planned.

## Deploy

- Docker: `docker build -t ryfc . && docker run -p 8000:8000 ryfc` (API). UI: same image with
  `streamlit run app/streamlit_app.py`.
- Render: [render.yaml](render.yaml) defines `ryfc-api` and `ryfc-ui` (UI calls the API via `RYFC_API_URL`).
- CI: [.github/workflows/ci.yml](.github/workflows/ci.yml) runs ruff, pytest, builds the image and checks `/health`.

## Monitoring

- Rooms below 0.65 confidence are flagged in the UI/API/report for manual correction; the low-confidence
  rate per plan is logged.
- Latency per request (`/monitoring`, `scripts/monitor_report.py`).
- Rent drift: mean log residual per area vs. the training reference; >8% shift raises a flag.
- Catalog: age and implied inflation since `catalog_date`; stale after 90 days.

## Limitations and next steps

1. **Real plans.** The model has only seen generated plans. Next: pre-train on CubiCasa5k, label 200-300 local
   developer plans in CVAT (polygons per room), fine-tune, and report the same metrics on that set. Until then
   the editable room table is the safety net, and low-confidence rooms are flagged.
2. **Real prices and rents.** Catalog prices and rent priors are anchored on public asking prices; replace with
   contractor quotes / finished-job invoices (cost MAPE target) and transacted or listed rents (Property
   Finder / Bayut feeds within their terms).
3. Near-perfect synthetic scores mean the synthetic test set is saturated; the real-plan set in (1) is the
   number that matters next.
4. Non-rectangular rooms use the contour perimeter; curved walls and double-height spaces are not modelled.
