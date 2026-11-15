# Rental Yield & Finishing Cost Estimator — Architecture

## Scenario
Owners of new units (often delivered unfinished or semi-finished) ask: how much will finishing cost, and what rent can I get afterwards? Answering requires reading the floor plan, measuring rooms, pricing finishing packages, and estimating rent. The system reads a floor-plan image, detects rooms and areas, estimates finishing cost per package, and predicts rental yield.

## Input
- Floor-plan image or PDF (with scale or total area).
- Unit info: area/district, compound, unit type, size, floor, view.
- Finishing package selection (basic / standard / premium) and price catalog.
- Rental market data for training.

## Output
- Detected rooms with type and area (m²), annotated floor plan.
- Finishing cost estimate per package with breakdown (flooring, paint, kitchen, bathrooms, electrical, AC).
- Predicted monthly rent range and gross/net rental yield; payback period.
- Downloadable report.

## Architecture

```
 Floor plan upload
        |
        v
 [Pre-processing]  OpenCV: binarize, deskew, remove text noise; PDF -> image
        |
        v
 [Room segmentation]  YOLOv8-seg / U-Net (segmentation-models-pytorch)
                      classes: bedroom, living, kitchen, bathroom, balcony, corridor
        |
        v
 [Scale & area]  OCR dimensions (PaddleOCR) or user total area -> pixel-to-m² -> room areas
        |
        v
 [Cost engine]  rules: room area x package unit costs (catalog) + fixed items
                + gradient boosting adjustment from historical finishing jobs
        |
        v
 [Rent model]  LightGBM / CatBoost on area, size, rooms, finishing level, amenities
               + quantile models for rent range
        |
        v
 [Yield calculator]  gross/net yield, payback, PV
        |
        v
 [UI]  annotated plan, cost breakdown, rent range, report
```

## Tools Used
- Vision: OpenCV, Albumentations, Ultralytics YOLOv8-seg, segmentation-models-pytorch, PyTorch
- OCR: PaddleOCR
- Labeling: CVAT / Roboflow
- ML: LightGBM, CatBoost, scikit-learn, Optuna, SHAP
- Data: pandas, NumPy
- MLOps: MLflow, ONNX Runtime (inference), Docker, GitHub Actions
- Backend & UI: FastAPI, Streamlit
- Quality: pytest, ruff

## Tools Not Used (and why)
- LLM vision for exact area measurement: numeric precision too low; used only optionally for room labeling assistance.
- Single point rent estimate: quantile range instead.

## Methodologies
- Segmentation + geometric scaling; rules-plus-ML cost estimation; quantile regression for rent.
- Public floor-plan datasets (e.g. CubiCasa5k) for pre-training, fine-tune on local plans.

## Evaluation Metrics
- Segmentation: mIoU, per-class IoU, mAP50-95 (YOLO-seg).
- Area error % per room and total.
- Cost estimate MAPE vs. actual finishing quotes.
- Rent model: MAE, MAPE, interval coverage.

## Production Monitoring
- Low-confidence segmentations flagged for manual correction.
- Rent residual drift by area; catalog price updates (inflation).
- Inference latency.

## Deployment Plan
- Render: FastAPI (Docker, ONNX Runtime CPU) + Streamlit UI (or Streamlit Community Cloud calling the API).
- Demo plans: public dataset samples.
