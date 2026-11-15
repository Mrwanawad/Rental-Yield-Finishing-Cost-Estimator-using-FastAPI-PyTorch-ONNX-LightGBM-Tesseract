# Segmentation and area evaluation

300 held-out synthetic plans (seeds 20000000+), ONNX Runtime CPU. Tesseract on the first 80.

| Metric | Value |
|---|---|
| mIoU (wall + 6 room classes) | 0.991 |
| IoU outside | 1.000 |
| IoU wall | 0.976 |
| IoU living | 0.998 |
| IoU bedroom | 0.996 |
| IoU kitchen | 0.992 |
| IoU bathroom | 0.994 |
| IoU balcony | 0.994 |
| IoU corridor | 0.987 |
| Room precision @IoU 0.5 | 99.8% |
| Room recall @IoU 0.5 | 99.1% |
| Mean IoU of matched rooms | 0.992 |
| Room type accuracy (model) | 100.0% |
| Room type accuracy (model + OCR labels) | 100.0% |
| Room area error, exact scale (segmentation only) | 0.6% |
| Room area error, scale from entered net area | 0.6% |
| Room area error p90, scale from net area | 1.2% |
| Plans where OCR dimensions gave a scale | 52.5% |
| Total area error, scale from OCR dimensions | 2.2% |
| ONNX CPU latency p50 (512x512) | 249 ms |

Caveat: synthetic plans share the generator's drawing style. Real developer plans need a labelled
local set (CVAT/Roboflow) and fine-tuning before these numbers transfer; see README.

## Previous model (v1, before brochure-style fine-tune), same test plans

| Metric | Value |
|---|---|
| mIoU (wall + 6 room classes) | 0.953 |
| IoU kitchen | 0.876 |
| Room precision @IoU 0.5 | 98.6% |
| Room recall @IoU 0.5 | 98.2% |
| Room type accuracy (model) | 99.4% |
| Room area error, scale from entered net area | 1.1% |

