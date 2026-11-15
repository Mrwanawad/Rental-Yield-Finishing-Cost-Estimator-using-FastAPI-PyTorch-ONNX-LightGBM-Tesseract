# Real-plan evaluation (CubiCasa5k test split)

58 plans from the CubiCasa5k test split (Finnish apartments, scanned and vector drawings), never seen in training. No metric scale exists in the dataset, so prediction and ground truth are both scaled to 100 m2 net; OCR off. Storage/garage/sauna/undefined spaces are excluded from ground truth.

| Metric | Value |
|---|---|
| Plans processed | 58 / 58 |
| Rooms per plan, ground truth / predicted (median) | 7 / 15 |
| Bedroom count exact | 24% |
| Bathroom count exact | 5% |
| Area share error, living (mean abs, pp) | 21.4 |
| Area share error, bedroom (mean abs, pp) | 17.7 |
| Area share error, kitchen (mean abs, pp) | 12.8 |
| Area share error, bathroom (mean abs, pp) | 13.9 |
| Area share error, balcony (mean abs, pp) | 16.3 |
| Area share error, corridor (mean abs, pp) | 9.3 |
| Standard finishing cost error, median abs | 49.6% |
| Standard finishing cost error, p90 abs | 164.6% |
| Plans with cost error <= 10% | 16% |
| Detect latency p50 (CPU) | 2378 ms |

Reading: the synthetic-only model does not transfer. Walls are found, but blank paper, page borders and
fixtures are taken as rooms (median 15 predicted vs 7 real), and types follow the synthetic drawing style
(a Finnish balcony with a door swing reads as kitchen). The cost engine and rent model are fine; the input
room table is what goes wrong. Fix: fine-tune on CubiCasa5k train (4,200 plans, CC BY-NC-SA 4.0) plus a small
labelled Egyptian developer set, then rerun `scripts/eval_real.py`. Until then the editable room table in the UI
is required, not optional.
