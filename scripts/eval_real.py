"""Evaluate on real plans: CubiCasa5k test split (F1_scaled.png + model.svg room polygons).

CubiCasa has no metric scale, so both sides are scaled to the same net area (100 m2) and compared on what
drives the outputs: bedroom/bathroom counts, area share per room type, and the standard-package finishing cost.
Usage: python scripts/eval_real.py <dir with one folder per plan> [--out reports/real_plans.md]
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ryfc import cost, pipeline
from ryfc.config import ROOM_CLASSES

NET = 100.0
# CubiCasa "Space <Name>" classes -> ours; storage, garage, sauna, undefined etc. are left out
MAP = [("bed", "bedroom"), ("bath", "bathroom"), ("toilet", "bathroom"), ("kitchen", "kitchen"),
       ("living", "living"), ("dining", "living"), ("entry", "corridor"), ("hall", "corridor"),
       ("lobby", "corridor"), ("corridor", "corridor"), ("outdoor", "balcony"), ("balcony", "balcony"),
       ("terrace", "balcony")]


def gt_rooms(svg: Path) -> list[dict]:
    rooms = []
    for g in ET.parse(svg).getroot().iter("{http://www.w3.org/2000/svg}g"):
        cls = g.get("class", "")
        if not cls.startswith("Space "):
            continue
        name = cls.split()[1].lower()
        kind = next((k for key, k in MAP if key in name), None)
        poly = g.find("{http://www.w3.org/2000/svg}polygon")
        if kind is None or poly is None:
            continue
        pts = np.array([float(v) for v in re.split(r"[ ,]+", poly.get("points").strip())]).reshape(-1, 2)
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        rooms.append({"type": kind, "area_m2": area})
    total = sum(r["area_m2"] for r in rooms)
    return [r | {"area_m2": r["area_m2"] * NET / total} for r in rooms] if total else []


def shares(rooms: list[dict]) -> np.ndarray:
    return np.array([sum(r["area_m2"] for r in rooms if r["type"] == c) for c in ROOM_CLASSES]) / NET


def std_cost(rooms: list[dict]) -> float:
    rooms = [{"type": r["type"], "area_m2": r["area_m2"]} for r in rooms]  # same perimeter rule on both sides
    return cost.estimate(rooms, "standard", "semi_finished", "New Cairo", 3, "mid", pipeline.cost_adjuster())[
        "adjusted"]["estimate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    unit = pipeline.UnitInfo(net_area_m2=NET)
    rows = []
    for d in sorted(p for p in a.root.iterdir() if (p / "model.svg").exists() and (p / "F1_scaled.png").exists()):
        gt = gt_rooms(d / "model.svg")
        if not gt:
            continue
        try:
            det = pipeline.detect((d / "F1_scaled.png").read_bytes(), "plan.png", unit, use_ocr=False)
        except ValueError as e:
            rows.append({"plan": d.name, "failed": str(e)})
            continue
        pred = det["rooms"]
        n = lambda rs, t: sum(r["type"] == t for r in rs)  # noqa: E731
        rows.append({"plan": d.name, "n_gt": len(gt), "n_pred": len(pred),
                     "bed_ok": n(gt, "bedroom") == n(pred, "bedroom"),
                     "bath_ok": n(gt, "bathroom") == n(pred, "bathroom"),
                     "share_err": np.abs(shares(gt) - shares(pred)),
                     "cost_err": std_cost(pred) / std_cost(gt) - 1, "latency": det["latency_s"]})
        print(d.name, f"rooms {len(gt)}/{len(pred)}  cost err {rows[-1]['cost_err']:+.1%}", flush=True)

    ok = [r for r in rows if "failed" not in r]
    se = np.array([r["share_err"] for r in ok])
    ce = np.abs([r["cost_err"] for r in ok])
    lines = [
        "# Real-plan evaluation (CubiCasa5k test split)", "",
        f"{len(rows)} plans from the CubiCasa5k test split (Finnish apartments, scanned and vector drawings), "
        "never seen in training. No metric scale exists in the dataset, so prediction and ground truth are both "
        f"scaled to {NET:.0f} m2 net; OCR off. Storage/garage/sauna/undefined spaces are excluded from ground truth.",
        "", "| Metric | Value |", "|---|---|",
        f"| Plans processed | {len(ok)} / {len(rows)} |",
        f"| Rooms per plan, ground truth / predicted (median) | {np.median([r['n_gt'] for r in ok]):.0f} / "
        f"{np.median([r['n_pred'] for r in ok]):.0f} |",
        f"| Bedroom count exact | {np.mean([r['bed_ok'] for r in ok]):.0%} |",
        f"| Bathroom count exact | {np.mean([r['bath_ok'] for r in ok]):.0%} |",
        *[f"| Area share error, {c} (mean abs, pp) | {se[:, i].mean() * 100:.1f} |" for i, c in enumerate(ROOM_CLASSES)],
        f"| Standard finishing cost error, median abs | {np.median(ce):.1%} |",
        f"| Standard finishing cost error, p90 abs | {np.percentile(ce, 90):.1%} |",
        f"| Plans with cost error <= 10% | {np.mean(ce <= 0.10):.0%} |",
        f"| Detect latency p50 (CPU) | {np.median([r['latency'] for r in ok]) * 1000:.0f} ms |",
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        a.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
