"""Evaluate segmentation + room extraction + scale on held-out synthetic plans -> reports/segmentation.md

Pixel: per-class IoU, mIoU. Rooms: precision/recall at IoU>=0.5, type accuracy (model vs model+OCR),
area error % per room and total (scale from entered net area, and from OCR'd dimensions).
"""

import argparse
import time

import cv2
import numpy as np

from ryfc import ocr, vision
from ryfc.config import CLASS_ID, CLASSES, REPORTS, ROOM_CLASSES
from ryfc.synth import generate
from ryfc.tracking import run

SEED0 = 20_000_000


def pct_or_na(v):
    return "n/a" if v is None else f"{v:.1f}%"


def gt_instances(mask):
    inst = np.zeros(mask.shape, np.int32)
    out = []
    for c in ROOM_CLASSES:
        n, cc = cv2.connectedComponents((mask == CLASS_ID[c]).astype(np.uint8), connectivity=4)
        for k in range(1, n):
            m = cc == k
            if m.sum() >= vision.MIN_ROOM_PX:
                out.append({"type": c, "id": len(out) + 1, "pixels": int(m.sum())})
                inst[m] = len(out)
    return out, inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ocr-n", type=int, default=60, help="samples to run Tesseract on (slow)")
    args = ap.parse_args()
    seg = vision.Segmenter()
    k = len(CLASSES)
    conf = np.zeros((k, k), np.int64)
    tp = fp = fn = 0
    type_ok = type_ok_ocr = matched = matched_ocr_eval = 0
    area_err, area_err_netscale, total_err_dims, ious = [], [], [], []
    dims_found = 0
    lat = []
    for i in range(args.n):
        s = generate(SEED0 + i)
        t = time.perf_counter()
        probs = seg.probs(s.image)
        lat.append(time.perf_counter() - t)
        pred = probs.argmax(0)
        conf += np.bincount(s.mask.ravel().astype(np.int64) * k + pred.ravel(), minlength=k * k).reshape(k, k)
        rooms, inst = vision.rooms_from_probs(probs)
        model_types = {r["id"]: r["type"] for r in rooms}
        use_ocr = i < args.ocr_n and ocr.available()
        if use_ocr:
            words, dims = ocr.read(s.image)
            vision.apply_ocr(rooms, inst, words, dims)
            ppm_d = vision.scale_from_dims(rooms)
            if ppm_d:
                dims_found += 1
                total_err_dims.append(abs(sum(r["pixels"] for r in rooms) / ppm_d**2 - s.net_area_m2) / s.net_area_m2)
        ppm_net = (sum(r["pixels"] for r in rooms) / s.net_area_m2) ** 0.5
        gts, ginst = gt_instances(s.mask)
        used = set()
        for g in gts:
            gm = ginst == g["id"]
            cand = np.bincount(inst[gm], minlength=len(rooms) + 1)
            cand[0] = 0
            j = int(cand.argmax())
            if j == 0:
                fn += 1
                continue
            pm = inst == j
            iou = (gm & pm).sum() / (gm | pm).sum()
            if iou < 0.5 or j in used:
                fn += 1
                continue
            used.add(j)
            tp += 1
            ious.append(iou)
            r = rooms[j - 1]
            matched += 1
            type_ok += model_types[j] == g["type"]
            if use_ocr:
                matched_ocr_eval += 1
                type_ok_ocr += r["type"] == g["type"]
            true_a = g["pixels"] / s.ppm**2
            area_err.append(abs(r["pixels"] / s.ppm**2 - true_a) / true_a)
            area_err_netscale.append(abs(r["pixels"] / ppm_net**2 - true_a) / true_a)
        fp += len(rooms) - len(used)
    inter = np.diag(conf)
    iou_c = inter / np.maximum(conf.sum(0) + conf.sum(1) - inter, 1)
    miou = float(iou_c[1:].mean())
    m = {"miou": miou, "room_precision": tp / max(tp + fp, 1), "room_recall": tp / max(tp + fn, 1),
         "mean_matched_iou": float(np.mean(ious)), "type_acc_model": type_ok / max(matched, 1),
         "type_acc_model_ocr": type_ok_ocr / max(matched_ocr_eval, 1) if matched_ocr_eval else None,
         "room_area_mape_true_scale": float(np.mean(area_err) * 100),
         "room_area_mape_net_scale": float(np.mean(area_err_netscale) * 100),
         "room_area_p90_err_net_scale": float(np.quantile(area_err_netscale, 0.9) * 100),
         "ocr_dims_scale_rate": dims_found / max(min(args.ocr_n, args.n), 1) if ocr.available() else None,
         "total_area_mape_ocr_dims": float(np.mean(total_err_dims) * 100) if total_err_dims else None,
         "onnx_cpu_ms_p50": float(np.median(lat) * 1000)}
    with run("segmentation-eval", vars(args)) as log:
        log({k2: v for k2, v in m.items() if v is not None})
    fmt = lambda v, p=False: "n/a" if v is None else (f"{v * 100:.1f}%" if p else f"{v:.3f}")  # noqa: E731
    lines = [
        "# Segmentation and area evaluation", "",
        f"{args.n} held-out synthetic plans (seeds {SEED0}+), ONNX Runtime CPU. Tesseract on the first {args.ocr_n}.", "",
        "| Metric | Value |", "|---|---|",
        f"| mIoU (wall + 6 room classes) | {fmt(miou)} |",
        *[f"| IoU {c} | {fmt(v)} |" for c, v in zip(CLASSES, iou_c)],
        f"| Room precision @IoU 0.5 | {fmt(m['room_precision'], True)} |",
        f"| Room recall @IoU 0.5 | {fmt(m['room_recall'], True)} |",
        f"| Mean IoU of matched rooms | {fmt(m['mean_matched_iou'])} |",
        f"| Room type accuracy (model) | {fmt(m['type_acc_model'], True)} |",
        f"| Room type accuracy (model + OCR labels) | {fmt(m['type_acc_model_ocr'], True)} |",
        f"| Room area error, exact scale (segmentation only) | {m['room_area_mape_true_scale']:.1f}% |",
        f"| Room area error, scale from entered net area | {m['room_area_mape_net_scale']:.1f}% |",
        f"| Room area error p90, scale from net area | {m['room_area_p90_err_net_scale']:.1f}% |",
        f"| Plans where OCR dimensions gave a scale | {fmt(m['ocr_dims_scale_rate'], True)} |",
        f"| Total area error, scale from OCR dimensions | {pct_or_na(m['total_area_mape_ocr_dims'])} |",
        f"| ONNX CPU latency p50 (512x512) | {m['onnx_cpu_ms_p50']:.0f} ms |", "",
        "Caveat: synthetic plans share the generator's drawing style. Real developer plans need a labelled",
        "local set (CVAT/Roboflow) and fine-tuning before these numbers transfer; see README.", "",
    ]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "segmentation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
