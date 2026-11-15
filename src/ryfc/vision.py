"""Floor plan vision: load/pre-process, ONNX segmentation, room instances, scale, annotation."""

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from ryfc.config import CLASS_ID, CLASSES, MODELS, ROOM_CLASSES, SEG_SIZE

LOW_CONF = 0.65
MIN_ROOM_PX = 120
MIN_ROOM_SHARE = 0.007
PALETTE = {"living": (122, 170, 110), "bedroom": (110, 140, 190), "kitchen": (214, 160, 90),
           "bathroom": (170, 120, 170), "balcony": (120, 180, 180), "corridor": (170, 170, 160)}


@dataclass
class Plan:
    original: np.ndarray  # cropped gray at source resolution (for OCR)
    gray: np.ndarray  # SEG_SIZE x SEG_SIZE model input
    factor: float  # SEG_SIZE pixels per source pixel
    pad: tuple[int, int]  # (x, y) offset of the crop inside the square canvas, model pixels


def load_image(data: bytes, filename: str = "") -> np.ndarray:
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import pypdfium2 as pdfium  # ~10 MB vs ~100 MB for PyMuPDF: keeps the serverless bundle small

        try:
            page = pdfium.PdfDocument(data)[0]
        except pdfium.PdfiumError as e:
            raise ValueError("Unreadable PDF") from e
        return np.array(page.render(scale=200 / 72, grayscale=True).to_pil().convert("L"))
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Unsupported image format")
    return img


def deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate by the dominant near-axis line angle (|angle| < 10 deg)."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 80, minLineLength=max(gray.shape) // 8, maxLineGap=5)
    if lines is None:
        return gray
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):  # shape differs across OpenCV versions
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        a = (a + 45) % 90 - 45
        if abs(a) < 10:
            angles.append(a)
    if not angles:
        return gray
    ang = float(np.median(angles))
    if abs(ang) < 0.3:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def preprocess(gray: np.ndarray) -> Plan:
    if gray.mean() < 110:  # white-on-dark plans
        gray = 255 - gray
    gray = deskew(gray)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.nonzero(ink)
    if len(xs):
        m = int(0.03 * max(gray.shape))
        x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, gray.shape[1])
        y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, gray.shape[0])
        gray = gray[y0:y1, x0:x1]
    h, w = gray.shape
    # synthetic training plans occupy ~70-90% of the canvas: keep a margin
    factor = SEG_SIZE * 0.9 / max(h, w)
    small = cv2.resize(gray, (max(1, round(w * factor)), max(1, round(h * factor))), interpolation=cv2.INTER_AREA)
    # pad with the paper colour so the crop edge does not look like a wall
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    canvas = np.full((SEG_SIZE, SEG_SIZE), int(np.median(border)), np.uint8)
    px, py = (SEG_SIZE - small.shape[1]) // 2, (SEG_SIZE - small.shape[0]) // 2
    canvas[py:py + small.shape[0], px:px + small.shape[1]] = small
    return Plan(gray, canvas, factor, (px, py))


class Segmenter:
    def __init__(self, path: Path | None = None):
        path = path or MODELS / "seg.onnx"
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def probs(self, gray: np.ndarray) -> np.ndarray:
        x = ((255 - gray).astype(np.float32) / 255.0)[None, None]
        logits = self.session.run(None, {"image": x})[0][0]
        e = np.exp(logits - logits.max(0, keepdims=True))
        return e / e.sum(0, keepdims=True)


def rooms_from_probs(probs: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Free space (not wall / outside) split into walled components; each typed by probability vote.
    A component with a large second class (open-plan corridor + living) is split by class."""
    pred = probs.argmax(0)
    free = np.isin(pred, [CLASS_ID[c] for c in ROOM_CLASSES]).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, comp = cv2.connectedComponents(free, connectivity=4)
    inst = np.zeros(pred.shape, np.int32)
    rooms = []
    edge = np.zeros(pred.shape, bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    for k in range(1, n):
        region = comp == k
        if region.sum() < MIN_ROOM_PX or (region & edge).any():  # rooms never touch the image border
            continue
        votes = probs[2:, region].sum(1)
        shares = votes / votes.sum()
        parts = [(c, region & (pred == c + 2)) for c in np.argsort(-shares) if shares[c] >= 0.2]
        if len(parts) > 1 and all(p.sum() >= MIN_ROOM_PX * 2 for _, p in parts):
            assigned = np.zeros_like(region)
            for _, p in parts:
                assigned |= p
            # leftover pixels follow the majority class
            parts[0] = (parts[0][0], parts[0][1] | (region & ~assigned))
            pieces = parts
        else:
            pieces = [(int(np.argmax(shares)), region)]
        for c, p in pieces:
            m2, cc = cv2.connectedComponents(p.astype(np.uint8), connectivity=4)
            for j in range(1, m2):
                sub = cc == j
                if sub.sum() < MIN_ROOM_PX:
                    continue
                cls = ROOM_CLASSES[c]
                conf = float(probs[c + 2, sub].mean())
                cnts, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                ys, xs = np.nonzero(sub)
                rid = len(rooms) + 1
                inst[sub] = rid
                rooms.append({"id": rid, "type": cls, "model_type": cls, "pixels": int(sub.sum()),
                              "perimeter_px": float(cv2.arcLength(cnt, True)), "confidence": conf,
                              "bbox": [x, y, x + w, y + h], "centroid": [float(xs.mean()), float(ys.mean())],
                              "fill_ratio": float(sub.sum() / max(w * h, 1)), "source": "model"})
    # drop slivers (< 0.7% of the plan, ~1.2 m2 on a 170 m2 unit) and renumber
    total = sum(r["pixels"] for r in rooms)
    keep = [r for r in rooms if r["pixels"] >= max(MIN_ROOM_PX, MIN_ROOM_SHARE * total)]
    remap = np.zeros(len(rooms) + 1, np.int32)
    for new_id, r in enumerate(keep, 1):
        remap[r["id"]] = new_id
        r["id"] = new_id
    return keep, remap[inst]


def apply_ocr(rooms: list[dict], inst: np.ndarray, words: list[dict], dims: list[dict]) -> None:
    """OCR room names override the model type; OCR dimensions attach to the room they sit in."""
    by_id = {r["id"]: r for r in rooms}
    for w in words:
        x, y = (int(round(v)) for v in w["center"])
        if 0 <= y < inst.shape[0] and 0 <= x < inst.shape[1] and inst[y, x] in by_id:
            r = by_id[inst[y, x]]
            r["ocr_label"] = w["text"]
            if w["kind"] != r["type"]:
                r["type"], r["source"] = w["kind"], "ocr"
            r["confidence"] = max(r["confidence"], 0.9)
    for d in dims:
        x, y = (int(round(v)) for v in d["center"])
        if 0 <= y < inst.shape[0] and 0 <= x < inst.shape[1] and inst[y, x] in by_id:
            by_id[inst[y, x]]["ocr_dims"] = [d["a"], d["b"]]


def scale_from_dims(rooms: list[dict]) -> float | None:
    """Pixels per metre from rooms with printed interior dimensions (rectangular rooms only)."""
    est = []  # (px per metre, weight)
    for r in rooms:
        d = r.get("ocr_dims")
        if not d or r["fill_ratio"] < 0.85 or d[0] * d[1] < 1:
            continue
        w, h = r["bbox"][2] - r["bbox"][0], r["bbox"][3] - r["bbox"][1]
        shape, printed = max(w, h) / max(min(w, h), 1), max(d) / min(d)
        if abs(np.log(shape / printed)) > 0.25:  # label belongs to another room or was misread
            continue
        est.append((np.sqrt(r["pixels"] / (d[0] * d[1])), r["pixels"]))
    if not est:
        return None
    # area-weighted consensus: the estimate with most support within +-8%, then average its cluster
    support = [sum(w for e, w in est if abs(e / c - 1) < 0.08) for c, _ in est]
    center = est[int(np.argmax(support))][0]
    cluster = [(e, w) for e, w in est if abs(e / center - 1) < 0.08]
    return float(sum(e * w for e, w in cluster) / sum(w for _, w in cluster))


def apply_scale(rooms: list[dict], ppm: float) -> None:
    for r in rooms:
        r["area_m2"] = r["pixels"] / ppm**2
        r["perimeter_m"] = r["perimeter_px"] / ppm
        r["flagged"] = r["confidence"] < LOW_CONF


def _font(size):
    for name in ("timesbd.ttf", "Times New Roman Bold.ttf", "LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(gray: np.ndarray, inst: np.ndarray, rooms: list[dict], out_size: int = 1024) -> bytes:
    k = out_size / gray.shape[0]
    base = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB).astype(np.float32)
    big = cv2.resize(inst.astype(np.uint16), (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    for r in rooms:
        m = big == r["id"]
        rgb[m] = rgb[m] * 0.55 + np.array(PALETTE[r["type"]], np.float32) * 0.45
    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    f = _font(max(12, out_size // 60))
    for r in rooms:
        cnts, _ = cv2.findContours((big == r["id"]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (180, 30, 30) if r.get("flagged") else (40, 40, 40)
        for c in cnts:
            d.line([tuple(p[0]) for p in c] + [tuple(c[0][0])], fill=color, width=3 if r.get("flagged") else 1)
        cx, cy = r["centroid"][0] * k, r["centroid"][1] * k
        text = f"{r['id']}. {r['type'].title()}\n{r.get('area_m2', 0):.1f} m2" + (" ?" if r.get("flagged") else "")
        bb = d.multiline_textbbox((cx, cy), text, font=f, anchor="mm", align="center")
        d.rectangle([bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2], fill=(255, 255, 255))
        d.multiline_text((cx, cy), text, font=f, fill=color, anchor="mm", align="center")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


__all__ = ["CLASSES", "Plan", "Segmenter", "annotate", "apply_ocr", "apply_scale", "load_image",
           "preprocess", "rooms_from_probs", "scale_from_dims"]
