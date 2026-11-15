"""Room names and printed dimensions from the plan via Tesseract (pytesseract).

Returns positions in the source image; callers map them to model space.
Tesseract is optional: without the binary the pipeline still runs on segmentation only.
"""

import difflib
import re
import shutil

import cv2
import numpy as np

KEYWORDS = [
    (r"^(master|bed|bedroom|bedrm|br\d?|غرفة|نوم)$", "bedroom"),
    (r"^(living|reception|recep|hall|dining|salon|family|lounge|ريسبشن|معيشة)$", "living"),
    (r"^(kitchen|kit|kitch|مطبخ)$", "kitchen"),
    (r"^(bath|bathroom|wc|w\.c|toilet|shower|powder|حمام)$", "bathroom"),
    (r"^(balcony|terrace|balc|veranda|بلكونة|تراس)$", "balcony"),
    (r"^(corridor|lobby|passage|entrance|foyer|ممر)$", "corridor"),
]
VOCAB = {"bedroom": "bedroom", "master": "bedroom", "living": "living", "reception": "living", "kitchen": "kitchen",
         "bathroom": "bathroom", "toilet": "bathroom", "balcony": "balcony", "terrace": "balcony",
         "corridor": "corridor", "lobby": "corridor", "dining": "living"}
DIMS = re.compile(r"(\d{1,2}[.,]\d{1,2})\s*[xX×*«]{1,2}\s*(\d{1,2}[.,]\d{1,2})")


def available() -> bool:
    return shutil.which("tesseract") is not None


def classify_word(word: str) -> str | None:
    w = word.lower().strip(" .:,;-_()[]|0123456789")
    for pattern, kind in KEYWORDS:
        if re.match(pattern, w):
            return kind
    if len(w) >= 5:  # OCR slips such as EEDROOM / BEOROOW
        close = difflib.get_close_matches(w, VOCAB, n=1, cutoff=0.75)
        if close:
            return VOCAB[close[0]]
    return None


def text_only(gray: np.ndarray) -> np.ndarray:
    """Binarize and erase long horizontal/vertical strokes (walls, dimension lines) so text stands alone."""
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n = max(gray.shape) // 40
    lines = (cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (n, 1)))
             | cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, n))))
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    return 255 - (ink & ~lines)


def read(gray: np.ndarray) -> tuple[list[dict], list[dict]]:
    if not available():
        return [], []
    import pytesseract

    k = 2000 / max(gray.shape)
    gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    data = pytesseract.image_to_data(text_only(gray), config="--psm 11", output_type=pytesseract.Output.DICT)
    words, lines = [], {}
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text or float(data["conf"][i]) < 30:
            continue
        cx = (data["left"][i] + data["width"][i] / 2) / k
        cy = (data["top"][i] + data["height"][i] / 2) / k
        kind = classify_word(text)
        if kind:
            words.append({"text": text, "kind": kind, "center": (cx, cy)})
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((text, cx, cy))
    dims = []
    for toks in lines.values():
        s = " ".join(t for t, _, _ in toks)
        for m in DIMS.finditer(s):
            a, b = (float(v.replace(",", ".")) for v in m.groups())
            if 0.8 <= a <= 15 and 0.8 <= b <= 15:
                dims.append({"a": a, "b": b, "center": (float(np.mean([t[1] for t in toks])),
                                                         float(np.mean([t[2] for t in toks])))})
    return words, dims
