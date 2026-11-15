"""Synthetic Egyptian-style apartment floor plans with pixel-exact ground truth.

Each sample is a rendered plan (walls, doors, windows, furniture, labels, dimensions, noise)
plus a class mask and per-room areas in m2. Used to pre-train segmentation and to measure
area error, because real labelled local plans are not available yet.
Layout: night band (bedrooms + bathrooms) / corridor / day band (living + kitchen + guest WC),
optional balconies, random flips and rotation.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from ryfc.config import CLASS_ID, SEG_SIZE

LABELS = {
    "living": ["LIVING", "RECEPTION", "LIVING & DINING", "Reception", "Living Room", "HALL"],
    "bedroom": ["BEDROOM", "BED ROOM", "MASTER BEDROOM", "Bedroom", "Master Bed", "BEDROOM 2"],
    "kitchen": ["KITCHEN", "Kitchen", "KIT."],
    "bathroom": ["BATH", "BATHROOM", "WC", "Toilet", "Bath", "W.C"],
    "balcony": ["BALCONY", "TERRACE", "Balcony"],
    "corridor": ["CORRIDOR", "LOBBY", "Corridor", ""],
}
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_PLAIN]


@dataclass
class Room:
    kind: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0


@dataclass
class Sample:
    image: np.ndarray  # HxW uint8 gray
    mask: np.ndarray  # HxW uint8 class ids
    rooms: list[dict] = field(default_factory=list)
    ppm: float = 0.0  # pixels per metre
    net_area_m2: float = 0.0


def _layout(rng: np.random.Generator) -> tuple[list[Room], Room]:
    area = rng.uniform(75, 260)
    n_bed = 2 if area < 100 else int(rng.integers(2, 4)) if area < 150 else 3 if area < 200 else int(rng.integers(3, 5))
    n_bath = 1 if n_bed == 2 and area < 120 else 2 if n_bed < 4 else 3
    short = np.sqrt(area / rng.uniform(1.2, 1.8))
    long_ = area / short
    d_night = min(rng.uniform(3.4, 4.6), short * 0.45)
    cw = rng.uniform(1.1, 1.5)
    if short - d_night - cw < 3.5:
        short = d_night + cw + 3.5
    rooms: list[Room] = []

    # night band: bedrooms and bathrooms side by side
    bath_w = [rng.uniform(1.8, 2.6) for _ in range(n_bath - 1 if area > 140 else n_bath)]
    guest_wc = len(bath_w) < n_bath
    bed_w = max((long_ - sum(bath_w)) / n_bed, 2.8)
    long_ = max(long_, bed_w * n_bed + sum(bath_w))
    seq = ["bedroom"] * n_bed
    for _ in range(len(bath_w)):
        seq.insert(int(rng.integers(1, len(seq))), "bathroom")
    x, bi = 0.0, 0
    for kind in seq:
        w = bath_w[bi] if kind == "bathroom" else bed_w
        if kind == "bathroom":
            bi += 1
            bd = min(rng.uniform(2.0, 2.6), d_night - 0.9)
            rooms.append(Room("bathroom", x, 0, x + w, bd))
            rooms.append(Room("corridor", x, bd, x + w, d_night))  # lobby in front of bath
        else:
            rooms.append(Room("bedroom", x, 0, x + w, d_night))
        x += w

    # corridor + day band
    kw = rng.uniform(2.8, 4.0)
    xl = long_ - kw - (rng.uniform(1.6, 2.2) if guest_wc else 0)
    xl = max(xl, 4.0)
    rooms.append(Room("living", 0, d_night, xl, short))
    rooms.append(Room("corridor", xl, d_night, long_, d_night + cw))
    if guest_wc:
        rooms.append(Room("kitchen", xl, d_night + cw, xl + kw, short))
        rooms.append(Room("bathroom", xl + kw, d_night + cw, long_, short))
    else:
        rooms.append(Room("kitchen", xl, d_night + cw, long_, short))
    footprint = Room("footprint", 0, 0, long_, short)

    # balconies outside the footprint
    if rng.random() < 0.8:
        bw = rng.uniform(3.0, min(xl, 6.5))
        bx = rng.uniform(0, xl - bw)
        rooms.append(Room("balcony", bx, short, bx + bw, short + rng.uniform(1.4, 2.2)))
    if rng.random() < 0.3:
        bed = rooms[0]
        rooms.append(Room("balcony", bed.x0 + 0.3, -rng.uniform(1.2, 1.8), bed.x1 - 0.3, 0))
    return rooms, footprint


def _transform(rooms, footprint, rng):
    fx, fy, tr = rng.random() < 0.5, rng.random() < 0.5, rng.random() < 0.5
    out = []
    for r in rooms + [footprint]:
        x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
        if fx:
            x0, x1 = -x1, -x0
        if fy:
            y0, y1 = -y1, -y0
        if tr:
            x0, y0, x1, y1 = y0, x0, y1, x1
        out.append(Room(r.kind, x0, y0, x1, y1))
    mx = min(r.x0 for r in out)
    my = min(r.y0 for r in out)
    for r in out:
        r.x0, r.x1, r.y0, r.y1 = r.x0 - mx, r.x1 - mx, r.y0 - my, r.y1 - my
    return out[:-1], out[-1]


def _shared_edge(a: Room, b: Room, eps=1e-6):
    """Shared wall segment between two rectangles: ('h', y, x0, x1) or ('v', x, y0, y1)."""
    for ya in (a.y0, a.y1):
        for yb in (b.y0, b.y1):
            if abs(ya - yb) < eps:
                lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
                if hi - lo > 0:
                    return ("h", ya, lo, hi)
    for xa in (a.x0, a.x1):
        for xb in (b.x0, b.x1):
            if abs(xa - xb) < eps:
                lo, hi = max(a.y0, b.y0), min(a.y1, b.y1)
                if hi - lo > 0:
                    return ("v", xa, lo, hi)
    return None


DOOR_TARGETS = {
    "bedroom": ["corridor", "living"],
    "bathroom": ["corridor", "living", "bedroom", "kitchen"],
    "kitchen": ["living", "corridor"],
    "balcony": ["living", "bedroom"],
    "corridor": ["living", "corridor"],
}


def generate(rng: np.random.Generator | int | None = None, size: int = SEG_SIZE) -> Sample:
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    rooms, fp = _transform(*_layout(rng), rng)
    ext_w = max(r.x1 for r in rooms + [fp])
    ext_h = max(r.y1 for r in rooms + [fp])
    margin = rng.uniform(28, 60)
    ppm = min((size - 2 * margin) / ext_w, (size - 2 * margin) / ext_h) * rng.uniform(0.8, 1.0)
    ox = rng.uniform(margin, size - margin - ext_w * ppm)
    oy = rng.uniform(margin, size - margin - ext_h * ppm)

    def P(x, y):
        return int(round(ox + x * ppm)), int(round(oy + y * ppm))

    t_int = max(2, int(round(rng.uniform(0.1, 0.16) * ppm)))
    t_ext = max(t_int + 1, int(round(rng.uniform(0.2, 0.3) * ppm)))
    t_rail = max(1, int(round(0.05 * ppm)))

    mask = np.zeros((size, size), np.uint8)
    wall = np.zeros((size, size), np.uint8)
    rail = np.zeros((size, size), np.uint8)
    for r in rooms:
        cv2.rectangle(mask, P(r.x0, r.y0), P(r.x1, r.y1), CLASS_ID[r.kind], -1)
    for r in rooms:
        target = rail if r.kind == "balcony" else wall
        cv2.rectangle(target, P(r.x0, r.y0), P(r.x1, r.y1), 255, t_rail if r.kind == "balcony" else t_int)
    cv2.rectangle(wall, P(fp.x0, fp.y0), P(fp.x1, fp.y1), 255, t_ext)
    solid = cv2.bitwise_or(wall, rail)
    openings = np.zeros_like(wall)
    symbols = []  # deferred thin-line drawings (doors, windows)

    def on_boundary(edge):
        o, c, lo, hi = edge
        if o == "h":
            return abs(c - fp.y0) < 1e-6 or abs(c - fp.y1) < 1e-6
        return abs(c - fp.x0) < 1e-6 or abs(c - fp.x1) < 1e-6

    def cut(edge, a, b, thick):
        o, c, _, _ = edge
        half = thick / 2 + 1.5
        if o == "h":
            p0, p1 = P(a, c), P(b, c)
            cv2.rectangle(openings, (p0[0], int(p0[1] - half)), (p1[0], int(p1[1] + half)), 255, -1)
        else:
            p0, p1 = P(c, a), P(c, b)
            cv2.rectangle(openings, (int(p0[0] - half), p0[1]), (int(p1[0] + half), p1[1]), 255, -1)

    # doors
    for r in rooms:
        for tk in DOOR_TARGETS.get(r.kind, []):
            nbrs = [n for n in rooms if n is not r and n.kind == tk]
            edges = [e for n in nbrs if (e := _shared_edge(r, n)) and e[3] - e[2] > 1.2]
            if not edges:
                continue
            e = edges[int(rng.integers(len(edges)))]
            o, c, lo, hi = e
            open_plan = r.kind == "corridor"
            dw = min(hi - lo - 0.2, rng.uniform(1.0, 2.0)) if open_plan else (0.8 if r.kind == "bathroom" else 0.9)
            a = rng.uniform(lo + 0.1, hi - dw - 0.1) if hi - lo - dw > 0.2 else lo + 0.05
            cut(e, a, a + dw, t_int)
            if not open_plan:
                rc = (r.y0 + r.y1) / 2 if o == "h" else (r.x0 + r.x1) / 2
                symbols.append(("door", o, c, a, dw, 1 if rc > c else -1))
            break
    living = next(r for r in rooms if r.kind == "living")
    ext_edges = [e for e in _edges(living) if on_boundary(e) and e[3] - e[2] > 1.5]
    if ext_edges:  # entrance
        e = ext_edges[int(rng.integers(len(ext_edges)))]
        a = rng.uniform(e[2] + 0.2, e[3] - 1.2)
        cut(e, a, a + 1.0, t_ext)
        mid = (living.y0 + living.y1) / 2 if e[0] == "h" else (living.x0 + living.x1) / 2
        symbols.append(("door", e[0], e[1], a, 1.0, 1 if mid > e[1] else -1))
    # windows on exterior walls
    for r in rooms:
        if r.kind in ("corridor", "balcony"):
            continue
        for e in _edges(r):
            if not on_boundary(e) or e[3] - e[2] < 1.2 or rng.random() < 0.35:
                continue
            ww = min(e[3] - e[2] - 0.6, 0.6 if r.kind == "bathroom" else rng.uniform(1.0, 2.2))
            a = rng.uniform(e[2] + 0.3, e[3] - ww - 0.3) if e[3] - e[2] - ww > 0.6 else e[2] + 0.3
            cut(e, a, a + ww, t_ext)
            symbols.append(("window", e[0], e[1], a, ww, 0))

    mask[solid > 0] = CLASS_ID["wall"]  # door/window openings stay wall: rooms are closed regions
    drawn_wall = cv2.bitwise_and(solid, cv2.bitwise_not(openings))

    # render
    img = np.full((size, size), 255, np.uint8)
    if rng.random() < 0.4:  # brochure-style floor finishes
        _floor_fills(img, rooms, P, ppm, rng)
    style = rng.integers(4)
    ink = int(rng.integers(0, 60))
    if style == 0:
        img[drawn_wall > 0] = ink
    elif style == 1:
        cnts, _ = cv2.findContours(drawn_wall, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, ink, 1)
    elif style == 2:
        img[drawn_wall > 0] = int(rng.integers(90, 160))
        cnts, _ = cv2.findContours(drawn_wall, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, ink, 1)
    else:
        hatch = np.zeros_like(img)
        for k in range(-size, size, 4):
            cv2.line(hatch, (k, 0), (k + size, size), 255, 1)
        img[(drawn_wall > 0) & (hatch > 0)] = ink
        cnts, _ = cv2.findContours(drawn_wall, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, ink, 1)

    line = int(rng.integers(40, 120))
    for kind, o, c, a, w, sign in symbols:
        if kind == "door":
            if o == "h":
                hinge = P(a, c)
                leaf = P(a, c + sign * w)
                ang = (0, 90) if sign > 0 else (270, 360)
            else:
                hinge = P(c, a)
                leaf = P(c + sign * w, a)
                ang = (0, 90) if sign > 0 else (90, 180)
            r_px = int(w * ppm)
            cv2.line(img, hinge, leaf, line, 1, cv2.LINE_AA)
            cv2.ellipse(img, hinge, (r_px, r_px), 0, ang[0], ang[1], line, 1, cv2.LINE_AA)
        else:
            for off in (-t_ext / 2, 0, t_ext / 2):
                if o == "h":
                    p0, p1 = P(a, c), P(a + w, c)
                    p0, p1 = (p0[0], int(p0[1] + off)), (p1[0], int(p1[1] + off))
                else:
                    p0, p1 = P(c, a), P(c, a + w)
                    p0, p1 = (int(p0[0] + off), p0[1]), (int(p1[0] + off), p1[1])
                cv2.line(img, p0, p1, ink, 1)

    for r in rooms:
        if rng.random() < 0.75:
            _furniture(img, r, P, ppm, rng, line)

    # labels, dimensions
    font = FONTS[int(rng.integers(len(FONTS)))]
    out_rooms = []
    for r in rooms:
        (x0, y0), (x1, y1) = P(r.x0, r.y0), P(r.x1, r.y1)
        inside = mask[y0:y1 + 1, x0:x1 + 1] == CLASS_ID[r.kind]
        px = int(inside.sum())
        # printed dimensions are clear internal dimensions, measured face to face of walls
        cols, rows = np.nonzero(inside.any(0))[0], np.nonzero(inside.any(1))[0]
        iw = (cols[-1] - cols[0] + 1) / ppm if len(cols) else r.w
        ih = (rows[-1] - rows[0] + 1) / ppm if len(rows) else r.h
        out_rooms.append({"type": r.kind, "area_m2": px / ppm**2, "bbox": [x0, y0, x1, y1],
                          "dims": (round(iw, 2), round(ih, 2))})
        if rng.random() < 0.8:
            label = LABELS[r.kind][int(rng.integers(len(LABELS[r.kind])))]
            dims = f"{iw:.2f} x {ih:.2f}" if rng.random() < 0.55 and r.kind != "corridor" else ""
            _put_text(img, [s for s in (label, dims) if s], ((x0 + x1) // 2, (y0 + y1) // 2),
                      font, ppm, x1 - x0, ink)
    if rng.random() < 0.5:
        _dimension_chain(img, fp, P, ppm, ink, font)

    img, mask = _degrade(img, mask, rng)
    net = sum(r["area_m2"] for r in out_rooms)
    return Sample(img, mask, out_rooms, ppm, net)


def _edges(r: Room):
    return [("h", r.y0, r.x0, r.x1), ("h", r.y1, r.x0, r.x1),
            ("v", r.x0, r.y0, r.y1), ("v", r.x1, r.y0, r.y1)]


def _floor_fills(img, rooms: list[Room], P, ppm, rng):
    """Light tone per room type, tile grids in wet rooms, wood strips in bedrooms (marketing plans)."""
    tones = {k: int(rng.integers(205, 250)) for k in LABELS}
    for r in rooms:
        (x0, y0), (x1, y1) = P(r.x0, r.y0), P(r.x1, r.y1)
        cv2.rectangle(img, (x0, y0), (x1, y1), tones[r.kind], -1)
        pattern = rng.random()
        shade = tones[r.kind] - int(rng.integers(15, 45))
        if r.kind in ("bathroom", "kitchen", "balcony") and pattern < 0.6:
            step = max(3, int(rng.uniform(0.3, 0.6) * ppm))
            for x in range(x0, x1, step):
                cv2.line(img, (x, y0), (x, y1), shade, 1)
            for y in range(y0, y1, step):
                cv2.line(img, (x0, y), (x1, y), shade, 1)
        elif r.kind in ("bedroom", "living") and pattern < 0.4:
            step = max(3, int(rng.uniform(0.15, 0.25) * ppm))
            horizontal = rng.random() < 0.5
            for k in range(y0 if horizontal else x0, y1 if horizontal else x1, step):
                if horizontal:
                    cv2.line(img, (x0, k), (x1, k), shade, 1)
                else:
                    cv2.line(img, (k, y0), (k, y1), shade, 1)


def _put_text(img, lines, center, font, ppm, max_w, color):
    if not lines:
        return
    scale = float(np.clip(ppm / 50, 0.28, 0.6)) * (1.3 if font == cv2.FONT_HERSHEY_PLAIN else 1)
    for _ in range(4):
        widths = [cv2.getTextSize(s, font, scale, 1)[0][0] for s in lines]
        if max(widths) < max_w * 0.92:
            break
        scale *= 0.8
    h = cv2.getTextSize("A", font, scale, 1)[0][1]
    y = center[1] - (len(lines) - 1) * (h + 4) // 2 + h // 2
    for s, w in zip(lines, widths):
        cv2.putText(img, s, (center[0] - w // 2, y), font, scale, color, 1, cv2.LINE_AA)
        y += h + 5


def _furniture(img, r: Room, P, ppm, rng, c):
    m = 0.25
    if r.w < 1.2 or r.h < 1.2:
        return
    if r.kind == "bedroom":
        bw, bl = (1.6, 2.0) if r.w > 3.2 and rng.random() < 0.6 else (1.1, 1.9)
        bx = r.x0 + (r.w - bw) / 2
        cv2.rectangle(img, P(bx, r.y0 + m), P(bx + bw, r.y0 + m + bl), c, 1)
        cv2.rectangle(img, P(bx + 0.1, r.y0 + m + 0.1), P(bx + bw - 0.1, r.y0 + m + 0.45), c, 1)
        cv2.rectangle(img, P(r.x0 + m, r.y1 - m - 0.6), P(min(r.x1 - m, r.x0 + m + 1.8), r.y1 - m), c, 1)
    elif r.kind == "living":
        sx, sy = r.x0 + m + 0.3, r.y1 - m - 0.9
        cv2.rectangle(img, P(sx, sy), P(sx + min(2.4, r.w - 1), sy + 0.9), c, 1)
        cx, cy = (r.x0 + r.x1) / 2 + 0.8, r.y0 + r.h * 0.35
        cv2.circle(img, P(cx, cy), int(0.6 * ppm), c, 1, cv2.LINE_AA)
        for k in range(4):
            a = k * np.pi / 2
            cv2.rectangle(img, P(cx + 0.85 * np.cos(a) - 0.2, cy + 0.85 * np.sin(a) - 0.2),
                          P(cx + 0.85 * np.cos(a) + 0.2, cy + 0.85 * np.sin(a) + 0.2), c, 1)
    elif r.kind == "kitchen":
        cv2.rectangle(img, P(r.x0 + 0.1, r.y0 + 0.1), P(r.x1 - 0.1, r.y0 + 0.7), c, 1)
        cv2.rectangle(img, P(r.x0 + 0.1, r.y0 + 0.1), P(r.x0 + 0.7, r.y1 - 0.1), c, 1)
        for k in range(4):
            cv2.circle(img, P(r.x0 + 1.4 + 0.3 * (k % 2), r.y0 + 0.25 + 0.25 * (k // 2)), max(1, int(0.08 * ppm)), c, 1)
        cv2.rectangle(img, P(r.x1 - 1.2, r.y0 + 0.2), P(r.x1 - 0.5, r.y0 + 0.6), c, 1)
    elif r.kind == "bathroom":
        cv2.ellipse(img, P(r.x0 + 0.5, r.y0 + 0.45), (max(2, int(0.18 * ppm)), max(2, int(0.25 * ppm))), 0, 0, 360, c, 1)
        cv2.rectangle(img, P(r.x0 + 0.3, r.y0 + 0.1), P(r.x0 + 0.7, r.y0 + 0.2), c, 1)
        cv2.ellipse(img, P(r.x1 - 0.45, r.y0 + 0.4), (max(2, int(0.22 * ppm)), max(2, int(0.15 * ppm))), 0, 0, 360, c, 1)
        if r.h > 2.0:
            p0, p1 = P(r.x0 + 0.1, r.y1 - 0.9), P(min(r.x1 - 0.1, r.x0 + 1.7), r.y1 - 0.1)
            cv2.rectangle(img, p0, p1, c, 1)
            cv2.line(img, p0, p1, c, 1)
    elif r.kind == "balcony":
        for k in range(int(rng.integers(1, 4))):
            cv2.circle(img, P(r.x0 + 0.5 + k * 0.8, (r.y0 + r.y1) / 2), max(2, int(0.2 * ppm)), c, 1, cv2.LINE_AA)


def _dimension_chain(img, fp: Room, P, ppm, ink, font):
    off = 0.9
    y = fp.y0 - off
    a, b = P(fp.x0, y), P(fp.x1, y)
    if a[1] > 8:
        cv2.line(img, a, b, ink, 1)
        for p in (a, b):
            cv2.line(img, (p[0] - 3, p[1] + 3), (p[0] + 3, p[1] - 3), ink, 1)
        cv2.putText(img, f"{fp.w:.2f}", ((a[0] + b[0]) // 2 - 15, a[1] - 4), font, 0.35, ink, 1, cv2.LINE_AA)
    x = fp.x0 - off
    a, b = P(x, fp.y0), P(x, fp.y1)
    if a[0] > 8:
        cv2.line(img, a, b, ink, 1)
        for p in (a, b):
            cv2.line(img, (p[0] - 3, p[1] + 3), (p[0] + 3, p[1] - 3), ink, 1)


def _degrade(img, mask, rng):
    size = img.shape[0]
    if rng.random() < 0.3:
        ang = rng.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=255)
        mask = cv2.warpAffine(mask, M, (size, size), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < 0.3:  # paper tint / scan background
        tint = np.full_like(img, int(rng.integers(200, 250)))
        img = cv2.addWeighted(img, rng.uniform(0.75, 0.95), tint, 0, 0) + int(rng.integers(0, 20))
        img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.5:
        noise = rng.normal(0, rng.uniform(2, 12), img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        q = int(rng.integers(30, 85))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)
    return img, mask
