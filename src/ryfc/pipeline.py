"""End-to-end: plan -> rooms & areas -> finishing cost per package -> rent range -> yield.

detect()    vision only (rooms, scale, annotated plan); run once per upload
evaluate()  financials from a room list; re-run cheaply after the user corrects rooms
"""

import base64
import time
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

from ryfc import cost, monitoring, ocr, vision
from ryfc.config import PACKAGES, load_yaml
from ryfc.rent import RentModel
from ryfc.yieldcalc import YieldInputs, compute


class UnitInfo(BaseModel):
    area: str = "New Cairo"
    compound_tier: Literal["A", "B", "C"] = "B"
    unit_type: str = "apartment"
    bua_m2: float | None = Field(None, gt=10, description="Sales area (BUA). Net = BUA x net_to_bua")
    net_area_m2: float | None = Field(None, gt=10, description="Net internal area, overrides BUA for scale")
    floor: int = 3
    view: str = "landscape"
    furnished: bool = False
    amenities: int = Field(3, ge=0, le=5)
    delivery: Literal["core_shell", "semi_finished"] = "semi_finished"
    contractor_tier: Literal["budget", "mid", "top"] = "mid"
    purchase_price: float | None = Field(None, gt=0, description="Defaults to area price x BUA x 0.85")
    vacancy_months: float | None = None
    management_fee: float | None = None
    maintenance_rate: float | None = None
    discount_rate: float | None = None
    horizon_years: int | None = None

    def net_area(self) -> float | None:
        if self.net_area_m2:
            return self.net_area_m2
        if self.bua_m2:
            return self.bua_m2 * load_yaml("market.yaml")["yield_defaults"]["net_to_bua"]
        return None

    def bua(self, net: float) -> float:
        return self.bua_m2 or net / load_yaml("market.yaml")["yield_defaults"]["net_to_bua"]


@lru_cache
def segmenter() -> vision.Segmenter:
    return vision.Segmenter()


@lru_cache
def rent_model() -> RentModel:
    return RentModel.load()


@lru_cache
def cost_adjuster() -> cost.CostAdjuster:
    return cost.CostAdjuster.load()


def detect(data: bytes, filename: str, unit: UnitInfo, use_ocr: bool = True) -> dict:
    t0 = time.perf_counter()
    plan = vision.preprocess(vision.load_image(data, filename))
    probs = segmenter().probs(plan.gray)
    rooms, inst = vision.rooms_from_probs(probs)
    t_seg = time.perf_counter() - t0

    words, dims = ocr.read(plan.original) if use_ocr else ([], [])
    to_model = lambda p: (p[0] * plan.factor + plan.pad[0], p[1] * plan.factor + plan.pad[1])  # noqa: E731
    for w in words + dims:
        w["center"] = to_model(w["center"])
    vision.apply_ocr(rooms, inst, words, dims)

    total_px = sum(r["pixels"] for r in rooms)
    ppm_dims = vision.scale_from_dims(rooms)
    net = unit.net_area()
    warnings = []
    if net:
        ppm, method = (total_px / net) ** 0.5, "total area"
        if ppm_dims and abs(ppm_dims / ppm - 1) > 0.1:
            warnings.append(f"Printed room dimensions imply {total_px / ppm_dims**2:.0f} m2 net vs "
                            f"{net:.0f} m2 entered: check the area input.")
    elif ppm_dims:
        ppm, method = ppm_dims, "OCR room dimensions"
    else:
        raise ValueError("No scale: enter the unit area (BUA or net) or upload a plan with printed room dimensions.")
    vision.apply_scale(rooms, ppm)
    if not rooms:
        raise ValueError("No rooms detected. Upload a clearer plan image.")
    flagged = sum(r["flagged"] for r in rooms)
    if flagged:
        warnings.append(f"{flagged} room(s) have low confidence: please check type and area.")
    png = vision.annotate(plan.gray, inst, rooms)
    latency = time.perf_counter() - t0
    monitoring.log_event("detect", {"latency_s": latency, "seg_s": t_seg, "rooms": len(rooms),
                                    "flagged": flagged, "scale": method, "ocr_words": len(words)})
    clean = [{k: v for k, v in r.items() if k not in ("pixels", "perimeter_px", "bbox", "centroid", "fill_ratio")}
             for r in rooms]
    return {"rooms": clean, "scale": {"method": method, "px_per_m": ppm, "ocr_dims_px_per_m": ppm_dims},
            "net_area_m2": sum(r["area_m2"] for r in rooms), "annotated_png": base64.b64encode(png).decode(),
            "warnings": warnings, "latency_s": latency, "ocr_available": ocr.available()}


def evaluate(rooms: list[dict], unit: UnitInfo) -> dict:
    t0 = time.perf_counter()
    m = load_yaml("market.yaml")
    d = m["yield_defaults"]
    net = sum(r["area_m2"] for r in rooms)
    bua = unit.bua(net)
    beds = max(1, sum(r["type"] == "bedroom" for r in rooms))
    baths = max(1, sum(r["type"] == "bathroom" for r in rooms))
    price = unit.purchase_price or m["areas"][unit.area]["sale_per_m2"] * bua * 0.85
    base = {"area": unit.area, "compound_tier": unit.compound_tier, "unit_type": unit.unit_type,
            "view": unit.view, "furnished": unit.furnished, "amenities": unit.amenities, "size_m2": bua,
            "bedrooms": beds, "bathrooms": baths, "floor": unit.floor}
    rents = dict(zip(PACKAGES, rent_model().predict([base | {"finishing": p} for p in PACKAGES])))
    adj = cost_adjuster()
    results = {}
    for p in PACKAGES:
        c = cost.estimate(rooms, p, unit.delivery, unit.area, unit.floor, unit.contractor_tier, adj)
        fin = c["adjusted"]["estimate"]
        yi = dict(purchase_price=price, finishing_cost=fin,
                  vacancy_months=unit.vacancy_months if unit.vacancy_months is not None else d["vacancy_months"],
                  management_fee=unit.management_fee if unit.management_fee is not None else d["management_fee"],
                  maintenance_rate=unit.maintenance_rate if unit.maintenance_rate is not None else d["maintenance_rate"],
                  annual_rent_growth=m["annual_rent_growth"],
                  discount_rate=unit.discount_rate if unit.discount_rate is not None else d["discount_rate"],
                  horizon_years=unit.horizon_years or d["horizon_years"])
        r = rents[p]
        results[p] = {"cost": c, "rent": r,
                      "yield": compute(YieldInputs(monthly_rent=r["median"], **yi)),
                      "yield_low": compute(YieldInputs(monthly_rent=r["low"], **yi)),
                      "yield_high": compute(YieldInputs(monthly_rent=r["high"], **yi))}
    best = max(PACKAGES, key=lambda p: results[p]["yield"]["npv_of_finishing"])
    monitoring.log_event("evaluate", {"latency_s": time.perf_counter() - t0, "area": unit.area,
                                      "net_m2": net, "rent_q50_standard": rents["standard"]["median"]})
    return {"net_area_m2": net, "bua_m2": bua, "bedrooms": beds, "bathrooms": baths, "purchase_price": price,
            "purchase_price_estimated": unit.purchase_price is None, "packages": results,
            "recommended_package": best}


def analyze(data: bytes, filename: str, unit: UnitInfo, use_ocr: bool = True) -> dict:
    det = detect(data, filename, unit, use_ocr)
    return det | {"financials": evaluate(det["rooms"], unit)}
