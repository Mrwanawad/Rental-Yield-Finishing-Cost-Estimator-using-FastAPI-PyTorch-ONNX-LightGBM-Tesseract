"""Finishing cost: deterministic catalog rules + a LightGBM adjustment learned from past jobs.

rules(rooms, package)       -> line items and category breakdown from the price catalog
CostAdjuster                -> multiplier for region / floor / size / contractor effects that
                               the catalog does not capture, plus catalog inflation since its date
"""

import json
from dataclasses import dataclass
from datetime import date

import numpy as np

from ryfc.config import MODELS, PACKAGES, load_yaml
from ryfc.trees import TreeModel

CATEGORIES = ["flooring", "plaster_paint_ceiling", "kitchen", "bathrooms", "electrical", "ac",
              "doors", "contingency"]
REGIONS = list(load_yaml("market.yaml")["areas"])
TIERS = ["budget", "mid", "top"]
DELIVERY = ["core_shell", "semi_finished"]


def _perimeter(room: dict) -> float:
    return room.get("perimeter_m") or 4 * np.sqrt(room["area_m2"]) * 1.05


def ac_hp(area: float) -> list[float]:
    """Split units per room, by cooling load rule of thumb (~1 HP per 9-10 m2)."""
    units = []
    while area > 0:
        chunk = min(area, 32.0)
        units.append(1.5 if chunk <= 14 else 2.25 if chunk <= 22 else 3.0 if chunk <= 28 else 4.0)
        area -= chunk
    return units


def rules(rooms: list[dict], package: str, delivery: str = "semi_finished") -> dict:
    c = load_yaml("catalog.yaml")
    p = package
    items: list[dict] = []

    def add(cat, name, qty, unit, price):
        if qty > 0:
            items.append({"category": cat, "item": name, "qty": round(float(qty), 2), "unit": unit,
                          "unit_price": float(price), "total": float(qty * price)})

    semi = delivery == "semi_finished"
    disc = c["semi_finished_discount"]
    h = c["ceiling_height_m"]
    net = sum(r["area_m2"] for r in rooms)
    by = lambda *kinds: [r for r in rooms if r["type"] in kinds]  # noqa: E731

    dry = sum(r["area_m2"] for r in by("living", "bedroom", "corridor"))
    wet = sum(r["area_m2"] for r in by("kitchen", "bathroom"))
    balc = sum(r["area_m2"] for r in by("balcony"))
    add("flooring", "Flooring - dry rooms", dry, "m2", c["flooring_per_m2"]["dry"][p])
    add("flooring", "Flooring - kitchen & bathrooms", wet, "m2", c["flooring_per_m2"]["wet"][p])
    add("flooring", "Flooring - balcony", balc, "m2", c["flooring_per_m2"]["balcony"][p])

    # wall area of rooms that get plaster + paint (bathrooms are tiled), 15% off for openings
    paint_wall = sum(_perimeter(r) * h * 0.85 for r in by("living", "bedroom", "corridor", "kitchen"))
    all_wall = paint_wall + sum(_perimeter(r) * h * 0.85 for r in by("bathroom"))
    add("plaster_paint_ceiling", "Plaster", all_wall * (0 if semi else 1), "m2 wall", c["plaster_per_m2_wall"][p])
    add("plaster_paint_ceiling", "Paint", paint_wall, "m2 wall", c["paint_per_m2_wall"][p])
    add("plaster_paint_ceiling", "Ceiling", dry + wet, "m2", c["ceiling_per_m2"][p])

    for k in by("kitchen"):
        lm = float(np.clip(_perimeter(k) * 0.45, 2.5, 8.0))
        add("kitchen", "Kitchen cabinets & worktop", lm, "linear m", c["kitchen_per_linear_m"][p])
        add("kitchen", "Kitchen backsplash tiles", lm * 0.6, "m2", c["wall_tiles_per_m2"][p])
    baths = by("bathroom")
    for b in baths:
        small = b["area_m2"] < 3.5
        name = "Guest WC set" if small else "Bathroom set"
        add("bathrooms", name, 1, "set", c["bathroom_set"][p] * (c["guest_wc_factor"] if small else 1))
        add("bathrooms", "Bathroom wall tiles", _perimeter(b) * 2.6 * 0.9, "m2", c["wall_tiles_per_m2"][p])
    wet_rooms = len(baths) + len(by("kitchen"))
    plumb = c["plumbing_per_wet_room"][p] * (1 - disc["plumbing"] if semi else 1)
    add("bathrooms", "Plumbing", wet_rooms, "wet room", plumb)

    elec = c["electrical_per_m2"][p] * (1 - disc["electrical"] if semi else 1)
    add("electrical", "Electrical (wiring, panel, points, lighting)", net, "m2", elec)

    for r in by("living", "bedroom"):
        for hp in ac_hp(r["area_m2"]):
            add("ac", f"AC {hp} HP ({r['type']})", 1, "unit",
                c["ac_unit_by_hp"][hp] * c["ac_package_factor"][p] + c["ac_install_each"])

    add("doors", "Internal doors", len(by("bedroom", "bathroom", "kitchen")), "door", c["door_each"][p])

    subtotal = sum(i["total"] for i in items)
    add("contingency", "Contingency & contractor margin", 1, "lot", subtotal * c["contingency"][p])
    breakdown = {cat: sum(i["total"] for i in items if i["category"] == cat) for cat in CATEGORIES}
    total = sum(breakdown.values())
    return {"package": p, "items": items, "breakdown": breakdown, "total": total,
            "per_m2": total / net if net else 0.0}


def months_since_catalog(today: date | None = None) -> float:
    d0 = date.fromisoformat(load_yaml("catalog.yaml")["catalog_date"])
    today = today or date.today()
    return (today.year - d0.year) * 12 + (today.month - d0.month) + (today.day - d0.day) / 30


def job_features(region, floor, net_area, package, n_bath, delivery, tier) -> dict:
    return {"region": region, "floor": floor, "net_area": net_area, "package": package,
            "n_bath": n_bath, "delivery": delivery, "tier": tier}


def _typed(df):
    """Training side: pandas frame with fixed categories (lightgbm)."""
    import pandas as pd

    df = df.copy()
    for col, cats in (("region", REGIONS), ("package", PACKAGES), ("delivery", DELIVERY), ("tier", TIERS)):
        df[col] = pd.Categorical(df[col], categories=cats)
    return df[["region", "floor", "net_area", "package", "n_bath", "delivery", "tier"]]


@dataclass
class CostAdjuster:
    model: TreeModel
    q_lo: float  # residual log-ratio quantiles for the range
    q_hi: float

    @classmethod
    def load(cls, path=None):
        meta = json.loads((MODELS / "cost_meta.json").read_text())
        return cls(TreeModel.load(path or MODELS / "cost_adj.json"), meta["q_lo"], meta["q_hi"])

    def adjust(self, rule_total: float, feats: dict, months: float) -> dict:
        infl = (1 + load_yaml("catalog.yaml")["monthly_inflation"]) ** max(months, 0)
        log_adj = float(self.model.predict([feats])[0])
        mid = rule_total * infl * np.exp(log_adj)
        return {"multiplier": float(np.exp(log_adj)), "inflation_factor": float(infl),
                "estimate": float(mid), "low": float(mid * np.exp(self.q_lo)),
                "high": float(mid * np.exp(self.q_hi))}


def estimate(rooms, package, delivery="semi_finished", region="New Cairo", floor=3, tier="mid",
             adjuster: CostAdjuster | None = None) -> dict:
    out = rules(rooms, package, delivery)
    n_bath = sum(r["type"] == "bathroom" for r in rooms)
    net = sum(r["area_m2"] for r in rooms)
    if adjuster is not None:
        out["adjusted"] = adjuster.adjust(out["total"], job_features(region, floor, net, package, n_bath,
                                                                      delivery, tier), months_since_catalog())
    else:
        out["adjusted"] = {"multiplier": 1.0, "inflation_factor": 1.0, "estimate": out["total"],
                           "low": out["total"] * 0.9, "high": out["total"] * 1.15}
    return out
