"""Synthetic training tables until real data is connected.

make_rentals: rental listings with Egyptian area priors (configs/market.yaml), heteroscedastic noise.
make_jobs:    historical finishing jobs = catalog rules x hidden effects (region logistics, high floors,
              economies of scale, contractor tier, overruns) x inflation since the catalog date.
Both carry the generating effects only implicitly, so models must learn them.
"""

import numpy as np
import pandas as pd

from ryfc import cost
from ryfc.config import PACKAGES, load_yaml
from ryfc.synth import _layout

UNIT_TYPES = list(load_yaml("market.yaml")["unit_type"])
VIEWS = list(load_yaml("market.yaml")["view"])
TIERS = ["A", "B", "C"]


def make_rentals(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    m = load_yaml("market.yaml")
    rng = np.random.default_rng(seed)
    areas = list(m["areas"])
    df = pd.DataFrame({
        "area": rng.choice(areas, n, p=_weights(len(areas), rng)),
        "compound_tier": rng.choice(TIERS, n, p=[0.25, 0.5, 0.25]),
        "unit_type": rng.choice(UNIT_TYPES, n, p=[0.62, 0.1, 0.06, 0.08, 0.06, 0.08]),
        "view": rng.choice(VIEWS, n, p=[0.3, 0.15, 0.3, 0.25]),
        "finishing": rng.choice(PACKAGES, n, p=[0.35, 0.45, 0.2]),
        "furnished": rng.random(n) < 0.35,
        "amenities": rng.integers(0, 6, n),
        "months_ago": rng.integers(0, 12, n),
    })
    base_size = {"apartment": 140, "duplex": 220, "penthouse": 200, "townhouse": 230, "twinhouse": 270, "villa": 380}
    df["size_m2"] = [max(55, rng.normal(base_size[t], base_size[t] * 0.28)) for t in df.unit_type]
    df["bedrooms"] = np.clip(np.round(df.size_m2 / 55 + rng.normal(0, 0.5, n)), 1, 6).astype(int)
    df["bathrooms"] = np.clip(np.round(df.bedrooms * 0.75 + rng.normal(0, 0.4, n)), 1, 5).astype(int)
    df["floor"] = np.where(df.unit_type == "apartment", rng.integers(0, 12, n), 0)

    rpm = df.area.map({k: v["rent_per_m2"] for k, v in m["areas"].items()})
    log_rent = (np.log(rpm * df.size_m2) - 0.12 * np.log(df.size_m2 / 150)
                + np.log(df.compound_tier.map(m["compound_tier"]))
                + np.log(df.unit_type.map(m["unit_type"]))
                + np.log(df.view.map(m["view"]))
                + np.log(df.finishing.map(m["finishing"]))
                + np.log1p(m["furnished_uplift"] * df.furnished)
                + 0.025 * df.amenities
                + 0.03 * (df.bathrooms - df.bedrooms * 0.75)
                + np.where(df.floor == 0, 0.03, 0) + np.where(df.floor >= 8, 0.02, 0))
    sigma = 0.11 + 0.05 * (df.compound_tier == "A") + 0.04 * (df.unit_type == "villa")
    log_rent += rng.normal(0, sigma)
    # asking rents posted months ago, before recent growth
    growth = (1 + m["annual_rent_growth"]) ** (df.months_ago / 12)
    df["rent_monthly"] = np.round(np.exp(log_rent) / growth, -2)
    return df


def _weights(k, rng):
    w = rng.uniform(0.5, 1.5, k)
    return w / w.sum()


def make_jobs(n: int = 4000, seed: int = 1) -> pd.DataFrame:
    c = load_yaml("catalog.yaml")
    rng = np.random.default_rng(seed)
    region_eff = {"New Cairo": 0.0, "Sheikh Zayed": 0.02, "6th of October": -0.04, "New Capital": 0.05,
                  "Madinaty": 0.01, "Mostakbal City": 0.03, "Shorouk": -0.03, "Alexandria": -0.06}
    tier_eff = {"budget": -0.09, "mid": 0.0, "top": 0.13}
    rows = []
    for _ in range(n):
        rooms_rect, _ = _layout(rng)
        rooms = [{"type": r.kind, "area_m2": r.w * r.h * 0.94, "perimeter_m": 2 * (r.w + r.h)} for r in rooms_rect]
        pkg = str(rng.choice(PACKAGES, p=[0.35, 0.45, 0.2]))
        delivery = str(rng.choice(cost.DELIVERY, p=[0.3, 0.7]))
        region = str(rng.choice(cost.REGIONS))
        tier = str(rng.choice(cost.TIERS, p=[0.3, 0.5, 0.2]))
        floor = int(rng.integers(0, 20))
        months = float(rng.uniform(-18, 2))  # jobs priced up to 18 months before the catalog date
        rule_total = cost.rules(rooms, pkg, delivery)["total"]
        net = sum(r["area_m2"] for r in rooms)
        n_bath = sum(r["type"] == "bathroom" for r in rooms)
        log_eff = (region_eff[region] + tier_eff[tier] + 0.03 * (floor > 8) + 0.04 * (floor > 15)
                   - 0.08 * np.log(net / 150) + (0.05 if pkg == "premium" else 0)
                   + 0.03 * (n_bath >= 3) + rng.normal(0, 0.07))
        actual = rule_total * (1 + c["monthly_inflation"]) ** months * np.exp(log_eff)
        rows.append({"region": region, "floor": floor, "net_area": net, "package": pkg, "n_bath": n_bath,
                     "delivery": delivery, "tier": tier, "months": months, "rule_total": rule_total,
                     "actual_total": actual})
    return pd.DataFrame(rows)
