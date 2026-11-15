"""Rent range: LightGBM quantile models (q10/q50/q90) on log rent, split-conformal (CQR) calibrated.

Targets are asking rents grown to today with the configured annual rent growth, so the model
predicts current rents without needing to extrapolate time. Serving evaluates the exported trees
(`ryfc.trees`) so it needs neither lightgbm nor pandas.
"""

import json
import math
from dataclasses import dataclass

from ryfc.config import MODELS, PACKAGES, load_yaml
from ryfc.trees import TreeModel

FEATURES = ["area", "compound_tier", "unit_type", "view", "finishing", "furnished", "amenities",
            "size_m2", "bedrooms", "bathrooms", "floor"]
QUANTILES = (0.1, 0.5, 0.9)
DRIVER_LABELS = {"area": "location", "size_m2": "size", "compound_tier": "compound tier", "unit_type": "unit type"}


def categories() -> dict[str, list]:
    m = load_yaml("market.yaml")
    return {"area": list(m["areas"]), "compound_tier": ["A", "B", "C"], "unit_type": list(m["unit_type"]),
            "view": list(m["view"]), "finishing": PACKAGES}


def to_frame(rows):
    """Training side: pandas frame with fixed categories (lightgbm)."""
    import pandas as pd

    df = pd.DataFrame([rows] if isinstance(rows, dict) else rows).copy()
    for col, cats in categories().items():
        df[col] = pd.Categorical(df[col], categories=cats)
    df["furnished"] = df["furnished"].astype(int)
    return df[FEATURES]


def target(df):
    import numpy as np

    g = load_yaml("market.yaml")["annual_rent_growth"]
    return np.log(df["rent_monthly"] * (1 + g) ** (df["months_ago"] / 12)).to_numpy()


def _row(r: dict) -> dict:
    return {f: (int(r[f]) if f == "furnished" else r[f]) for f in FEATURES}


@dataclass
class RentModel:
    models: dict[float, TreeModel]
    conformal: float  # CQR widening in log space

    @classmethod
    def load(cls):
        meta = json.loads((MODELS / "rent_meta.json").read_text())
        return cls({q: TreeModel.load(MODELS / f"rent_q{int(q * 100)}.json") for q in QUANTILES}, meta["conformal"])

    def predict_log(self, rows: list[dict]) -> dict[float, list[float]]:
        p = {q: m.predict(rows) for q, m in self.models.items()}
        lo = [min(a, b) - self.conformal for a, b in zip(p[0.1], p[0.5])]
        hi = [max(a, b) + self.conformal for a, b in zip(p[0.9], p[0.5])]
        return {0.1: lo, 0.5: p[0.5], 0.9: hi}

    def predict(self, rows) -> list[dict]:
        rows = [_row(r) for r in ([rows] if isinstance(rows, dict) else rows)]
        p = self.predict_log(rows)
        out = []
        for i, r in enumerate(rows):
            contrib = self.models[0.5].contrib(r)
            drivers = sorted(contrib.items(), key=lambda t: -abs(t[1]))[:5]
            out.append({"low": math.exp(p[0.1][i]), "median": math.exp(p[0.5][i]), "high": math.exp(p[0.9][i]),
                        "drivers": [{"feature": DRIVER_LABELS.get(f, f), "effect_pct": math.expm1(v) * 100}
                                    for f, v in drivers]})
        return out
