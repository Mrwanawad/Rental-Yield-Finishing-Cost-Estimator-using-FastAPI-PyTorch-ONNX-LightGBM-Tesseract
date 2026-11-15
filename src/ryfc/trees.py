"""Dependency-free LightGBM inference from `Booster.dump_model()` JSON.

Serving needs no lightgbm / scipy / pandas (~175 MB), which keeps the serverless bundle under limits.
Matches LightGBM's decision rules for numerical splits (missing types None/Zero/NaN) and categorical
splits ("a||b||c" category sets). `contrib` gives Saabas path attributions from node internal values.
"""

import json
import math
from pathlib import Path

ZERO = 1e-35  # LightGBM kZeroThreshold


class TreeModel:
    def __init__(self, dump: dict):
        self.features: list[str] = dump["feature_names"]
        self.trees = [t["tree_structure"] for t in dump["tree_info"]]
        # categorical columns (non-empty "values") in feature order map to pandas category lists; code = index
        infos = dump["feature_infos"]
        cat_cols = [f for f in self.features if infos.get(f, {}).get("values")]
        self.categories: dict[str, list] = {c: list(v) for c, v in zip(cat_cols, dump.get("pandas_categorical") or [])}
        for t in self.trees:
            _prepare(t)

    @classmethod
    def load(cls, path: Path) -> "TreeModel":
        return cls(json.loads(Path(path).read_text()))

    def encode(self, row: dict) -> list[float]:
        out = []
        for f in self.features:
            v = row[f]
            if f in self.categories:
                cats = self.categories[f]
                out.append(float(cats.index(v)) if v in cats else math.nan)
            else:
                out.append(float(v) if v is not None else math.nan)
        return out

    def predict(self, rows: list[dict]) -> list[float]:
        return [sum(_leaf(t, x)["leaf_value"] for t in self.trees) for x in map(self.encode, rows)]

    def contrib(self, row: dict) -> dict[str, float]:
        """Saabas attribution: change in node value along the decision path, summed per feature."""
        x = self.encode(row)
        out = dict.fromkeys(self.features, 0.0)
        for t in self.trees:
            node = t
            while "leaf_value" not in node:
                child = node["left_child"] if _go_left(node, x[node["split_feature"]]) else node["right_child"]
                out[self.features[node["split_feature"]]] += _value(child) - node["internal_value"]
                node = child
        return out


def _prepare(node):
    if "leaf_value" in node:
        return
    if node["decision_type"] == "==":
        node["_cats"] = {int(c) for c in str(node["threshold"]).split("||")}
    _prepare(node["left_child"])
    _prepare(node["right_child"])


def _value(node):
    return node["leaf_value"] if "leaf_value" in node else node["internal_value"]


def _go_left(node, x: float) -> bool:
    if node["decision_type"] == "==":
        if math.isnan(x) or x < 0:
            return False
        return int(x) in node["_cats"]
    missing = node.get("missing_type", "None")
    if math.isnan(x) and missing != "NaN":
        x = 0.0
    if (missing == "Zero" and abs(x) <= ZERO) or (missing == "NaN" and math.isnan(x)):
        return bool(node["default_left"])
    return x <= node["threshold"]


def _leaf(node, x):
    while "leaf_value" not in node:
        node = node["left_child"] if _go_left(node, x[node["split_feature"]]) else node["right_child"]
    return node


def export(booster, path: Path) -> None:
    """Training side: write a booster (lightgbm) as JSON for serving."""
    Path(path).write_text(json.dumps(booster.dump_model()))
