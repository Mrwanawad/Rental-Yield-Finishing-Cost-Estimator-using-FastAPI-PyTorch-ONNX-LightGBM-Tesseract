"""Convert trained LightGBM models to JSON for lightgbm-free serving; check parity and attribution sums.

python scripts/export_trees.py
"""

import lightgbm as lgb
import numpy as np

from ryfc import cost, rent, trees
from ryfc.config import MODELS
from ryfc.datasets import make_jobs, make_rentals


def check(booster, model, frame, rows, name):
    ref = booster.predict(frame)
    got = np.array(model.predict(rows))
    contrib = model.contrib(rows[0])
    base = model.trees and sum(t.get("internal_value", t.get("leaf_value", 0)) for t in model.trees)
    print(f"{name}: max |diff| {np.abs(ref - got).max():.2e} over {len(rows)} rows | "
          f"attribution sum error {abs(base + sum(contrib.values()) - got[0]):.2e}")
    assert np.abs(ref - got).max() < 1e-9


def main():
    df = make_rentals(2000, seed=7)
    frame = rent.to_frame(df)
    rows = frame.astype(object).to_dict("records")
    for q in rent.QUANTILES:
        b = lgb.Booster(model_file=str(MODELS / f"rent_q{int(q * 100)}.txt"))
        trees.export(b, MODELS / f"rent_q{int(q * 100)}.json")
        check(b, trees.TreeModel.load(MODELS / f"rent_q{int(q * 100)}.json"), frame, rows, f"rent q{int(q * 100)}")
    jobs = make_jobs(300, seed=9)
    frame = cost._typed(jobs)
    b = lgb.Booster(model_file=str(MODELS / "cost_adj.txt"))
    trees.export(b, MODELS / "cost_adj.json")
    check(b, trees.TreeModel.load(MODELS / "cost_adj.json"), frame, frame.astype(object).to_dict("records"), "cost adj")


if __name__ == "__main__":
    main()
