"""Monitoring snapshot -> reports/monitoring.md: latency + low-confidence rate from the event log,
catalog staleness, and a rent drift check on a batch of newly observed rents.

python scripts/monitor_report.py [--observed new_rents.csv]
Without --observed, simulates a month where New Capital rents rose 15% to show the drift alarm.
"""

import argparse

import numpy as np
import pandas as pd

from ryfc import monitoring, rent
from ryfc.config import REPORTS
from ryfc.datasets import make_rentals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observed")
    args = ap.parse_args()
    if args.observed:
        obs = pd.read_csv(args.observed)
        note = f"Observed rents: {args.observed}"
    else:
        obs = make_rentals(3000, seed=99)
        obs["rent_monthly"] = np.exp(rent.target(obs))  # today's rents
        obs["months_ago"] = 0
        obs.loc[obs.area == "New Capital", "rent_monthly"] *= 1.15
        note = "Simulated batch (3,000 rents, New Capital +15%)"
    model = rent.RentModel.load()
    med = [p["median"] for p in model.predict(obs[rent.FEATURES].to_dict("records"))]
    drift = monitoring.rent_drift(obs, pd.Series(med).to_numpy())
    s = monitoring.summary()
    cat = monitoring.catalog_status()
    lines = ["# Monitoring snapshot", "", "## Requests", ""]
    for kind in ("detect", "evaluate"):
        if kind in s:
            k = s[kind]
            low = f", low-confidence plans {k['low_confidence_rate'] * 100:.0f}%" if "low_confidence_rate" in k else ""
            lines.append(f"- {kind}: {k['count']} calls, p50 {k['p50_latency_s']:.2f} s, p95 {k['p95_latency_s']:.2f} s{low}")
    lines += ["", "## Catalog", "",
              f"- Catalog date {cat['catalog_date']}, age {cat['age_days']} days, implied inflation since "
              f"{cat['implied_inflation_pct']}%{' - STALE: refresh prices' if cat['stale'] else ''}", "",
              "## Rent residual drift by area", "", note, "",
              "| Area | n | Shift vs training | Drift |", "|---|---|---|---|",
              *[f"| {r.area} | {r['count']} | {r.shift_pct:+.1f}% | {'yes' if r.drift else ''} |" for _, r in drift.iterrows()],
              ""]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "monitoring.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
