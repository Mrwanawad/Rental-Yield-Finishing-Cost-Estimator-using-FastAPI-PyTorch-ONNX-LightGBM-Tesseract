"""Production monitoring: JSONL event log (latency, low-confidence rooms), latency summary,
rent residual drift by area against the training reference, catalog staleness."""

import json
import threading
import time
from datetime import date

import numpy as np

from ryfc.config import MODELS, WORK, load_yaml

_lock = threading.Lock()


def log_path():
    return WORK / "events.jsonl"


def log_event(kind: str, payload: dict) -> None:
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        with _lock, open(log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "kind": kind} | payload, default=float) + "\n")
    except OSError:
        pass  # monitoring must never break a request


def summary() -> dict:
    p = log_path()
    events = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line] if p.exists() else []
    out = {"events": len(events)}
    for kind in sorted({e["kind"] for e in events}):
        g = [e for e in events if e["kind"] == kind]
        lat = [e["latency_s"] for e in g if e.get("latency_s") is not None]
        out[kind] = {"count": len(g), "p50_latency_s": float(np.quantile(lat, 0.5)) if lat else None,
                     "p95_latency_s": float(np.quantile(lat, 0.95)) if lat else None}
        if kind == "detect":
            out[kind]["low_confidence_rate"] = sum((e.get("flagged") or 0) > 0 for e in g) / len(g)
    out["catalog"] = catalog_status()
    return out


def catalog_status(today: date | None = None) -> dict:
    c = load_yaml("catalog.yaml")
    age = ((today or date.today()) - date.fromisoformat(c["catalog_date"])).days
    drift = (1 + c["monthly_inflation"]) ** (age / 30) - 1
    return {"catalog_date": c["catalog_date"], "age_days": age, "implied_inflation_pct": round(drift * 100, 1),
            "stale": age > 90}


def rent_drift(observed, predicted_median: np.ndarray, threshold: float = 0.08):
    """Batch job (pandas): mean log residual per area on newly observed rents vs. the training reference."""
    import pandas as pd

    ref = json.loads((MODELS / "rent_meta.json").read_text())["residual_ref"]
    df = pd.DataFrame({"area": observed["area"].to_numpy(),
                       "r": np.log(observed["rent_monthly"].to_numpy()) - np.log(predicted_median)})
    g = df.groupby("area")["r"].agg(["mean", "count"]).reset_index()
    g["ref_mean"] = g["area"].map(lambda a: ref.get(a, {}).get("mean", 0.0))
    g["shift_pct"] = (np.exp(g["mean"] - g["ref_mean"]) - 1) * 100
    g["drift"] = g["shift_pct"].abs() > threshold * 100
    return g.round(3)
