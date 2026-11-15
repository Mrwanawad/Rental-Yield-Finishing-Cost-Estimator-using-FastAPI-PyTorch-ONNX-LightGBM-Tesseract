"""Train the rent quantile models and the finishing-cost adjuster; write reports/tabular.md.

python scripts/train_tabular.py [--trials 25]
"""

import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from ryfc import cost, rent, trees
from ryfc.config import MODELS, REPORTS
from ryfc.datasets import make_jobs, make_rentals
from ryfc.tracking import run

BASE = {"verbosity": -1, "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "lambda_l2": 1.0}


def mape(y, p):
    return float(np.mean(np.abs(p - y) / y) * 100)


def split(n, rng, fracs=(0.6, 0.2, 0.2)):
    idx = rng.permutation(n)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def tune_q50(Xtr, ytr, Xva, yva, trials):
    def objective(trial):
        p = BASE | {"objective": "quantile", "alpha": 0.5,
                    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                    "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True)}
        b = lgb.train(p, lgb.Dataset(Xtr, ytr), 2000, valid_sets=[lgb.Dataset(Xva, yva)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return float(np.mean(np.abs(np.exp(b.predict(Xva)) - np.exp(yva))))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=trials)
    return BASE | study.best_params


def train_rent(trials, log):
    df = make_rentals(20000)
    X, y = rent.to_frame(df), rent.target(df)
    rng = np.random.default_rng(0)
    tr, ca, te = split(len(df), rng)
    # inner split of train for early stopping / tuning
    n_in = int(len(tr) * 0.8)
    tr_in, va_in = tr[:n_in], tr[n_in:]
    params = tune_q50(X.iloc[tr_in], y[tr_in], X.iloc[va_in], y[va_in], trials)
    boosters = {}
    for q in rent.QUANTILES:
        p = params | {"objective": "quantile", "alpha": q}
        b = lgb.train(p, lgb.Dataset(X.iloc[tr_in], y[tr_in]), 3000,
                      valid_sets=[lgb.Dataset(X.iloc[va_in], y[va_in])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        boosters[q] = b
    # split-conformal (CQR) on the calibration fold for 80% intervals
    def predict_log(Xs, widen):
        p = {q: b.predict(Xs) for q, b in boosters.items()}
        return {0.1: np.minimum(p[0.1], p[0.5]) - widen, 0.5: p[0.5], 0.9: np.maximum(p[0.9], p[0.5]) + widen}

    pc = predict_log(X.iloc[ca], 0.0)
    scores = np.maximum(pc[0.1] - y[ca], y[ca] - pc[0.9])
    level = np.ceil((len(ca) + 1) * 0.8) / len(ca)
    conformal = float(np.quantile(scores, level))

    pt = predict_log(X.iloc[te], conformal)
    yt, med, lo, hi = np.exp(y[te]), np.exp(pt[0.5]), np.exp(pt[0.1]), np.exp(pt[0.9])
    raw = boosters  # coverage before conformal for the report
    rlo = np.exp(np.minimum(raw[0.1].predict(X.iloc[te]), pt[0.5]))
    rhi = np.exp(np.maximum(raw[0.9].predict(X.iloc[te]), pt[0.5]))
    # naive baseline: area rent/m2 x size
    base = df.iloc[tr].assign(ppm=np.exp(y[tr]) / df.size_m2.iloc[tr]).groupby("area")["ppm"].median()
    base_pred = df.area.iloc[te].map(base).to_numpy() * df.size_m2.iloc[te].to_numpy()
    metrics = {
        "rent_mae": float(np.mean(np.abs(med - yt))), "rent_mape": mape(yt, med),
        "rent_baseline_mape": mape(yt, base_pred),
        "rent_coverage_raw": float(np.mean((yt >= rlo) & (yt <= rhi))),
        "rent_coverage_80": float(np.mean((yt >= lo) & (yt <= hi))),
        "rent_rel_width": float(np.median((hi - lo) / med)),
    }
    by_area = (pd.DataFrame({"area": df.area.iloc[te].to_numpy(), "ape": np.abs(med - yt) / yt * 100})
               .groupby("area")["ape"].mean().round(1))
    imp = pd.Series(boosters[0.5].feature_importance("gain"), index=rent.FEATURES)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False).round(1)
    # residuals per area at training time: reference for drift monitoring
    resid_ref = (pd.DataFrame({"area": df.area.iloc[te].to_numpy(), "r": np.log(yt) - np.log(med)})
                 .groupby("area")["r"].agg(["mean", "std"]).round(4).to_dict("index"))

    MODELS.mkdir(exist_ok=True)
    for q, b in boosters.items():
        b.save_model(str(MODELS / f"rent_q{int(q * 100)}.txt"))
        trees.export(b, MODELS / f"rent_q{int(q * 100)}.json")
    (MODELS / "rent_meta.json").write_text(json.dumps(
        {"conformal": conformal, "params": params, "residual_ref": resid_ref, "metrics": metrics}, indent=2))
    log(metrics)
    return metrics, by_area, imp, params


def train_cost(log):
    jobs = make_jobs(4000)
    infl = (1 + 0.012) ** jobs.months
    y = np.log(jobs.actual_total / (jobs.rule_total * infl))
    X = cost._typed(jobs)
    rng = np.random.default_rng(1)
    tr, va, te = split(len(jobs), rng, (0.6, 0.2, 0.2))
    p = BASE | {"objective": "regression", "num_leaves": 15, "min_data_in_leaf": 30}
    b = lgb.train(p, lgb.Dataset(X.iloc[tr], y.iloc[tr]), 3000, valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    res_va = y.iloc[va] - b.predict(X.iloc[va])
    q_lo, q_hi = float(np.quantile(res_va, 0.1)), float(np.quantile(res_va, 0.9))
    act = jobs.actual_total.iloc[te].to_numpy()
    rules_only = (jobs.rule_total * infl).iloc[te].to_numpy()
    adjusted = rules_only * np.exp(b.predict(X.iloc[te]))
    covered = (act >= adjusted * np.exp(q_lo)) & (act <= adjusted * np.exp(q_hi))
    metrics = {"cost_mape_rules_only": mape(act, rules_only), "cost_mape_rules_plus_ml": mape(act, adjusted),
               "cost_range_coverage_80": float(covered.mean())}
    pkg = (pd.DataFrame({"package": jobs.package.iloc[te].to_numpy(), "per_m2": act / jobs.net_area.iloc[te].to_numpy()})
           .groupby("package")["per_m2"].median().round(-2))
    b.save_model(str(MODELS / "cost_adj.txt"))
    trees.export(b, MODELS / "cost_adj.json")
    (MODELS / "cost_meta.json").write_text(json.dumps({"q_lo": q_lo, "q_hi": q_hi, "metrics": metrics}, indent=2))
    log(metrics)
    return metrics, pkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    args = ap.parse_args()
    with run("tabular", vars(args)) as log:
        rm, by_area, imp, params = train_rent(args.trials, log)
        cm, pkg = train_cost(log)
    REPORTS.mkdir(exist_ok=True)
    shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items() if k != "verbosity"}
    lines = [
        "# Tabular models: rent range and finishing-cost adjuster", "",
        "Data: synthetic (`ryfc.datasets`), 20k rental listings and 4k finishing jobs. Numbers show the",
        "pipeline works and is calibrated; they are not market accuracy claims until real data is plugged in.", "",
        "## Rent model (LightGBM quantile q10/q50/q90 + split-conformal)", "",
        "| Metric | Value |", "|---|---|",
        f"| MAE (EGP/month) | {rm['rent_mae']:,.0f} |",
        f"| MAPE | {rm['rent_mape']:.1f}% |",
        f"| Baseline MAPE (area median rent/m2 x size) | {rm['rent_baseline_mape']:.1f}% |",
        f"| 80% interval coverage before conformal | {rm['rent_coverage_raw'] * 100:.1f}% |",
        f"| 80% interval coverage after conformal | {rm['rent_coverage_80'] * 100:.1f}% |",
        f"| Median relative interval width | {rm['rent_rel_width'] * 100:.0f}% |", "",
        "MAPE by area (test):", "", "| Area | MAPE % |", "|---|---|",
        *[f"| {a} | {v} |" for a, v in by_area.items()], "",
        "Feature importance (gain %, q50):", "", "| Feature | % |", "|---|---|",
        *[f"| {f} | {v} |" for f, v in imp.items()], "",
        f"Tuned params (Optuna, {args.trials} trials): `{json.dumps(shown)}`", "",
        "## Finishing cost (catalog rules + LightGBM adjustment)", "",
        "| Metric | Value |", "|---|---|",
        f"| MAPE, catalog rules + inflation only | {cm['cost_mape_rules_only']:.1f}% |",
        f"| MAPE, rules + ML adjustment | {cm['cost_mape_rules_plus_ml']:.1f}% |",
        f"| 80% cost range coverage | {cm['cost_range_coverage_80'] * 100:.1f}% |", "",
        "Median actual cost per net m2 by package (EGP):", "", "| Package | EGP/m2 |", "|---|---|",
        *[f"| {k} | {v:,.0f} |" for k, v in pkg.items()], "",
    ]
    (REPORTS / "tabular.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
