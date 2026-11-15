# Tabular models: rent range and finishing-cost adjuster

Data: synthetic (`ryfc.datasets`), 20k rental listings and 4k finishing jobs. Numbers show the
pipeline works and is calibrated; they are not market accuracy claims until real data is plugged in.

## Rent model (LightGBM quantile q10/q50/q90 + split-conformal)

| Metric | Value |
|---|---|
| MAE (EGP/month) | 4,878 |
| MAPE | 10.8% |
| Baseline MAPE (area median rent/m2 x size) | 24.2% |
| 80% interval coverage before conformal | 74.0% |
| 80% interval coverage after conformal | 78.0% |
| Median relative interval width | 33% |

MAPE by area (test):

| Area | MAPE % |
|---|---|
| 6th of October | 10.7 |
| Alexandria | 10.8 |
| Madinaty | 11.5 |
| Mostakbal City | 10.8 |
| New Cairo | 10.5 |
| New Capital | 10.9 |
| Sheikh Zayed | 10.5 |
| Shorouk | 10.6 |

Feature importance (gain %, q50):

| Feature | % |
|---|---|
| size_m2 | 37.7 |
| area | 14.7 |
| compound_tier | 12.1 |
| finishing | 9.1 |
| unit_type | 7.3 |
| furnished | 5.8 |
| bedrooms | 4.8 |
| view | 3.4 |
| amenities | 3.0 |
| floor | 1.2 |
| bathrooms | 0.9 |

Tuned params (Optuna, 20 trials): `{"learning_rate": 0.0501, "num_leaves": 44, "min_data_in_leaf": 157, "feature_fraction": 0.8274, "bagging_fraction": 0.9, "bagging_freq": 1, "lambda_l2": 0.0012}`

## Finishing cost (catalog rules + LightGBM adjustment)

| Metric | Value |
|---|---|
| MAPE, catalog rules + inflation only | 9.0% |
| MAPE, rules + ML adjustment | 5.7% |
| 80% cost range coverage | 80.6% |

Median actual cost per net m2 by package (EGP):

| Package | EGP/m2 |
|---|---|
| basic | 5,500 |
| premium | 19,500 |
| standard | 9,100 |
