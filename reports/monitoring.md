# Monitoring snapshot

## Requests

- detect: 7 calls, p50 1.64 s, p95 2.89 s, low-confidence plans 43%
- evaluate: 10 calls, p50 0.06 s, p95 0.20 s

## Catalog

- Catalog date 2026-10-01, age 44 days, implied inflation since 1.8%

## Rent residual drift by area

Simulated batch (3,000 rents, New Capital +15%)

| Area | n | Shift vs training | Drift |
|---|---|---|---|
| 6th of October | 330 | +0.1% |  |
| Alexandria | 400 | -0.4% |  |
| Madinaty | 430 | +0.3% |  |
| Mostakbal City | 388 | +0.9% |  |
| New Cairo | 358 | -0.3% |  |
| New Capital | 497 | +13.9% | yes |
| Sheikh Zayed | 350 | -0.8% |  |
| Shorouk | 247 | -0.5% |  |
