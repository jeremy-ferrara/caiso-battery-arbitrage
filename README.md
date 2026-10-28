# California Battery Storage Arbitrage

How much could a 100 MW / 400 MWh battery have earned arbitraging CAISO day-ahead prices, and how much of that is capturable without perfect foresight?

**Status:** backtest and analysis complete. Full write-up: [`reports/findings.md`](reports/findings.md).

## Headline results

Day-ahead hourly prices at the SP15 and NP15 trading hubs, Jul 2023 - Aug 2026. Strategies are compared on the 1,121 days per hub where all three have results (Aug 7 2023 - Aug 31 2026). Figures are net of a $15/MWh degradation charge.

| | SP15 | NP15 | Both hubs |
|---|---:|---:|---:|
| Perfect foresight (upper bound) | $11.70M | $5.73M | $17.43M |
| Rule-based | $6.64M (56.8%) | $3.30M (57.6%) | $9.94M (57.0%) |
| Forecast-driven | $10.66M (91.1%) | $4.55M (79.3%) | $15.21M (87.2%) |

Percentages are capture rate: net profit as a share of perfect foresight on the same days.

- **The rule-based gap is mostly volume, not foresight.** The rule's single charge/discharge block moves at most half a cycle per day. Against perfect foresight capped at the same volume it captures 89.5% (SP15) and 79.5% (NP15).
- **The forecast doesn't beat a naive baseline over the full period.** The ridge forecast is less accurate than yesterday's prices, and dispatching on yesterday's prices captures slightly more (89.4% pooled). The shortfall is entirely in the first 11 months, when the model had little training data. From Jul 2024 on, the ridge forecast beats that baseline at both hubs on accuracy and profit, by about 1 capture point.
- **Fixing the evening bias didn't fix profit.** The forecast over-predicts evening peaks. A post-hoc rolling-window test removed most of that bias but moved capture by at most 2.6 points, in both directions.
- **Arbitrage value is falling.** SP15 perfect-foresight value fell from $47.7/kW-yr (2024) to $19.4/kW-yr (Sep 2025 - Aug 2026) as daily price spreads narrowed.

These figures cover day-ahead energy arbitrage at hub prices only. Real-time trading, ancillary services, resource adequacy, and nodal settlement are not modeled.

## Methodology

- **Data.** CAISO day-ahead hourly LMPs via `gridstatus`. CAISO's public OASIS API keeps only about the trailing 3 years of LMP data, so the range is Jul 2023 - Aug 2026 rather than the originally planned 2021-2025.
- **Battery model.** A `cvxpy` linear program with:
  - 100 MW power limit and 400 MWh capacity;
  - 85% round-trip efficiency;
  - max 1 cycle per day;
  - $15/MWh degradation cost;
  - state of charge reset to 50% at each Pacific midnight.

  Every strategy uses the same parameters.
- **Strategies.**
  - **Perfect foresight:** LP on the day's actual prices (upper bound).
  - **Rule-based:** fixed charge and discharge blocks chosen from the trailing 30-day average hourly profile.
  - **Forecast-driven:** the same LP solved on a price forecast, then settled at actual prices.
- **Forecast.** One ridge regression per hour of day. Features are lagged day-ahead prices (D-1, D-7, 7-day mean), the previous day's mean and spread, day of week, and season.
  - The model is refit daily on an expanding window.
  - It uses only data published before the D-1 bid deadline.
  - A unit test scrambles all prices from day D onward and checks that D's forecast is unchanged.

## Repo layout

| Path | Purpose |
| --- | --- |
| `src/load_prices.py` | Pull CAISO prices to `data/raw` |
| `src/battery.py` | Battery config and perfect-foresight LP |
| `src/rule_based.py` | Rule-based strategy |
| `src/forecast.py` | Walk-forward ridge forecast and forecast-driven strategy |
| `src/backtest.py` | Run all strategies at both hubs |
| `src/analysis.py` | Capture rates, accuracy metrics, price trends |
| `src/window_test.py` | Rolling vs expanding training window test |
| `src/tableau_export.py` | Dashboard CSVs |
| `tests/` | Unit tests, including the no-lookahead tests |
| `reports/findings.md` | Full results write-up |
| `tableau/` | Dashboard field notes |
| `data/raw`, `data/processed` | Price data and outputs (not committed) |

## Reproducing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/load_prices.py --start 2023-07-01 --end 2026-08-31
python src/backtest.py
python src/analysis.py
python src/window_test.py
python src/tableau_export.py
pytest
```

The price pull takes a while because CAISO's API is rate-limited. Everything after it runs in a few minutes.
