# California Battery Storage Arbitrage

**Status: in progress**

## Question

How much revenue could a 100 MW / 400 MWh battery have earned by arbitraging CAISO day-ahead and real-time prices, and how much of that is capturable without perfect foresight?

**Data range: July 2023 - August 2026.** CAISO's public OASIS API (queried via `gridstatus`) only keeps roughly the trailing 3 years of LMP data, so that's as far back as this project can pull. The original plan called for 2021-2025; 2021 and most of 2022 are not available through this API.

## Planned approach

1. Pull CAISO day-ahead and real-time prices with `gridstatus`.
2. Model the battery as a linear program (`cvxpy`) with round-trip efficiency, state-of-charge limits, cycle limits, and a degradation cost.
3. Compare three strategies:
   - **Perfect foresight**: optimal dispatch on known prices (upper bound)
   - **Rule-based**: simple charge-low / discharge-high schedule
   - **Forecast-driven**: optimal dispatch on forecast prices, settled at actuals
4. Publish the results as a Tableau Public dashboard and a one-page market note.

## Repo layout

| Folder | Purpose |
| --- | --- |
| `data/raw`, `data/processed` | Downloaded and cleaned price data (not committed) |
| `src` | Reusable code (data loading, optimizer, strategies) |
| `notebooks` | Exploration and result write-ups |
| `tableau` | Dashboard extracts and workbook notes |
| `reports` | Market note |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
