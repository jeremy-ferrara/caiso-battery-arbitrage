"""Run perfect-foresight and rule-based dispatch over every downloaded month at both hubs
(day-ahead prices) and save one row per Pacific day, hub, and strategy.

Writes data/processed/daily_results.parquet and a .csv copy for Tableau.

Rule-based has no rows for the first days of the series, where it lacks price history
(see rule_based.MIN_HISTORY_DAYS); compare strategies on matching dates.

Usage:
    python src/backtest.py
"""
import time
from pathlib import Path

import pandas as pd

from battery import BatteryConfig, filter_hub_market, solve_multiday, to_price_series
from load_prices import RAW_DIR, read_months
from rule_based import rule_based_multiday

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
HUBS = ["SP15", "NP15"]
STRATEGIES = {
    "perfect_foresight": solve_multiday,
    "rule_based": rule_based_multiday,
}
OUTPUT_COLUMNS = ["date", "hub", "strategy", "gross_revenue_usd", "degradation_cost_usd",
                  "net_profit_usd", "charged_mwh", "discharged_mwh"]


def available_months() -> list[pd.Period]:
    return sorted(pd.Period(p.stem.removeprefix("caiso_lmp_"), freq="M") for p in RAW_DIR.glob("caiso_lmp_*.parquet"))


def run(config: BatteryConfig = BatteryConfig()) -> pd.DataFrame:
    months = available_months()
    print(f"{len(months)} months: {months[0]} -> {months[-1]}")
    df = read_months(months)

    frames = []
    for hub in HUBS:
        prices = to_price_series(filter_hub_market(df, hub=hub))
        for name, strategy in STRATEGIES.items():
            t0 = time.time()
            daily = strategy(prices, config).daily.reset_index()
            print(f"{hub} {name}: {len(daily)} days in {time.time() - t0:.0f}s, "
                  f"net ${daily['net_profit'].sum():,.0f}")
            frames.append(daily.assign(hub=hub, strategy=name))

    out = pd.concat(frames, ignore_index=True).rename(columns={
        "pacific_date": "date",
        "gross_revenue": "gross_revenue_usd",
        "degradation_cost": "degradation_cost_usd",
        "net_profit": "net_profit_usd",
    })
    out["date"] = pd.to_datetime(out["date"])
    return out[OUTPUT_COLUMNS].sort_values(["hub", "strategy", "date"], ignore_index=True)


if __name__ == "__main__":
    results = run()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "daily_results.parquet"
    results.to_parquet(path, index=False)
    results.to_csv(path.with_suffix(".csv"), index=False)
    print(f"Saved {len(results):,} rows -> {path} (+ .csv)")
