"""Write one tidy, denormalized CSV per Tableau dashboard view to data/processed.

Inputs come from earlier steps: run src/backtest.py and src/window_test.py first.
Strategy comparisons use only the dates where every strategy has a result (Aug 7 2023 -
Aug 31 2026, 1,121 days per hub), so cumulative and average figures are like-for-like.
The headline forecast strategy is the expanding-window ridge forecast; the persistence LP
and rolling-window variants appear only in forecast_accuracy.csv.

Outputs (data/processed):
    daily_pnl.csv          one row per date, hub, strategy
    hourly_dispatch.csv    one row per Pacific hour of day, month, hub, strategy
    capture_rates.csv      one row per year, season, hub, strategy (with the profit sums
                           behind each rate, so Tableau can re-aggregate as SUM/SUM)
    price_trends.csv       one row per month and hub (day-ahead prices, full data range)
    forecast_accuracy.csv  one row per period, hub, model, Pacific hour of day

Usage:
    python src/tableau_export.py
"""
import numpy as np
import pandas as pd

from analysis import SEASONS, day_ahead_prices, daily_spread, load_results, matched_wide, negative_hours
from backtest import HUBS, PROCESSED_DIR, STRATEGIES, available_months
from battery import BatteryConfig, filter_hub_market, to_price_series
from load_prices import LOCAL_TZ, read_months
from window_test import ALL_DIVERGED

STRATEGY_LABELS = {
    "perfect_foresight": "Perfect Foresight",
    "rule_based": "Rule-Based",
    "forecast_driven": "Forecast-Driven",
}
MODEL_LABELS = {
    "expanding": "Ridge: Expanding Window (headline)",
    "rolling_90": "Ridge: 90-Day Rolling Window",
    "rolling_180": "Ridge: 180-Day Rolling Window",
    "rolling_365": "Ridge: 365-Day Rolling Window",
    "persistence": "Persistence (Yesterday's Prices)",
}


def matched_results() -> pd.DataFrame:
    results = load_results()
    results = results[results["strategy"].isin(STRATEGY_LABELS)]
    keys = matched_wide(results)[["hub", "date"]]
    return results.merge(keys, on=["hub", "date"])


def daily_pnl(results: pd.DataFrame) -> pd.DataFrame:
    r = results.sort_values(["hub", "strategy", "date"])
    r["cumulative"] = r.groupby(["hub", "strategy"])["net_profit_usd"].cumsum()
    return pd.DataFrame({
        "Date": r["date"].dt.date,
        "Hub": r["hub"],
        "Strategy": r["strategy"].map(STRATEGY_LABELS),
        "Net Profit ($)": r["net_profit_usd"].round(2),
        "Cumulative Net Profit ($)": r["cumulative"].round(2),
    })


def hourly_dispatch(matched_dates: pd.DataFrame, config: BatteryConfig = BatteryConfig()) -> pd.DataFrame:
    """Average MW by Pacific hour of day and month, over every day in the month (idle hours
    count as 0). Strategies are re-run here because daily_results only keeps daily totals."""
    df = read_months(available_months())
    fc_all = pd.read_parquet(PROCESSED_DIR / "forecasts_hourly.parquet")
    frames = []
    for hub in HUBS:
        prices = to_price_series(filter_hub_market(df, hub=hub))
        fc = fc_all[fc_all["hub"] == hub].set_index("interval_start_utc").drop(columns="hub")
        keep = set(matched_dates.loc[matched_dates["hub"] == hub, "date"].dt.date)
        for name in STRATEGY_LABELS:
            result = STRATEGIES[name](prices, config, fc)
            local = result.charge_mw.index.tz_convert(LOCAL_TZ)
            h = pd.DataFrame({"charge": result.charge_mw.to_numpy(), "discharge": result.discharge_mw.to_numpy(),
                              "date": local.date, "hour": local.hour,
                              "month": local.tz_localize(None).to_period("M").start_time})
            h = h[h["date"].isin(keep)]
            g = h.groupby(["hour", "month"])[["charge", "discharge"]].mean().reset_index()
            frames.append(g.assign(hub=hub, strategy=name))
    out = pd.concat(frames, ignore_index=True)
    out[["charge", "discharge"]] = out[["charge", "discharge"]].clip(lower=0)  # drop -1e-9 solver noise
    return pd.DataFrame({
        "Hour of Day (Pacific)": out["hour"],
        "Month": out["month"].dt.date,
        "Hub": out["hub"],
        "Strategy": out["strategy"].map(STRATEGY_LABELS),
        "Avg Charge (MW)": out["charge"].round(2),
        "Avg Discharge (MW)": out["discharge"].round(2),
    }).sort_values(["Hub", "Strategy", "Month", "Hour of Day (Pacific)"])


def capture_rates(results: pd.DataFrame) -> pd.DataFrame:
    """Capture = strategy net profit / perfect-foresight net profit, per year x season x hub.
    Season is assigned by calendar month (December counts in its own calendar year's winter)."""
    r = results.assign(year=results["date"].dt.year, season=results["date"].dt.month.map(SEASONS))
    g = r.pivot_table(index=["year", "season", "hub"], columns="strategy",
                      values="net_profit_usd", aggfunc="sum")
    days = r[r["strategy"] == "perfect_foresight"].groupby(["year", "season", "hub"]).size()
    rows = []
    for name in ["rule_based", "forecast_driven"]:
        rows.append(pd.DataFrame({
            "Year": g.index.get_level_values("year"),
            "Season": g.index.get_level_values("season"),
            "Hub": g.index.get_level_values("hub"),
            "Strategy": STRATEGY_LABELS[name],
            "Days": days.reindex(g.index).to_numpy(),
            "Strategy Net Profit ($)": g[name].round(2).to_numpy(),
            "Perfect Foresight Net Profit ($)": g["perfect_foresight"].round(2).to_numpy(),
            "Capture Rate (%)": (100 * g[name] / g["perfect_foresight"]).round(2).to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def price_trends() -> pd.DataFrame:
    prices = day_ahead_prices()
    spread = daily_spread(prices).assign(month=lambda d: d["date"].dt.to_period("M"))
    s = spread.groupby(["month", "hub"])[["max_min_spread", "top4_bottom4_spread"]].mean()
    p = prices.groupby(["month", "hub"])["lmp"].mean().rename("avg_price")
    neg = negative_hours(prices).set_index(["month", "hub"])
    out = s.join(p).join(neg).reset_index()
    return pd.DataFrame({
        "Month": out["month"].dt.start_time.dt.date,
        "Hub": out["hub"],
        "Avg Daily Spread Max-Min ($/MWh)": out["max_min_spread"].round(2),
        "Avg Daily Spread Top 4 vs Bottom 4 Hours ($/MWh)": out["top4_bottom4_spread"].round(2),
        "Negative Price Hours": out["negative_hours"].astype(int),
        "Hours in Month": out["hours"].astype(int),
        "Avg Day-Ahead Price ($/MWh)": out["avg_price"].round(2),
    })


def forecast_accuracy() -> pd.DataFrame:
    hourly = pd.read_parquet(PROCESSED_DIR / "window_test_forecasts_hourly.parquet")
    periods = {"Full Period (Aug 2023 - Aug 2026)": hourly,
               f"All Windows Differ ({ALL_DIVERGED:%b %Y} - Aug 2026)": hourly[hourly["date"] >= ALL_DIVERGED]}
    rows = []
    for period, h in periods.items():
        for model, label in MODEL_LABELS.items():
            e = (h[model] - h["actual"]).groupby([h["hub"], h["hour"]])
            g = pd.DataFrame({"mae": e.apply(lambda x: x.abs().mean()),
                              "rmse": e.apply(lambda x: np.sqrt((x ** 2).mean())),
                              "bias": e.mean()}).reset_index()
            rows.append(g.assign(period=period, model=label))
    out = pd.concat(rows, ignore_index=True)
    return pd.DataFrame({
        "Period": out["period"],
        "Hub": out["hub"],
        "Model": out["model"],
        "Hour of Day (Pacific)": out["hour"],
        "MAE ($/MWh)": out["mae"].round(3),
        "RMSE ($/MWh)": out["rmse"].round(3),
        "Bias Forecast minus Actual ($/MWh)": out["bias"].round(3),
    })


if __name__ == "__main__":
    results = matched_results()
    tables = {
        "daily_pnl": daily_pnl(results),
        "hourly_dispatch": hourly_dispatch(results[["hub", "date"]].drop_duplicates()),
        "capture_rates": capture_rates(results),
        "price_trends": price_trends(),
        "forecast_accuracy": forecast_accuracy(),
    }
    for name, table in tables.items():
        path = PROCESSED_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"{path.name}: {len(table):,} rows, {path.stat().st_size / 1024:,.0f} KB")
