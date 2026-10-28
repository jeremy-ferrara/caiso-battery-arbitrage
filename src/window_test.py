"""Rolling vs expanding training window for the ridge forecast.

Motivated by a result already seen: the expanding-window model over-forecasts the evening
peak (hours 17-20) in every year, with the bias shrinking over time. Hypothesis: the
expanding window keeps old, higher-priced evenings while prices fall. If so, a rolling
window should cut the evening bias. Everything except the training window is identical
(features, decision time, ridge penalty, MIN_TRAIN_DAYS, same LP and BatteryConfig), and
all models produce forecasts for the same days.

The windows only start to differ from the expanding model once history is longer than the
window (rolling_365 from 2024-07-08), so every table is also reported for the period from
ALL_DIVERGED onward, where all four models are genuinely different.

Writes data/processed/window_test_*.csv and prints markdown tables.

Usage:
    python src/window_test.py
"""
import time

import numpy as np
import pandas as pd

from backtest import HUBS, PROCESSED_DIR, available_months
from battery import BatteryConfig, filter_hub_market, solve_multiday, to_price_series
from forecast import forecast_driven_multiday, walk_forward_forecast
from load_prices import LOCAL_TZ, read_months

WINDOWS = {"expanding": None, "rolling_90": 90, "rolling_180": 180, "rolling_365": 365}
MODELS = [*WINDOWS, "persistence"]
EVENING_HOURS = range(17, 21)
ALL_DIVERGED = pd.Timestamp("2024-07-08")  # first day rolling_365 differs from expanding


def run(config: BatteryConfig = BatteryConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly forecasts (one column per model) and daily net profit per hub, model, and PF."""
    df = read_months(available_months())
    hourly, daily = [], []
    for hub in HUBS:
        prices = to_price_series(filter_hub_market(df, hub=hub))
        fc = None
        for name, window in WINDOWS.items():
            t0 = time.time()
            f = walk_forward_forecast(prices, window_days=window)
            print(f"{hub} {name}: {len(f)} hours in {time.time() - t0:.0f}s")
            if fc is None:
                fc = f[["actual", "persistence"]].copy()
            fc[name] = f["forecast"]
        hourly.append(fc.assign(hub=hub))

        pf = solve_multiday(fc["actual"], config).daily
        daily.append(pf.assign(hub=hub, model="perfect_foresight"))
        for name in MODELS:
            d = forecast_driven_multiday(prices, config, fc, column=name).daily
            daily.append(d.assign(hub=hub, model=name))

    hourly = pd.concat(hourly)
    local = hourly.index.tz_convert(LOCAL_TZ)
    hourly = hourly.assign(date=pd.to_datetime(local.date), hour=local.hour, year=local.year)
    daily = pd.concat(daily).rename_axis("date").reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return hourly, daily


def errors(hourly: pd.DataFrame) -> pd.DataFrame:
    return hourly[MODELS].sub(hourly["actual"], axis=0)


def evening_bias_by_year(hourly: pd.DataFrame) -> pd.DataFrame:
    """Mean forecast - actual ($/MWh) over hours 17-20, per hub and year."""
    e = errors(hourly).assign(hub=hourly["hub"], year=hourly["year"])[hourly["hour"].isin(EVENING_HOURS)]
    return e.groupby(["hub", "year"])[MODELS].mean()


def error_by_hour(hourly: pd.DataFrame, metric: str) -> pd.DataFrame:
    e = errors(hourly).assign(hub=hourly["hub"], hour=hourly["hour"])
    g = e.groupby(["hub", "hour"])[MODELS]
    return g.apply(lambda x: x.abs().mean()) if metric == "mae" else g.apply(lambda x: np.sqrt((x ** 2).mean()))


def overall_accuracy(hourly: pd.DataFrame) -> pd.DataFrame:
    e = errors(hourly).assign(hub=hourly["hub"])
    g = e.groupby("hub")[MODELS]
    ev = e[hourly["hour"].isin(EVENING_HOURS)].groupby("hub")[MODELS]
    return pd.concat({"mae": g.apply(lambda x: x.abs().mean()),
                      "rmse": g.apply(lambda x: np.sqrt((x ** 2).mean())),
                      "bias": g.mean(),
                      "evening_mae": ev.apply(lambda x: x.abs().mean()),
                      "evening_bias": ev.mean()}, names=["metric"]).swaplevel().sort_index()


def daily_spread(hourly: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean daily max - min ($/MWh) of each forecast and of actual prices."""
    cols = ["actual", *MODELS]
    spread = hourly.groupby(["hub", "date"])[cols].agg(lambda x: x.max() - x.min())
    spread = spread.reset_index().assign(year=lambda d: d["date"].dt.year)
    return spread.groupby(by)[cols].mean()


def dispatch_summary(daily: pd.DataFrame) -> pd.DataFrame:
    pf = daily[daily["model"] == "perfect_foresight"].groupby("hub")["net_profit"].sum()
    g = daily.groupby(["hub", "model"]).agg(
        days=("date", "size"), net_profit_usd=("net_profit", "sum"),
        mean_discharged_mwh=("discharged_mwh", "mean"),
        idle_days=("discharged_mwh", lambda x: int((x < 1e-3).sum())),
        loss_days=("net_profit", lambda x: int((x < -0.01).sum())))
    g["capture_pct"] = 100 * g["net_profit_usd"] / g.index.get_level_values("hub").map(pf)
    return g


def capture_by_year(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.assign(year=daily["date"].dt.year)
    wide = d.pivot_table(index=["hub", "year"], columns="model", values="net_profit", aggfunc="sum")
    return 100 * wide[MODELS].div(wide["perfect_foresight"], axis=0)


def md(df: pd.DataFrame, floatfmt: str = ",.1f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


if __name__ == "__main__":
    hourly, daily = run()
    tables = {
        "evening_bias_by_year": evening_bias_by_year(hourly),
        "overall_accuracy": overall_accuracy(hourly),
        "mae_by_hour": error_by_hour(hourly, "mae"),
        "rmse_by_hour": error_by_hour(hourly, "rmse"),
        "daily_spread": daily_spread(hourly, ["hub"]),
        "daily_spread_by_year": daily_spread(hourly, ["hub", "year"]),
        "dispatch": dispatch_summary(daily),
        "capture_by_year": capture_by_year(daily),
    }
    late_h, late_d = hourly[hourly["date"] >= ALL_DIVERGED], daily[daily["date"] >= ALL_DIVERGED]
    tables |= {
        "overall_accuracy_all_diverged": overall_accuracy(late_h),
        "daily_spread_all_diverged": daily_spread(late_h, ["hub"]),
        "dispatch_all_diverged": dispatch_summary(late_d),
    }
    for name, table in tables.items():
        print(f"\n### {name}\n\n{md(table)}")
        table.to_csv(PROCESSED_DIR / f"window_test_{name}.csv")
    hourly.to_parquet(PROCESSED_DIR / "window_test_forecasts_hourly.parquet")
    daily.to_csv(PROCESSED_DIR / "window_test_daily.csv", index=False)
