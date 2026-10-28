"""Summary statistics for the backtest and day-ahead price trends.

Reads data/processed/daily_results.parquet (from backtest.py) and the raw price files.
Prints markdown tables and writes the underlying tables to data/processed/analysis_*.csv.

Capture rate = sum of a strategy's net profit / sum of perfect-foresight net profit over the
same set of dates (ratio of sums, not a mean of daily ratios). Only dates where every
strategy has a result are used, so all strategies are compared on identical days.

Forecast accuracy (from data/processed/forecasts_hourly.parquet) is reported separately from
profit: MAE and RMSE by Pacific hour of day, plus two shape measures -- MAE after removing
each day's mean from both forecast and actual, and the mean within-day rank correlation
between forecast and actual hourly prices.

Usage:
    python src/analysis.py
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest import HUBS, PROCESSED_DIR, available_months
from battery import BatteryConfig, filter_hub_market, solve_multiday, to_price_series
from load_prices import LOCAL_TZ, read_months
from rule_based import stored_energy_per_day_mwh

SEASONS = {12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)",
           3: "Spring (MAM)", 4: "Spring (MAM)", 5: "Spring (MAM)",
           6: "Summer (JJA)", 7: "Summer (JJA)", 8: "Summer (JJA)",
           9: "Fall (SON)", 10: "Fall (SON)", 11: "Fall (SON)"}
SEASON_ORDER = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"]
COMPARED = ["rule_based", "forecast_driven", "persistence_lp"]  # each vs perfect_foresight
FORECASTS = ["forecast", "persistence", "mean7"]


def load_results() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "daily_results.parquet")


def matched_wide(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (hub, date) with every strategy's net profit; unmatched dates dropped."""
    wide = results.pivot_table(index=["hub", "date"], columns="strategy",
                               values="net_profit_usd").dropna().reset_index()
    wide["year"] = wide["date"].dt.year
    wide["season"] = pd.Categorical(wide["date"].dt.month.map(SEASONS), SEASON_ORDER, ordered=True)
    return wide


def capture(wide: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = wide.groupby(by, observed=True).agg(days=("date", "size"), pf_net_usd=("perfect_foresight", "sum"),
                                            **{f"{s}_net_usd": (s, "sum") for s in COMPARED})
    for s in COMPARED:
        g[f"{s}_capture_pct"] = 100 * g[f"{s}_net_usd"] / g["pf_net_usd"]
    return g


def annual_profit(results: pd.DataFrame, config: BatteryConfig = BatteryConfig()) -> pd.DataFrame:
    """Net profit per calendar year per hub and strategy, with days covered and $/kW-yr.
    $/kW-yr is left blank for partial years: profit is strongly seasonal, so scaling a
    partial year up by day count would be biased."""
    r = results.assign(year=results["date"].dt.year)
    g = r.groupby(["hub", "strategy", "year"]).agg(days=("date", "size"), net_profit_usd=("net_profit_usd", "sum"))
    g["usd_per_kw_yr"] = (g["net_profit_usd"] / (config.power_mw * 1000)).where(g["days"] >= 365)
    return g


def trailing_12m(results: pd.DataFrame, config: BatteryConfig = BatteryConfig()) -> pd.DataFrame:
    end = results["date"].max()
    start = (end - pd.DateOffset(years=1)) + pd.Timedelta(days=1)
    r = results[results["date"] >= start]
    g = r.groupby(["hub", "strategy"]).agg(days=("date", "size"), net_profit_usd=("net_profit_usd", "sum"))
    g["usd_per_kw_yr"] = g["net_profit_usd"] / (config.power_mw * 1000)
    g.attrs["window"] = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    return g


def day_ahead_prices() -> pd.DataFrame:
    df = read_months(available_months())
    frames = []
    for hub in HUBS:
        s = to_price_series(filter_hub_market(df, hub=hub))
        local = s.index.tz_convert(LOCAL_TZ)
        frames.append(pd.DataFrame({"hub": hub, "lmp": s.to_numpy(), "date": pd.to_datetime(local.date),
                                    "month": local.tz_localize(None).to_period("M")}))
    return pd.concat(frames, ignore_index=True)


def daily_spread(prices: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Per hub and day: max - min price, and mean of top-n hours minus mean of bottom-n hours."""
    def one_day(x):
        v = x.sort_values().to_numpy()
        return pd.Series({"max_min_spread": v[-1] - v[0], f"top{n}_bottom{n}_spread": v[-n:].mean() - v[:n].mean()})
    out = prices.groupby(["hub", "date"])["lmp"].apply(one_day).unstack().reset_index()
    out["year"] = out["date"].dt.year
    return out


def negative_hours(prices: pd.DataFrame) -> pd.DataFrame:
    """Count of day-ahead hours with LMP < 0, per hub and month."""
    return (prices.assign(neg=prices["lmp"] < 0)
            .groupby(["hub", "month"])["neg"].agg(negative_hours="sum", hours="size").reset_index())


def same_months_by_year(table: pd.DataFrame, value: str, months: range, agg: str) -> pd.DataFrame:
    """Compare years over a common set of calendar months (like-for-like for partial years)."""
    t = table[table["month"].dt.month.isin(months)].assign(year=table["month"].dt.year)
    covered = t.groupby("year")["month"].nunique()
    full_years = covered[covered == len(months)].index  # drop years missing any of the months
    return t[t["year"].isin(full_years)].pivot_table(index="year", columns="hub", values=value, aggfunc=agg)


def concentration(results: pd.DataFrame, top_frac: float = 0.10) -> pd.DataFrame:
    rows = []
    for (hub, strat), g in results.groupby(["hub", "strategy"]):
        v = g["net_profit_usd"].sort_values(ascending=False)
        k = max(1, round(len(v) * top_frac))
        rows.append({"hub": hub, "strategy": strat, "days": len(v),
                     f"top_{int(top_frac * 100)}pct_days_share_pct": 100 * v.iloc[:k].sum() / v.sum(),
                     "best_day": g.loc[v.index[0], "date"].date(), "best_day_usd": v.iloc[0],
                     "negative_days": int((v < -0.01).sum()),  # ignore LP solver noise around 0
                     "idle_days": int((g["discharged_mwh"] < 1e-3).sum())})  # LP leaves ~1e-9 noise
    return pd.DataFrame(rows).set_index(["hub", "strategy"])


def volume_matched_pf(results: pd.DataFrame, config: BatteryConfig = BatteryConfig()) -> pd.DataFrame:
    """Diagnostic: perfect foresight limited to the rule's daily volume (one charge block and one
    discharge block can move at most stored_energy_per_day_mwh, i.e. half a cycle with SoC pinned
    at 50%). Rule net profit / this = capture with the volume handicap removed. The forecast
    strategies are not volume-limited, so their capture is shown against both for reference."""
    limited = replace(config, max_cycles_per_day=stored_energy_per_day_mwh(config) * config.eta_discharge
                      / config.energy_mwh)
    df = read_months(available_months())
    rows = []
    for hub in HUBS:
        daily = solve_multiday(to_price_series(filter_hub_market(df, hub=hub)), limited).daily
        rows.append(pd.DataFrame({"hub": hub, "date": pd.to_datetime(daily.index),
                                  "pf_volume_matched": daily["net_profit"].to_numpy()}))
    wide = matched_wide(results).merge(pd.concat(rows), on=["hub", "date"])
    g = wide.groupby("hub").agg(pf_net_usd=("perfect_foresight", "sum"),
                                pf_volume_matched_net_usd=("pf_volume_matched", "sum"),
                                **{f"{s}_net_usd": (s, "sum") for s in COMPARED})
    for s in COMPARED:
        g[f"{s}_vs_pf_pct"] = 100 * g[f"{s}_net_usd"] / g["pf_net_usd"]
        g[f"{s}_vs_volume_matched_pf_pct"] = 100 * g[f"{s}_net_usd"] / g["pf_volume_matched_net_usd"]
    return g


def load_forecasts() -> pd.DataFrame:
    fc = pd.read_parquet(PROCESSED_DIR / "forecasts_hourly.parquet")
    local = pd.DatetimeIndex(fc["interval_start_utc"]).tz_convert(LOCAL_TZ)
    return fc.assign(date=pd.to_datetime(local.date), hour=local.hour, year=local.year)


def forecast_errors_by_hour(fc: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE ($/MWh) per hub and Pacific hour for each forecast."""
    err = fc[FORECASTS].sub(fc["actual"], axis=0).assign(hub=fc["hub"], hour=fc["hour"])
    g = err.groupby(["hub", "hour"])
    mae = g[FORECASTS].apply(lambda x: x.abs().mean()).add_suffix("_mae")
    rmse = g[FORECASTS].apply(lambda x: np.sqrt((x ** 2).mean())).add_suffix("_rmse")
    return mae.join(rmse)[[f"{f}_{m}" for m in ("mae", "rmse") for f in FORECASTS]]


def forecast_summary(fc: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Level and shape accuracy. demeaned_mae: MAE after subtracting each day's mean from both
    forecast and actual (pure shape error). rank_corr: mean over days of the Spearman correlation
    between the day's forecast and actual hourly prices. spread: mean daily max - min."""
    day_key = ["hub", "date"]
    rows = {}
    for f in FORECASTS:
        e = fc[f] - fc["actual"]
        day_mean_err = e.groupby([fc[k] for k in day_key]).transform("mean")
        ranks = fc.groupby(day_key)[[f, "actual"]].rank()
        daily = (fc.assign(rf=ranks[f], ra=ranks["actual"]).groupby(day_key)
                 .apply(lambda d: d["rf"].corr(d["ra"]), include_groups=False).rename("rank_corr"))
        spread = fc.groupby(day_key)[f].agg(lambda x: x.max() - x.min()).rename("spread")
        per_day = pd.concat([daily, spread], axis=1).reset_index()
        per_day = per_day.merge(fc[day_key + [k for k in by if k not in day_key]].drop_duplicates(), on=day_key)
        hourly = fc.assign(abs_e=e.abs(), sq_e=e ** 2, bias=e, dm=(e - day_mean_err).abs())
        h = hourly.groupby(by).agg(mae=("abs_e", "mean"), rmse=("sq_e", lambda x: np.sqrt(x.mean())),
                                   bias=("bias", "mean"), demeaned_mae=("dm", "mean"))
        d = per_day.groupby(by).agg(rank_corr=("rank_corr", "mean"), forecast_spread=("spread", "mean"))
        rows[f] = h.join(d)
    actual_spread = (fc.groupby(day_key)["actual"].agg(lambda x: x.max() - x.min()).rename("spread").reset_index()
                     .merge(fc[day_key + [k for k in by if k not in day_key]].drop_duplicates(), on=day_key)
                     .groupby(by)["spread"].mean())
    out = pd.concat(rows, names=["forecast"]).reorder_levels([*range(1, len(by) + 1), 0]).sort_index()
    out["actual_spread"] = out.index.droplevel("forecast").map(actual_spread)
    return out


def plan_vs_realized(results: pd.DataFrame) -> pd.DataFrame:
    """Per hub and strategy: days, idle days, loss days, mean MWh discharged per day."""
    r = results[results["strategy"].isin(["perfect_foresight", *COMPARED])]
    return r.groupby(["hub", "strategy"]).agg(
        days=("date", "size"), net_profit_usd=("net_profit_usd", "sum"),
        mean_discharged_mwh=("discharged_mwh", "mean"),
        idle_days=("discharged_mwh", lambda x: int((x < 1e-3).sum())),
        loss_days=("net_profit_usd", lambda x: int((x < -0.01).sum())))


def md(df: pd.DataFrame, floatfmt: str = ",.1f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


if __name__ == "__main__":
    results = load_results()
    wide = matched_wide(results)
    prices = day_ahead_prices()
    spread = daily_spread(prices)
    neg = negative_hours(prices)

    fc = load_forecasts()

    tables = {
        "capture_overall": capture(wide, ["hub"]),
        "capture_overall_both_hubs": capture(wide.assign(all="all"), ["all"]),
        "capture_by_year": capture(wide, ["year", "hub"]),
        "capture_by_year_both_hubs": capture(wide, ["year"]),
        "capture_by_season": capture(wide, ["season", "hub"]),
        "capture_by_season_both_hubs": capture(wide, ["season"]),
        "annual_profit": annual_profit(results),
        "trailing_12m": trailing_12m(results),
        "concentration": concentration(results),
        "volume_matched_pf": volume_matched_pf(results),
        "strategy_activity": plan_vs_realized(results.merge(wide[["hub", "date"]], on=["hub", "date"])),
        "forecast_error_by_hour": forecast_errors_by_hour(fc),
        "forecast_summary": forecast_summary(fc, ["hub"]),
        "forecast_summary_by_year": forecast_summary(fc, ["hub", "year"]),
        "spread_by_year": spread.groupby(["year", "hub"])[["max_min_spread", "top4_bottom4_spread"]].mean(),
        "neg_hours_by_month": neg.pivot_table(index="month", columns="hub", values="negative_hours"),
        "neg_hours_by_year": neg.assign(year=neg["month"].dt.year)
                               .groupby(["year", "hub"])[["negative_hours", "hours"]].sum(),
        "neg_hours_jan_aug": same_months_by_year(neg, "negative_hours", range(1, 9), "sum"),
        "neg_hours_jul_dec": same_months_by_year(neg, "negative_hours", range(7, 13), "sum"),
        "spread_jan_aug": same_months_by_year(
            spread.assign(month=spread["date"].dt.to_period("M")), "top4_bottom4_spread", range(1, 9), "mean"),
        "spread_jul_dec": same_months_by_year(
            spread.assign(month=spread["date"].dt.to_period("M")), "top4_bottom4_spread", range(7, 13), "mean"),
        "mean_price_by_year": prices.assign(year=prices["date"].dt.year)
                                    .pivot_table(index="year", columns="hub", values="lmp", aggfunc="mean"),
    }

    for name, table in tables.items():
        title = name + (f" ({table.attrs['window']})" if "window" in table.attrs else "")
        print(f"\n### {title}\n\n{md(table)}")
        table.to_csv(PROCESSED_DIR / f"analysis_{name}.csv")
    spread.to_csv(PROCESSED_DIR / "analysis_daily_spread.csv", index=False)
    neg.to_csv(PROCESSED_DIR / "analysis_negative_hours_by_month.csv", index=False)
