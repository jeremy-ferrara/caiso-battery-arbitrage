"""Summary statistics for the backtest and day-ahead price trends.

Reads data/processed/daily_results.parquet (from backtest.py) and the raw price files.
Prints markdown tables and writes the underlying tables to data/processed/analysis_*.csv.

Capture rate = sum of rule-based net profit / sum of perfect-foresight net profit over the
same set of dates (ratio of sums, not a mean of daily ratios). Only dates where both
strategies have a result are used.

Usage:
    python src/analysis.py
"""
from dataclasses import replace

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


def load_results() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "daily_results.parquet")


def matched_wide(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (hub, date) with both strategies' net profit; unmatched dates dropped."""
    wide = results.pivot_table(index=["hub", "date"], columns="strategy",
                               values="net_profit_usd").dropna().reset_index()
    wide["year"] = wide["date"].dt.year
    wide["season"] = pd.Categorical(wide["date"].dt.month.map(SEASONS), SEASON_ORDER, ordered=True)
    return wide


def capture(wide: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = wide.groupby(by, observed=True).agg(days=("date", "size"),
                                            pf_net_usd=("perfect_foresight", "sum"),
                                            rule_net_usd=("rule_based", "sum"))
    g["capture_pct"] = 100 * g["rule_net_usd"] / g["pf_net_usd"]
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
                     "idle_days": int((g["discharged_mwh"] == 0).sum())})
    return pd.DataFrame(rows).set_index(["hub", "strategy"])


def volume_matched_pf(results: pd.DataFrame, config: BatteryConfig = BatteryConfig()) -> pd.DataFrame:
    """Diagnostic: perfect foresight limited to the rule's daily volume (one charge block and one
    discharge block can move at most stored_energy_per_day_mwh, i.e. half a cycle with SoC pinned
    at 50%). Rule net profit / this = capture with the volume handicap removed."""
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
                                rule_net_usd=("rule_based", "sum"))
    g["capture_vs_pf_pct"] = 100 * g["rule_net_usd"] / g["pf_net_usd"]
    g["capture_vs_volume_matched_pf_pct"] = 100 * g["rule_net_usd"] / g["pf_volume_matched_net_usd"]
    return g


def md(df: pd.DataFrame, floatfmt: str = ",.1f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


if __name__ == "__main__":
    results = load_results()
    wide = matched_wide(results)
    prices = day_ahead_prices()
    spread = daily_spread(prices)
    neg = negative_hours(prices)

    tables = {
        "capture_overall": capture(wide, ["hub"]),
        "capture_by_year": capture(wide, ["year", "hub"]),
        "capture_by_year_both_hubs": capture(wide, ["year"]),
        "capture_by_season": capture(wide, ["season", "hub"]),
        "capture_by_season_both_hubs": capture(wide, ["season"]),
        "annual_profit": annual_profit(results),
        "trailing_12m": trailing_12m(results),
        "concentration": concentration(results),
        "volume_matched_pf": volume_matched_pf(results),
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
