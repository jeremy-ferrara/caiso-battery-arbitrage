"""Rule-based battery dispatch: a fixed daily charge block and discharge block chosen
from historical average prices, with no knowledge of the day's actual prices.

For each Pacific calendar day:

1. Build an hour-of-day price profile (Pacific local hour) from the trailing
   `lookback_days` calendar days strictly before that day. Days with fewer than
   `min_history_days` of history are skipped (no result row), not guessed at.
2. Size the trade. State of charge is pinned to the same start/end level every day
   (same BatteryConfig as the LP), and the schedule is one charge block plus one
   discharge block, so the energy moved in and out of storage must balance. With the
   default config (SoC pinned at 50%), that caps a day at 200 MWh stored -- half the
   LP's one-cycle allowance, which the LP can use in full by discharging in the
   morning, recharging midday, and discharging again in the evening.
3. Pick the two non-overlapping contiguous blocks (no wrap past midnight) that
   maximize expected profit on the historical profile. Each block is the minimum
   number of whole hours needed at full power; power is spread evenly across it.
4. If even the best blocks don't beat degradation cost on the historical profile,
   stay idle that day.

The schedule is then settled at the day's actual prices.

Usage:
    python src/rule_based.py
"""
import datetime
import math

import numpy as np
import pandas as pd

from battery import BatteryConfig, DispatchResult, filter_hub_market, to_price_series
from load_prices import LOCAL_TZ, read_months

LOOKBACK_DAYS = 30
MIN_HISTORY_DAYS = 7


def hour_of_day_profile(prices: pd.Series, local_tz: str = LOCAL_TZ) -> pd.Series:
    """Mean price by Pacific local hour (0-23). Hours absent from the input are NaN."""
    hours = prices.index.tz_convert(local_tz).hour
    return prices.groupby(hours).mean().reindex(range(24))


def stored_energy_per_day_mwh(config: BatteryConfig) -> float:
    """Energy moved into (and back out of) storage by one charge block + one discharge block,
    feasible in either order and within the daily cycle limit."""
    headroom_up = config.soc_max_mwh - config.soc_initial_mwh  # charge-first
    headroom_down = config.soc_initial_mwh - config.soc_min_mwh  # discharge-first
    cycle_cap = config.max_cycles_per_day * config.energy_mwh / config.eta_discharge
    return max(0.0, min(headroom_up, headroom_down, cycle_cap))


def choose_blocks(profile: pd.Series, config: BatteryConfig = BatteryConfig()) -> dict | None:
    """Best (charge block, discharge block) on an hour-of-day profile, or None if idle.

    Returns dict with charge_hours, discharge_hours (lists of local hours), charge_mw,
    discharge_mw, and expected_net_profit on the profile."""
    stored = stored_energy_per_day_mwh(config)
    if stored <= 0 or profile.isna().any():
        return None
    dt = config.dt_hours
    grid_in_mwh = stored / config.eta_charge
    delivered_mwh = stored * config.eta_discharge
    k_c = math.ceil(grid_in_mwh / (config.power_mw * dt) - 1e-9)
    k_d = math.ceil(delivered_mwh / (config.power_mw * dt) - 1e-9)

    values = profile.to_numpy(dtype=float)
    n = len(values)
    best = None
    for c0 in range(n - k_c + 1):
        c_mean = values[c0:c0 + k_c].mean()
        for d0 in range(n - k_d + 1):
            if c0 < d0 + k_d and d0 < c0 + k_c:  # overlap
                continue
            d_mean = values[d0:d0 + k_d].mean()
            expected = (delivered_mwh * d_mean - grid_in_mwh * c_mean
                        - config.degradation_cost_per_mwh * delivered_mwh)
            if best is None or expected > best[0]:
                best = (expected, c0, d0)

    if best is None or best[0] <= 0:
        return None
    expected, c0, d0 = best
    return {
        "charge_hours": list(range(c0, c0 + k_c)),
        "discharge_hours": list(range(d0, d0 + k_d)),
        "charge_mw": grid_in_mwh / (k_c * dt),
        "discharge_mw": delivered_mwh / (k_d * dt),
        "expected_net_profit": expected,
    }


def dispatch_day(day_prices: pd.Series, blocks: dict | None, config: BatteryConfig = BatteryConfig(),
                 local_tz: str = LOCAL_TZ) -> DispatchResult:
    """Apply a fixed block schedule to one Pacific day's actual prices."""
    dt = config.dt_hours
    local_hours = day_prices.index.tz_convert(local_tz).hour
    charge = np.zeros(len(day_prices))
    discharge = np.zeros(len(day_prices))

    if blocks is not None:
        # First occurrence of each local hour only, so the repeated hour on a fall-back
        # DST day doesn't double up.
        first = ~pd.Index(local_hours).duplicated()
        charge[first & np.isin(local_hours, blocks["charge_hours"])] = blocks["charge_mw"]
        discharge[first & np.isin(local_hours, blocks["discharge_hours"])] = blocks["discharge_mw"]

        # A block that hits the missing hour on a spring-forward day comes up short;
        # scale the other side down so storage still ends where it started.
        stored_in = config.eta_charge * charge.sum() * dt
        stored_out = discharge.sum() * dt / config.eta_discharge
        if stored_in > stored_out and stored_in > 0:
            charge *= stored_out / stored_in
        elif stored_out > stored_in and stored_out > 0:
            discharge *= stored_in / stored_out

    soc = config.soc_initial_mwh + np.concatenate(
        [[0.0], np.cumsum(config.eta_charge * charge * dt - discharge * dt / config.eta_discharge)])
    tol = 1e-6
    if soc.min() < config.soc_min_mwh - tol or soc.max() > config.soc_max_mwh + tol:
        raise RuntimeError(f"rule-based schedule violates SoC bounds ({soc.min():.3f}, {soc.max():.3f})")
    if abs(soc[-1] - config.soc_final_mwh) > tol:
        raise RuntimeError(f"rule-based schedule ends at SoC {soc[-1]:.3f}, expected {config.soc_final_mwh}")

    price_values = day_prices.to_numpy(dtype=float)
    gross_revenue = float(((discharge - charge) * price_values * dt).sum())
    degradation_cost = config.degradation_cost_per_mwh * float(discharge.sum()) * dt
    soc_index = day_prices.index.append(pd.DatetimeIndex([day_prices.index[-1] + pd.Timedelta(hours=dt)]))

    return DispatchResult(
        charge_mw=pd.Series(charge, index=day_prices.index, name="charge_mw"),
        discharge_mw=pd.Series(discharge, index=day_prices.index, name="discharge_mw"),
        soc_mwh=pd.Series(soc, index=soc_index, name="soc_mwh"),
        price=day_prices.copy(),
        gross_revenue=gross_revenue,
        degradation_cost=degradation_cost,
        net_profit=gross_revenue - degradation_cost,
    )


def rule_based_multiday(prices: pd.Series, config: BatteryConfig = BatteryConfig(),
                        lookback_days: int = LOOKBACK_DAYS, min_history_days: int = MIN_HISTORY_DAYS,
                        local_tz: str = LOCAL_TZ) -> DispatchResult:
    """Run the rule over every Pacific day in an hourly UTC price series. Each day's blocks
    come only from the trailing `lookback_days` before it; days lacking `min_history_days`
    of history are skipped. `daily` matches solve_multiday's columns plus the chosen blocks."""
    prices = prices.sort_index()
    dates = pd.Index(prices.index.tz_convert(local_tz).date)
    unique_dates = pd.unique(dates)

    charge_parts, discharge_parts, soc_parts, daily_rows = [], [], [], []
    for date in unique_dates:
        window_start = date - datetime.timedelta(days=lookback_days)
        in_window = (dates >= window_start) & (dates < date)
        if pd.unique(dates[in_window]).size < min_history_days:
            continue
        blocks = choose_blocks(hour_of_day_profile(prices[in_window], local_tz), config)
        day_prices = prices[dates == date]
        result = dispatch_day(day_prices, blocks, config, local_tz)

        charge_parts.append(result.charge_mw)
        discharge_parts.append(result.discharge_mw)
        soc_parts.append(result.soc_mwh if not soc_parts else result.soc_mwh.iloc[1:])
        daily_rows.append({
            "pacific_date": date,
            "gross_revenue": result.gross_revenue,
            "degradation_cost": result.degradation_cost,
            "net_profit": result.net_profit,
            "charged_mwh": float(result.charge_mw.sum()) * config.dt_hours,
            "discharged_mwh": float(result.discharge_mw.sum()) * config.dt_hours,
            "n_hours": len(day_prices),
            "charge_start_hour": blocks["charge_hours"][0] if blocks else None,
            "discharge_start_hour": blocks["discharge_hours"][0] if blocks else None,
        })

    if not daily_rows:
        raise ValueError("no day has enough price history for the rule-based strategy")
    daily = pd.DataFrame(daily_rows).set_index("pacific_date")

    return DispatchResult(
        charge_mw=pd.concat(charge_parts),
        discharge_mw=pd.concat(discharge_parts),
        soc_mwh=pd.concat(soc_parts),
        price=prices.copy(),
        gross_revenue=float(daily["gross_revenue"].sum()),
        degradation_cost=float(daily["degradation_cost"].sum()),
        net_profit=float(daily["net_profit"].sum()),
        daily=daily,
    )


if __name__ == "__main__":
    months = list(pd.period_range("2024-06", "2024-07", freq="M"))
    series = to_price_series(filter_hub_market(read_months(months), hub="SP15"))
    result = rule_based_multiday(series)
    july = result.daily[pd.to_datetime(result.daily.index) >= "2024-07-01"]
    print(july.round(2).to_string())
    print(f"\nJuly 2024 SP15 rule-based net profit: ${july['net_profit'].sum():,.0f}")
