"""Perfect-foresight battery dispatch: a linear program over known hourly prices.

Each calendar day is solved independently (state of charge is pinned to the same
fraction at the start and end of every day, so there's no link across day
boundaries -- splitting a multi-day series by day and solving each separately is
exactly equivalent to solving one big LP with daily reset constraints).

Decision variables per hour t: charge_mw[t] >= 0, discharge_mw[t] >= 0 (kept
separate rather than one signed variable, since each direction has its own
efficiency loss). soc_mwh has one more point than the hourly series: soc_mwh[t]
is the charge level at the start of hour t, and the trailing point is the level
at the end of the last hour.

Objective (maximize): revenue from discharging, minus the cost of charging,
minus a per-MWh degradation charge on throughput discharged.

No explicit constraint forbids charging and discharging in the same hour --
with round-trip efficiency < 100% and a positive degradation cost, doing both
at once is always strictly dominated by doing less of both (the price terms
cancel but the loss and degradation cost don't), so the LP relaxation never
chooses it except for solver-level numerical noise.

Usage:
    python src/battery.py
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import cvxpy as cp

from load_prices import LOCAL_TZ, month_path, read_months


@dataclass(frozen=True)
class BatteryConfig:
    power_mw: float = 100.0
    energy_mwh: float = 400.0
    round_trip_efficiency: float = 0.85
    degradation_cost_per_mwh: float = 15.0
    soc_min_mwh: float = 0.0
    soc_max_mwh: float | None = None  # None -> energy_mwh
    soc_initial_fraction: float = 0.5
    soc_final_fraction: float = 0.5
    max_cycles_per_day: float = 1.0
    dt_hours: float = 1.0

    def __post_init__(self):
        if self.soc_max_mwh is None:
            object.__setattr__(self, "soc_max_mwh", self.energy_mwh)

    @property
    def eta_charge(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def eta_discharge(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def soc_initial_mwh(self) -> float:
        return self.soc_initial_fraction * self.energy_mwh

    @property
    def soc_final_mwh(self) -> float:
        return self.soc_final_fraction * self.energy_mwh


@dataclass
class DispatchResult:
    charge_mw: pd.Series
    discharge_mw: pd.Series
    soc_mwh: pd.Series
    price: pd.Series
    gross_revenue: float
    degradation_cost: float
    net_profit: float
    daily: pd.DataFrame | None = field(default=None)


def _build_and_solve(price_values: np.ndarray, config: BatteryConfig,
                      solver: str | None = None, solver_kwargs: dict | None = None):
    """Pure numpy/cvxpy core. Returns (charge_mw, discharge_mw, soc_mwh, status) arrays,
    soc_mwh has length T+1."""
    t = len(price_values)
    dt = config.dt_hours
    eta_c, eta_d = config.eta_charge, config.eta_discharge

    charge = cp.Variable(t, nonneg=True)
    discharge = cp.Variable(t, nonneg=True)
    soc = cp.Variable(t + 1)

    constraints = [
        charge <= config.power_mw,
        discharge <= config.power_mw,
        soc[1:] == soc[:-1] + eta_c * dt * charge - (dt / eta_d) * discharge,
        soc >= config.soc_min_mwh,
        soc <= config.soc_max_mwh,
        soc[0] == config.soc_initial_mwh,
        soc[-1] == config.soc_final_mwh,
        cp.sum(discharge) * dt <= config.max_cycles_per_day * config.energy_mwh,
    ]

    gross_revenue = cp.sum(cp.multiply(price_values, discharge - charge)) * dt
    degradation_cost = config.degradation_cost_per_mwh * cp.sum(discharge) * dt
    objective = cp.Maximize(gross_revenue - degradation_cost)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, **(solver_kwargs or {}))

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver did not find an optimal solution (status={problem.status})")

    return charge.value, discharge.value, soc.value, problem.status


def solve_day(prices: pd.Series, config: BatteryConfig = BatteryConfig(),
              solver: str | None = None, solver_kwargs: dict | None = None) -> DispatchResult:
    """Solve the perfect-foresight dispatch LP for one contiguous block of hourly prices
    (nominally one calendar day, but works for any length -- used directly by tests with
    short synthetic series)."""
    if prices.empty:
        raise ValueError("prices must not be empty")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be sorted")

    charge, discharge, soc, _ = _build_and_solve(prices.to_numpy(dtype=float), config, solver, solver_kwargs)

    dt = config.dt_hours
    charge_s = pd.Series(charge, index=prices.index, name="charge_mw")
    discharge_s = pd.Series(discharge, index=prices.index, name="discharge_mw")
    soc_index = prices.index.append(pd.DatetimeIndex([prices.index[-1] + pd.Timedelta(hours=dt)]))
    soc_s = pd.Series(soc, index=soc_index, name="soc_mwh")

    gross_revenue = float(((discharge - charge) * prices.to_numpy(dtype=float) * dt).sum())
    degradation_cost = config.degradation_cost_per_mwh * float(discharge.sum()) * dt
    net_profit = gross_revenue - degradation_cost

    return DispatchResult(
        charge_mw=charge_s,
        discharge_mw=discharge_s,
        soc_mwh=soc_s,
        price=prices.copy(),
        gross_revenue=gross_revenue,
        degradation_cost=degradation_cost,
        net_profit=net_profit,
    )


def _pacific_calendar_date(index: pd.DatetimeIndex, local_tz: str = LOCAL_TZ) -> np.ndarray:
    return index.tz_convert(local_tz).date


def solve_multiday(prices: pd.Series, config: BatteryConfig = BatteryConfig(),
                    local_tz: str = LOCAL_TZ, solver: str | None = None,
                    solver_kwargs: dict | None = None) -> DispatchResult:
    """Group an hourly UTC price series into Pacific calendar days, solve each day's LP
    independently (state of charge resets at each day boundary), and concatenate results."""
    prices = prices.sort_index()
    dates = _pacific_calendar_date(prices.index, local_tz)

    charge_parts, discharge_parts, soc_parts = [], [], []
    daily_rows = []
    for date in pd.unique(dates):
        day_prices = prices[dates == date]
        result = solve_day(day_prices, config, solver, solver_kwargs)
        charge_parts.append(result.charge_mw)
        discharge_parts.append(result.discharge_mw)
        soc_parts.append(result.soc_mwh if not soc_parts else result.soc_mwh.iloc[1:])
        daily_rows.append({
            "pacific_date": date,
            "gross_revenue": result.gross_revenue,
            "degradation_cost": result.degradation_cost,
            "net_profit": result.net_profit,
            "n_hours": len(day_prices),
        })

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


def filter_hub_market(df: pd.DataFrame, hub: str = "SP15", market: str = "DAY_AHEAD_HOURLY") -> pd.DataFrame:
    out = df[(df["Market"] == market) & (df["Location"] == f"TH_{hub}_GEN-APND")]
    return out.sort_values("Interval Start")


def to_price_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df["LMP"].to_numpy(), index=pd.DatetimeIndex(df["Interval Start"]), name="LMP")


if __name__ == "__main__":
    demo_days = [
        ("2024-07", "2024-07-11", "summer"),
        ("2024-04", "2024-04-07", "spring"),
    ]
    config = BatteryConfig()

    for month, date_str, label in demo_days:
        df = read_months([pd.Period(month)])
        series = to_price_series(filter_hub_market(df, hub="SP15"))
        target_date = pd.Timestamp(date_str).date()
        day_series = series[series.index.tz_convert(LOCAL_TZ).date == target_date]

        result = solve_day(day_series, config)

        table = pd.DataFrame({
            "price_$/MWh": result.price.values,
            "charge_mw": result.charge_mw.values,
            "discharge_mw": result.discharge_mw.values,
            "soc_mwh": result.soc_mwh.iloc[:-1].values,
        }, index=result.price.index.tz_convert(LOCAL_TZ))

        print(f"\n=== {label.upper()}: {date_str} (SP15, day-ahead) ===")
        print(table.round(2).to_string())
        print(f"\nGross revenue:     ${result.gross_revenue:,.2f}")
        print(f"Degradation cost:  ${result.degradation_cost:,.2f}")
        print(f"Net profit:        ${result.net_profit:,.2f}")
