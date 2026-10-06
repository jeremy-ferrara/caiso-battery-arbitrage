import numpy as np
import pandas as pd
import pytest

from battery import BatteryConfig, DispatchResult, solve_day, solve_multiday, filter_hub_market, to_price_series
from load_prices import month_path, LOCAL_TZ

TOL_MW = 1e-4
TOL_MWH = 1e-3
TOL_USD = 1e-3

DEFAULT_CONFIG = BatteryConfig()


def make_hourly_series(values, start="2024-01-01", tz="UTC") -> pd.Series:
    index = pd.date_range(start=start, periods=len(values), freq="h", tz=tz)
    return pd.Series(values, index=index, name="LMP")


def real_day_prices(month: str, date: str, hub: str = "SP15") -> pd.Series:
    path = month_path(pd.Period(month))
    if not path.exists():
        pytest.skip(f"no data file for {month}")
    df = pd.read_parquet(path)
    series = to_price_series(filter_hub_market(df, hub=hub))
    target_date = pd.Timestamp(date).date()
    return series[series.index.tz_convert(LOCAL_TZ).date == target_date]


ALTERNATING_DAY = make_hourly_series([10, 80] * 12)
SUMMER_DAY = real_day_prices("2024-07", "2024-07-11")
SPRING_DAY = real_day_prices("2024-04", "2024-04-07")

DAY_SCENARIOS = [
    pytest.param(ALTERNATING_DAY, id="alternating-synthetic"),
    pytest.param(SUMMER_DAY, id="summer-2024-07-11"),
    pytest.param(SPRING_DAY, id="spring-2024-04-07"),
]


@pytest.mark.parametrize("prices", DAY_SCENARIOS)
def test_never_charges_and_discharges_same_hour(prices):
    result = solve_day(prices, DEFAULT_CONFIG)
    overlap = np.minimum(result.charge_mw.to_numpy(), result.discharge_mw.to_numpy())
    assert overlap.max() < TOL_MW


@pytest.mark.parametrize("prices", DAY_SCENARIOS)
def test_soc_within_bounds(prices):
    result = solve_day(prices, DEFAULT_CONFIG)
    assert result.soc_mwh.min() >= DEFAULT_CONFIG.soc_min_mwh - TOL_MWH
    assert result.soc_mwh.max() <= DEFAULT_CONFIG.soc_max_mwh + TOL_MWH


@pytest.mark.parametrize("prices", DAY_SCENARIOS)
def test_power_limit_respected(prices):
    result = solve_day(prices, DEFAULT_CONFIG)
    assert result.charge_mw.max() <= DEFAULT_CONFIG.power_mw + TOL_MW
    assert result.discharge_mw.max() <= DEFAULT_CONFIG.power_mw + TOL_MW
    assert result.charge_mw.min() >= -TOL_MW
    assert result.discharge_mw.min() >= -TOL_MW


def test_flat_prices_no_action():
    prices = make_hourly_series([50.0] * 24)
    result = solve_day(prices, DEFAULT_CONFIG)
    assert result.charge_mw.abs().max() < TOL_MW
    assert result.discharge_mw.abs().max() < TOL_MW
    assert result.soc_mwh.std() < TOL_MWH
    assert result.net_profit == pytest.approx(0, abs=TOL_USD)


def test_two_price_day_charges_cheap_discharges_expensive():
    prices = make_hourly_series([20.0] * 12 + [100.0] * 12)
    result = solve_day(prices, DEFAULT_CONFIG)

    assert result.discharge_mw.iloc[:12].max() < TOL_MW
    assert result.charge_mw.iloc[12:].max() < TOL_MW
    assert result.charge_mw.sum() > 50
    assert result.discharge_mw.sum() > 50
    assert result.net_profit > 0
    assert result.soc_mwh.iloc[-1] == pytest.approx(200.0, abs=TOL_MWH)


def test_energy_accounting_balances():
    prices = make_hourly_series([20.0] * 12 + [100.0] * 12)
    result = solve_day(prices, DEFAULT_CONFIG)

    dt = DEFAULT_CONFIG.dt_hours
    eta_c, eta_d = DEFAULT_CONFIG.eta_charge, DEFAULT_CONFIG.eta_discharge

    energy_added = eta_c * result.charge_mw.sum() * dt
    energy_removed = result.discharge_mw.sum() * dt / eta_d
    soc_delta = result.soc_mwh.iloc[-1] - result.soc_mwh.iloc[0]
    assert soc_delta == pytest.approx(energy_added - energy_removed, abs=TOL_MWH)

    assert result.soc_mwh.iloc[0] == pytest.approx(DEFAULT_CONFIG.soc_initial_mwh, abs=1e-6)
    assert result.soc_mwh.iloc[-1] == pytest.approx(DEFAULT_CONFIG.soc_final_mwh, abs=1e-6)

    expected_gross = (prices * (result.discharge_mw - result.charge_mw) * dt).sum()
    assert expected_gross == pytest.approx(result.gross_revenue, abs=TOL_USD)
    assert result.gross_revenue - result.degradation_cost == pytest.approx(result.net_profit, abs=1e-6)


def test_cycle_limit_respected():
    # Multiple large spikes -- more arbitrage opportunity than one cycle/day allows.
    prices = make_hourly_series([10, 10, 10, 300, 300, 10, 10, 300, 300, 10, 10, 300,
                                  300, 10, 10, 300, 300, 10, 10, 10, 10, 10, 10, 10])
    result = solve_day(prices, DEFAULT_CONFIG)
    dt = DEFAULT_CONFIG.dt_hours
    throughput = result.discharge_mw.sum() * dt
    assert throughput <= DEFAULT_CONFIG.max_cycles_per_day * DEFAULT_CONFIG.energy_mwh + TOL_MWH


def test_multiday_splits_into_independent_pacific_days():
    # Start at Pacific midnight (UTC-8 in January) so the 48h window covers exactly two
    # Pacific calendar days.
    prices = make_hourly_series([20.0] * 12 + [100.0] * 12 + [20.0] * 12 + [100.0] * 12,
                                 start="2024-01-01 08:00")
    result = solve_multiday(prices, DEFAULT_CONFIG)

    assert len(result.daily) == 2
    boundary_soc = result.soc_mwh.loc[result.soc_mwh.index == prices.index[24]]
    assert boundary_soc.iloc[0] == pytest.approx(200.0, abs=TOL_MWH)

    day1 = solve_day(prices.iloc[:24], DEFAULT_CONFIG)
    day2 = solve_day(prices.iloc[24:], DEFAULT_CONFIG)
    assert result.net_profit == pytest.approx(day1.net_profit + day2.net_profit, abs=TOL_USD)
