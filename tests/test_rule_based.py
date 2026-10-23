import numpy as np
import pandas as pd
import pytest

from battery import BatteryConfig, solve_day, filter_hub_market, to_price_series
from load_prices import month_path, LOCAL_TZ
from rule_based import (choose_blocks, dispatch_day, rule_based_multiday,
                        stored_energy_per_day_mwh)

TOL_MW = 1e-6
TOL_MWH = 1e-6
TOL_USD = 1e-6

DEFAULT_CONFIG = BatteryConfig()


def pacific_days(daily_values, start="2024-01-01") -> pd.Series:
    """Hourly UTC series built from Pacific-local-time days; each element of daily_values is a
    24-long list for one day. DST days are not handled here -- use dates outside DST shifts."""
    index = pd.date_range(start=start, periods=24 * len(daily_values), freq="h", tz=LOCAL_TZ).tz_convert("UTC")
    return pd.Series(np.concatenate(daily_values).astype(float), index=index, name="LMP")


# Cheap 10-13 local, expensive 18-20 local, flat 40 otherwise.
DUCK = [40.0] * 24
for h in range(10, 14):
    DUCK[h] = 5.0
for h in range(18, 21):
    DUCK[h] = 150.0


def check_physical(result, config=DEFAULT_CONFIG):
    assert result.charge_mw.max() <= config.power_mw + TOL_MW
    assert result.discharge_mw.max() <= config.power_mw + TOL_MW
    assert result.charge_mw.min() >= -TOL_MW
    assert result.discharge_mw.min() >= -TOL_MW
    assert result.soc_mwh.min() >= config.soc_min_mwh - TOL_MWH
    assert result.soc_mwh.max() <= config.soc_max_mwh + TOL_MWH
    assert result.soc_mwh.iloc[0] == pytest.approx(config.soc_initial_mwh, abs=TOL_MWH)
    assert result.soc_mwh.iloc[-1] == pytest.approx(config.soc_final_mwh, abs=TOL_MWH)
    assert result.discharge_mw.sum() * config.dt_hours <= config.max_cycles_per_day * config.energy_mwh + TOL_MWH


def test_stored_energy_limited_by_pinned_soc():
    # 50% start/end -> 200 MWh either way, below the 1-cycle cap.
    assert stored_energy_per_day_mwh(DEFAULT_CONFIG) == pytest.approx(200.0)
    assert stored_energy_per_day_mwh(BatteryConfig(soc_initial_fraction=0.25, soc_final_fraction=0.25)) \
        == pytest.approx(100.0)


def test_choose_blocks_picks_cheap_and_expensive_hours():
    blocks = choose_blocks(pd.Series(DUCK), DEFAULT_CONFIG)
    assert blocks is not None
    assert set(blocks["charge_hours"]) <= set(range(10, 14))
    assert set(blocks["discharge_hours"]) <= set(range(18, 21))
    assert blocks["charge_mw"] <= DEFAULT_CONFIG.power_mw
    assert blocks["discharge_mw"] <= DEFAULT_CONFIG.power_mw


def test_choose_blocks_idle_when_spread_below_degradation():
    flat = pd.Series([50.0] * 24)
    assert choose_blocks(flat, DEFAULT_CONFIG) is None
    # Spread exists but doesn't cover losses + $15/MWh degradation.
    small = pd.Series([50.0] * 12 + [60.0] * 12)
    assert choose_blocks(small, DEFAULT_CONFIG) is None


def test_dispatch_day_respects_physical_limits_and_balances():
    day = pacific_days([DUCK])
    blocks = choose_blocks(pd.Series(DUCK), DEFAULT_CONFIG)
    result = dispatch_day(day, blocks, DEFAULT_CONFIG)
    check_physical(result)

    dt = DEFAULT_CONFIG.dt_hours
    assert np.minimum(result.charge_mw, result.discharge_mw).max() < TOL_MW
    expected_gross = (day * (result.discharge_mw - result.charge_mw) * dt).sum()
    assert result.gross_revenue == pytest.approx(expected_gross, abs=TOL_USD)
    assert result.degradation_cost == pytest.approx(
        DEFAULT_CONFIG.degradation_cost_per_mwh * result.discharge_mw.sum() * dt, abs=TOL_USD)
    assert result.net_profit == pytest.approx(result.gross_revenue - result.degradation_cost, abs=TOL_USD)


def test_schedule_ignores_todays_prices():
    history = [DUCK] * 10
    quiet_today = pacific_days(history + [[40.0] * 24])
    wild_today = pacific_days(history + [[500.0] * 12 + [-100.0] * 12])

    a = rule_based_multiday(quiet_today, DEFAULT_CONFIG)
    b = rule_based_multiday(wild_today, DEFAULT_CONFIG)
    last = a.daily.index[-1]
    assert a.daily.loc[last, "charge_start_hour"] == b.daily.loc[last, "charge_start_hour"]
    assert a.daily.loc[last, "discharge_start_hour"] == b.daily.loc[last, "discharge_start_hour"]
    np.testing.assert_allclose(a.charge_mw.iloc[-24:].to_numpy(), b.charge_mw.iloc[-24:].to_numpy())
    np.testing.assert_allclose(a.discharge_mw.iloc[-24:].to_numpy(), b.discharge_mw.iloc[-24:].to_numpy())
    # Same schedule, different settlement.
    assert a.daily.loc[last, "net_profit"] != pytest.approx(b.daily.loc[last, "net_profit"])


def test_skips_days_without_enough_history():
    prices = pacific_days([DUCK] * 10)
    result = rule_based_multiday(prices, DEFAULT_CONFIG, lookback_days=30, min_history_days=7)
    assert len(result.daily) == 3  # days 8, 9, 10
    first_local_date = prices.index[0].tz_convert(LOCAL_TZ).date()
    assert (pd.to_datetime(result.daily.index).min() - pd.Timestamp(first_local_date)).days == 7


def test_lookback_window_excludes_old_history():
    # 10 days where evening is cheap and midday expensive, then 5 days of the normal duck.
    # With a 5-day lookback, the 6th duck day should follow the duck, not the old pattern.
    inverted = [DUCK[(h + 8) % 24] for h in range(24)]
    prices = pacific_days([inverted] * 10 + [DUCK] * 6)
    result = rule_based_multiday(prices, DEFAULT_CONFIG, lookback_days=5, min_history_days=5)
    last = result.daily.iloc[-1]
    assert last["charge_start_hour"] in range(10, 14)
    assert last["discharge_start_hour"] in range(18, 21)


@pytest.mark.parametrize("start, n_hours", [("2024-03-10", 23), ("2024-11-03", 25)])
def test_dst_days_stay_balanced(start, n_hours):
    # Charge and discharge blocks spanning the DST shift hour (1-3 am local).
    profile = pd.Series([0.0] * 4 + [200.0] * 4 + [50.0] * 16)
    index = pd.date_range(start=start, end=pd.Timestamp(start) + pd.Timedelta(days=1), freq="h",
                          tz=LOCAL_TZ, inclusive="left").tz_convert("UTC")
    assert len(index) == n_hours
    local_hours = index.tz_convert(LOCAL_TZ).hour
    day = pd.Series(profile.to_numpy()[local_hours], index=index)
    result = dispatch_day(day, choose_blocks(profile, DEFAULT_CONFIG), DEFAULT_CONFIG)
    check_physical(result)
    assert result.discharge_mw.sum() > 0


REAL_DAYS = [("2024-07", "2024-07-11"), ("2024-04", "2024-04-07")]


@pytest.mark.parametrize("month, date", REAL_DAYS)
def test_real_day_never_beats_perfect_foresight(month, date):
    prev = str(pd.Period(month) - 1)
    if not (month_path(pd.Period(month)).exists() and month_path(pd.Period(prev)).exists()):
        pytest.skip("price data not downloaded")
    df = pd.concat([pd.read_parquet(month_path(pd.Period(m))) for m in (prev, month)])
    series = to_price_series(filter_hub_market(df, hub="SP15"))
    target = pd.Timestamp(date).date()
    dates = series.index.tz_convert(LOCAL_TZ).date
    series = series[dates <= target]

    rule = rule_based_multiday(series, DEFAULT_CONFIG)
    pf = solve_day(series[dates[dates <= target] == target], DEFAULT_CONFIG)
    assert rule.daily.loc[target, "net_profit"] <= pf.net_profit + 1e-3
    assert rule.daily.loc[target, "discharged_mwh"] > 0
