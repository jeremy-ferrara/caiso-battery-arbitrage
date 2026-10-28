import numpy as np
import pandas as pd
import pytest

from battery import BatteryConfig, solve_day
from forecast import MIN_TRAIN_DAYS, day_hour_grid, forecast_driven_multiday, settle, walk_forward_forecast
from load_prices import LOCAL_TZ

DEFAULT_CONFIG = BatteryConfig()


def synthetic_prices(start="2024-01-01", days=60, seed=0) -> pd.Series:
    """Hourly UTC series of duck-shaped days with noise, on Pacific-local days."""
    index = pd.date_range(start=start, end=pd.Timestamp(start) + pd.Timedelta(days=days),
                          freq="h", tz=LOCAL_TZ, inclusive="left").tz_convert("UTC")
    hour = index.tz_convert(LOCAL_TZ).hour.to_numpy()
    rng = np.random.default_rng(seed)
    base = 40 - 30 * ((hour >= 9) & (hour <= 15)) + 80 * ((hour >= 17) & (hour <= 20))
    return pd.Series(base + rng.normal(0, 5, len(index)), index=index, name="LMP")


def test_no_lookahead_scrambling_future_leaves_forecast_unchanged():
    prices = synthetic_prices()
    full = walk_forward_forecast(prices)
    target = pd.Timestamp("2024-02-15").date()
    local_dates = prices.index.tz_convert(LOCAL_TZ).date

    scrambled = prices.copy()
    future = local_dates >= target
    scrambled[future] = np.random.default_rng(99).uniform(-500, 2000, future.sum())
    altered = walk_forward_forecast(scrambled)

    on_target = full.index.tz_convert(LOCAL_TZ).date == target
    assert on_target.sum() == 24
    np.testing.assert_allclose(altered.loc[full.index[on_target], "forecast"],
                               full.loc[on_target, "forecast"], rtol=0, atol=1e-9)
    # Sanity: the scrambling does change actuals for the target day (so the test isn't vacuous).
    assert not np.allclose(altered.loc[full.index[on_target], "actual"], full.loc[on_target, "actual"])


def test_first_forecast_day_respects_lags_and_min_training():
    prices = synthetic_prices()
    fc = walk_forward_forecast(prices)
    first = fc.index.tz_convert(LOCAL_TZ).date.min()
    # 7 days of lags, then MIN_TRAIN_DAYS complete days to train on.
    assert first == (pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 + MIN_TRAIN_DAYS)).date()


def test_persistence_is_previous_day_same_hour():
    prices = synthetic_prices()
    fc = walk_forward_forecast(prices)
    ts = fc.index[100]
    assert fc.loc[ts, "persistence"] == pytest.approx(prices[ts - pd.Timedelta(days=1)])


def test_dst_days_get_one_forecast_per_actual_hour():
    prices = synthetic_prices(start="2024-01-25", days=60)  # covers spring-forward, 2024-03-10
    fc = walk_forward_forecast(prices)
    counts = pd.Series(fc.index.tz_convert(LOCAL_TZ).date).value_counts()
    assert counts[pd.Timestamp("2024-03-10").date()] == 23
    assert fc["forecast"].notna().all()
    grid = day_hour_grid(prices)
    assert grid.loc["2024-03-10"].notna().all()


def test_settle_scores_plan_at_actual_prices():
    index = pd.date_range("2024-01-01", periods=24, freq="h", tz=LOCAL_TZ).tz_convert("UTC")
    forecast = pd.Series([10.0] * 12 + [100.0] * 12, index=index)
    actual = pd.Series([100.0] * 12 + [10.0] * 12, index=index)  # shape is exactly inverted
    plan = solve_day(forecast, DEFAULT_CONFIG)
    assert plan.net_profit > 0
    result = settle(plan, actual, DEFAULT_CONFIG)
    assert result.net_profit < 0
    expected = ((plan.discharge_mw - plan.charge_mw) * actual).sum() - 15.0 * plan.discharge_mw.sum()
    assert result.net_profit == pytest.approx(expected)


def test_perfect_forecast_matches_perfect_foresight():
    prices = synthetic_prices()
    fc = walk_forward_forecast(prices)
    result = forecast_driven_multiday(prices, DEFAULT_CONFIG, forecast=fc, column="actual")
    pf = sum(solve_day(day, DEFAULT_CONFIG).net_profit
             for _, day in fc["actual"].groupby(fc.index.tz_convert(LOCAL_TZ).date))
    assert result.net_profit == pytest.approx(pf, rel=1e-6)
