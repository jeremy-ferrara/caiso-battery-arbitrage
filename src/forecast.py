"""Walk-forward day-ahead price forecast and the forecast-driven dispatch strategy.

Timing (what is knowable when). A battery bids into CAISO's day-ahead (DA) market for
operating day D on the morning of D-1 (bids close ~10:00 PT). DA prices for D-1 were
published ~13:00 PT on D-2, so at bid time every DA price through the end of D-1 is
public and none for D are. The forecast for D therefore uses only DA prices from D-1
and earlier, plus calendar fields known in advance. Real-time prices are not used: at
bid time they are only known for part of D-1, and leaving them out keeps the cutoff simple.

Features for day D, local hour h (all from days strictly before D):
    lag1         DA price at hour h on D-1
    lag7         DA price at hour h on D-7 (same weekday last week)
    mean7        mean DA price at hour h over D-1 .. D-7
    prev_mean    mean DA price over all of D-1
    prev_spread  max - min DA price over D-1
    dow_*        day-of-week dummies for D (Monday = baseline)
    doy_sin/cos  day of year of D on a circle (season). Used instead of month dummies:
                 an expanding window that starts in July has no training rows for most
                 months during its first year, so per-month dummies would be unfit.

Model. One ridge regression per local hour (24 models), refit from scratch for every
target day on all earlier days (expanding window), or with `window_days` set, only on the
earlier days within that many calendar days of the target (rolling window). Features are standardized with the
training rows' mean and std only. The penalty and the feature list are fixed up front,
not tuned on backtest results, since tuning on the test period would also leak.

Leakage guard. Feature rows are built with day-level shifts of at least 1, and the
training set for day D is rows with date < D. tests/test_forecast.py checks this
directly: scrambling every price on and after D leaves D's forecast unchanged.

DST. Prices are put on a Pacific day x local hour (0-23) grid. The repeated hour on a
fall-back day is averaged; the missing hour on a spring-forward day is filled by linear
interpolation from the same day's neighbours (that day's own values only, so this only
affects later days' features). Forecasts map back to the actual UTC hours of D by local
hour; the repeated hour gets the same forecast twice.

Dispatch. The day's LP (battery.solve_day, same BatteryConfig) is solved on forecast
prices; the resulting schedule is settled at actual DA prices.

Usage:
    python src/forecast.py
"""
import numpy as np
import pandas as pd

from battery import BatteryConfig, DispatchResult, filter_hub_market, solve_day, to_price_series
from load_prices import LOCAL_TZ, read_months

RIDGE_ALPHA = 10.0
MIN_TRAIN_DAYS = 30
LAG_DAYS = 7
DOW_DUMMIES = [f"dow_{i}" for i in range(1, 7)]
FEATURES = ["lag1", "lag7", "mean7", "prev_mean", "prev_spread", *DOW_DUMMIES, "doy_sin", "doy_cos"]


def day_hour_grid(prices: pd.Series, local_tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Pacific date x local hour (0-23) price grid over every calendar day in range.
    Missing whole days stay NaN; a single missing hour (DST) is interpolated within the day."""
    local = prices.index.tz_convert(local_tz)
    grid = prices.groupby([pd.to_datetime(local.date), local.hour]).mean().unstack()
    grid = grid.reindex(columns=range(24))
    grid = grid.reindex(pd.date_range(grid.index.min(), grid.index.max(), freq="D"))
    has_data = grid.notna().any(axis=1)
    grid.loc[has_data] = grid.loc[has_data].interpolate(axis=1, limit_direction="both")
    return grid


def build_features(grid: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Per local hour: a frame indexed by target date with FEATURES and the target `y`.
    Every feature is a shift of >= 1 day, so row D uses nothing from D or later."""
    prev_day = grid.shift(1)
    prev_mean = prev_day.mean(axis=1, skipna=False)
    prev_spread = prev_day.max(axis=1, skipna=False) - prev_day.min(axis=1, skipna=False)
    lagged = [grid.shift(k) for k in range(1, LAG_DAYS + 1)]
    dow = pd.get_dummies(grid.index.dayofweek, prefix="dow", dtype=float).set_index(grid.index)
    dow = dow.reindex(columns=[f"dow_{i}" for i in range(7)], fill_value=0.0)[DOW_DUMMIES]
    angle = 2 * np.pi * (grid.index.dayofyear - 1) / 365.25

    out = {}
    for h in range(24):
        frame = pd.DataFrame({
            "lag1": lagged[0][h],
            "lag7": lagged[LAG_DAYS - 1][h],
            "mean7": pd.concat([lag[h] for lag in lagged], axis=1).mean(axis=1, skipna=False),
            "prev_mean": prev_mean,
            "prev_spread": prev_spread,
        }, index=grid.index)
        frame[DOW_DUMMIES] = dow
        frame["doy_sin"] = np.sin(angle)
        frame["doy_cos"] = np.cos(angle)
        frame["y"] = grid[h]
        out[h] = frame
    return out


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA):
    """Ridge on standardized features with an unpenalized intercept. Returns a predict function."""
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0  # e.g. a dow dummy with no variation yet
    z = (x - mu) / sd
    y_mean = y.mean()
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y_mean))
    return lambda x_new: y_mean + ((x_new - mu) / sd) @ coef


def walk_forward_forecast(prices: pd.Series, local_tz: str = LOCAL_TZ,
                          min_train_days: int = MIN_TRAIN_DAYS, alpha: float = RIDGE_ALPHA,
                          window_days: int | None = None) -> pd.DataFrame:
    """Hourly forecasts for every Pacific day that has complete features and at least
    `min_train_days` complete earlier days to train on. Indexed by the actual UTC hours.
    window_days=None trains on all earlier days (expanding); an integer keeps only training
    days in [target - window_days, target). The first forecast day is the same either way.

    Columns: actual, forecast (ridge), persistence (= lag1), mean7 (trailing 7-day same-hour
    mean). Prices in $/MWh. The two naive columns are baselines for accuracy comparison."""
    prices = prices.sort_index()
    grid = day_hour_grid(prices, local_tz)
    features = build_features(grid)
    complete = pd.concat([f.notna().all(axis=1) for f in features.values()], axis=1).all(axis=1)
    days = grid.index[complete]

    preds = {}
    for i in range(min_train_days, len(days)):
        target = days[i]
        train = days[:i]  # strictly before target
        if window_days is not None:
            train = train[train >= target - pd.Timedelta(days=window_days)]
        row = []
        for h in range(24):
            f = features[h]
            predict = fit_ridge(f.loc[train, FEATURES].to_numpy(), f.loc[train, "y"].to_numpy(), alpha)
            row.append(float(predict(f.loc[[target], FEATURES].to_numpy())[0]))
        preds[target] = row
    forecast_grid = pd.DataFrame.from_dict(preds, orient="index", columns=range(24))

    local = prices.index.tz_convert(local_tz)
    key = pd.MultiIndex.from_arrays([pd.to_datetime(local.date), local.hour])
    lag1 = pd.DataFrame({h: features[h]["lag1"] for h in range(24)})
    mean7 = pd.DataFrame({h: features[h]["mean7"] for h in range(24)})
    out = pd.DataFrame({
        "actual": prices.to_numpy(),
        "forecast": forecast_grid.stack().reindex(key).to_numpy(),
        "persistence": lag1.stack().reindex(key).to_numpy(),
        "mean7": mean7.stack().reindex(key).to_numpy(),
    }, index=prices.index)
    return out[out["forecast"].notna()]


def settle(schedule: DispatchResult, actual: pd.Series, config: BatteryConfig) -> DispatchResult:
    """Score a planned schedule at actual prices. Physical quantities are unchanged."""
    dt = config.dt_hours
    gross = float(((schedule.discharge_mw - schedule.charge_mw) * actual).sum() * dt)
    degradation = config.degradation_cost_per_mwh * float(schedule.discharge_mw.sum()) * dt
    return DispatchResult(charge_mw=schedule.charge_mw, discharge_mw=schedule.discharge_mw,
                          soc_mwh=schedule.soc_mwh, price=actual.copy(), gross_revenue=gross,
                          degradation_cost=degradation, net_profit=gross - degradation)


def forecast_driven_multiday(prices: pd.Series, config: BatteryConfig = BatteryConfig(),
                             forecast: pd.DataFrame | None = None, column: str = "forecast",
                             local_tz: str = LOCAL_TZ) -> DispatchResult:
    """Plan each Pacific day with the LP on forecast prices (`column` of `forecast`, computed
    by walk_forward_forecast if not given) and settle at actual prices. Only days with a
    forecast get a result. `daily` matches solve_multiday's columns plus planned_net_profit
    (what the LP expected to earn on its forecast)."""
    if forecast is None:
        forecast = walk_forward_forecast(prices, local_tz)
    forecast = forecast.sort_index()
    dates = forecast.index.tz_convert(local_tz).date

    charge_parts, discharge_parts, soc_parts, daily_rows = [], [], [], []
    for date in pd.unique(dates):
        day = forecast[dates == date]
        plan = solve_day(day[column].rename("LMP"), config)
        result = settle(plan, day["actual"], config)
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
            "n_hours": len(day),
            "planned_net_profit": plan.net_profit,
        })

    daily = pd.DataFrame(daily_rows).set_index("pacific_date")
    return DispatchResult(
        charge_mw=pd.concat(charge_parts),
        discharge_mw=pd.concat(discharge_parts),
        soc_mwh=pd.concat(soc_parts),
        price=forecast["actual"].copy(),
        gross_revenue=float(daily["gross_revenue"].sum()),
        degradation_cost=float(daily["degradation_cost"].sum()),
        net_profit=float(daily["net_profit"].sum()),
        daily=daily,
    )


if __name__ == "__main__":
    months = list(pd.period_range("2024-05", "2024-07", freq="M"))
    series = to_price_series(filter_hub_market(read_months(months), hub="SP15"))
    fc = walk_forward_forecast(series)
    err = fc[["forecast", "persistence", "mean7"]].sub(fc["actual"], axis=0)
    print(f"{fc.index.min()} -> {fc.index.max()}, {len(fc)} hours")
    print("MAE $/MWh:", err.abs().mean().round(2).to_dict())
    result = forecast_driven_multiday(series, forecast=fc)
    print(f"Forecast-driven SP15 net profit over window: ${result.net_profit:,.0f}")
