"""Download CAISO hub LMPs (day-ahead hourly + real-time 15-min) via gridstatus.

Writes one parquet file per month to data/raw, holding both markets and both hubs.
Months that already have a file are skipped, so an interrupted pull can be re-run.

Usage:
    python src/load_prices.py --start 2025-06-01 --end 2025-06-30

Every calendar month touched by [start, end] is pulled in full.
"""
import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from gridstatus import CAISO, Markets

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
HUBS = ["TH_SP15_GEN-APND", "TH_NP15_GEN-APND"]
# market -> intervals per hour
MARKETS = {Markets.DAY_AHEAD_HOURLY: 1, Markets.REAL_TIME_15_MIN: 4}
COLUMNS = ["Interval Start", "Interval End", "Market", "Location", "LMP", "Energy", "Congestion", "Loss"]
LOCAL_TZ = "US/Pacific"
RETRIES = 3

logging.getLogger("gridstatus").setLevel(logging.WARNING)


def month_path(month: pd.Period) -> Path:
    return RAW_DIR / f"caiso_lmp_{month}.parquet"


def pull_month(iso: CAISO, month: pd.Period, sleep: int) -> pd.DataFrame:
    """Pull both markets for one calendar month (Pacific time). Timestamps returned in UTC."""
    start, end = month.start_time, (month + 1).start_time  # end is exclusive
    hours = (end.tz_localize(LOCAL_TZ) - start.tz_localize(LOCAL_TZ)).total_seconds() / 3600

    frames = []
    for market, per_hour in MARKETS.items():
        for attempt in range(1, RETRIES + 1):
            try:
                df = iso.get_lmp(start, market=market, end=end, locations=HUBS, sleep=sleep)
                break
            except Exception as exc:
                if attempt == RETRIES:
                    raise
                wait = 30 * attempt
                print(f"  {market.value} attempt {attempt} failed ({exc!r}); retrying in {wait}s")
                time.sleep(wait)
        expected = int(hours * per_hour * len(HUBS))
        if len(df) != expected:
            print(f"  WARNING {market.value}: got {len(df)} rows, expected {expected}")
        frames.append(df[COLUMNS])

    out = pd.concat(frames, ignore_index=True)
    found = set(zip(out["Market"], out["Location"]))
    missing = {(m.value, h) for m in MARKETS for h in HUBS} - found
    if missing:  # don't write a partial month, or "skip existing" would hide it
        raise RuntimeError(f"{month}: missing market/hub combos {sorted(missing)}")

    for col in ("Interval Start", "Interval End"):
        out[col] = out[col].dt.tz_convert("UTC")
    return out


def load_range(start, end, sleep: int = 5) -> list[pd.Period]:
    months = list(pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"))
    iso = CAISO()
    for month in months:
        path = month_path(month)
        if path.exists():
            print(f"{month}: already downloaded, skipping")
            continue
        print(f"{month}: pulling...")
        t0 = time.time()
        df = pull_month(iso, month, sleep)
        tmp = path.with_suffix(".parquet.tmp")  # write-then-rename so a crash never leaves a bad file
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        print(f"{month}: saved {len(df):,} rows in {time.time() - t0:.0f}s -> {path.name}")
    return months


def read_months(months: list[pd.Period]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(month_path(m)) for m in months], ignore_index=True)


def summarize(df: pd.DataFrame) -> None:
    print(f"\nTotal rows: {len(df):,}")
    print(f"Range (UTC):     {df['Interval Start'].min()} -> {df['Interval End'].max()}")
    print(f"Range (Pacific): {df['Interval Start'].min().tz_convert(LOCAL_TZ)} -> "
          f"{df['Interval End'].max().tz_convert(LOCAL_TZ)}")
    print("\nRows by market and hub:")
    print(df.groupby(["Market", "Location"]).agg(rows=("LMP", "size"), mean_lmp=("LMP", "mean"),
                                                 min_lmp=("LMP", "min"), max_lmp=("LMP", "max")).round(2))
    print("\nSample rows:")
    print(df.groupby("Market").head(3).to_string(index=False))
    for market in MARKETS:
        print(f"{market.value} present: {market.value in set(df['Market'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="e.g. 2025-06-01")
    parser.add_argument("--end", required=True, help="e.g. 2025-06-30")
    parser.add_argument("--sleep", type=int, default=5, help="seconds between CAISO API calls (rate limit)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    summarize(read_months(load_range(args.start, args.end, args.sleep)))
