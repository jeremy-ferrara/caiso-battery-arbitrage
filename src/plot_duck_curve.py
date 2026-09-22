"""Plot average day-ahead LMP by hour of day (Pacific time) for one hub and month.

Usage:
    python src/plot_duck_curve.py --month 2025-06 --hub SP15
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from load_prices import LOCAL_TZ, month_path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Light-mode tokens from the dataviz reference palette
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, BASELINE, SERIES = "#e1e0d9", "#c3c2b7", "#2a78d6"
FONT = ["Helvetica Neue", "Arial", "DejaVu Sans"]


def hourly_profile(month: str, hub: str) -> pd.Series:
    df = pd.read_parquet(month_path(pd.Period(month)))
    da = df[(df["Market"] == "DAY_AHEAD_HOURLY") & (df["Location"] == f"TH_{hub}_GEN-APND")].copy()
    da["hour"] = da["Interval Start"].dt.tz_convert(LOCAL_TZ).dt.hour
    return da.groupby("hour")["LMP"].mean()


def plot(profile: pd.Series, month: str, hub: str) -> Path:
    plt.rcParams["font.family"] = FONT
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(profile.index, profile.values, color=SERIES, linewidth=2, solid_capstyle="round", solid_joinstyle="round")

    # Label only the story: the midday trough and the evening peak
    for hour, ha, dy in ((profile.idxmin(), "center", -22), (profile.idxmax(), "center", 12)):
        value = profile[hour]
        ax.plot(hour, value, "o", markersize=9, color=SERIES, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.annotate(f"${value:,.0f} at {pd.Timestamp(2000, 1, 1, hour):%-I %p}", (hour, value),
                    textcoords="offset points", xytext=(0, dy), ha=ha, fontsize=10, color=INK_2)

    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{pd.Timestamp(2000, 1, 1, h):%-I %p}" for h in range(0, 24, 3)])
    top = max(10, profile.max() * 1.15)
    bottom = min(0, profile.min() * 1.3)
    ax.set_ylim(bottom, top)
    ax.set_ylabel("$/MWh", color=INK_2, fontsize=10)
    ax.set_xlabel("Hour of day (Pacific time, hour beginning)", color=INK_2, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    if bottom < 0:
        ax.axhline(0, color=BASELINE, linewidth=1)

    fig.text(0.07, 0.95, f"Day-ahead prices dip at midday and spike in the evening: {hub}, {month}",
             fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.07, 0.905, "Average day-ahead LMP by hour of day, trading hub " + f"TH_{hub}_GEN-APND",
             fontsize=10, color=INK_2, ha="left", va="top")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.13)

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"duck_curve_{hub.lower()}_{month}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--month", required=True, help="e.g. 2025-06")
    parser.add_argument("--hub", default="SP15", choices=["SP15", "NP15"])
    args = parser.parse_args()

    profile = hourly_profile(args.month, args.hub)
    print(profile.round(2).to_string())
    print(f"Saved {plot(profile, args.month, args.hub)}")
