# CAISO Day-Ahead Battery Arbitrage: Backtest Findings

A 100 MW / 400 MWh battery arbitraging CAISO day-ahead hourly prices at the SP15 and NP15 trading hubs, Jul 2023 - Aug 2026, under three dispatch strategies. All dollar figures are net profit after a degradation charge unless stated. Figures come from the scripts listed under [Reproducing](#reproducing).

## Summary

- **Perfect foresight (the upper bound)** earned $12.58M at SP15 and $5.99M at NP15 over the full 38 months.
- **Comparison window.** On the 1,121 days per hub where all three strategies have results:
  - Rule-based dispatch captured **57.0%** of perfect-foresight profit (both hubs pooled).
  - Forecast-driven dispatch captured **87.2%** (SP15 91.1%, NP15 79.3%).
- **Volume, not forecasting, explains most of the gap between the two.** The rule can move only half a cycle per day; the forecast-driven LP can do a full cycle. Against perfect foresight capped at the rule's volume, the rule captures 89.5% (SP15) and 79.5% (NP15).
- **Accuracy and the naive baseline.** The ridge forecast is less accurate than simply reusing yesterday's prices over the full period, and dispatching on yesterday's prices earns slightly more (89.4% vs 87.2% pooled). The whole shortfall is in the first 11 months (Aug 2023 - Jul 2024), when the model had little training data. From Jul 8 2024 onward the ridge forecast is more accurate and earns more than the baseline at both hubs, by narrow margins.
- **Evening-peak bias.** The forecast over-predicts evening prices. A post-hoc test with rolling training windows removed most of that bias, but changed capture by at most 2.6 points, and not consistently upward.
- **Market trend.** Day-ahead arbitrage value fell steeply. SP15 perfect-foresight profit fell from $47.7/kW-yr in 2024 to $19.4/kW-yr in the trailing 12 months (Sep 2025 - Aug 2026).

## 1. Data

- **Prices.** CAISO day-ahead hourly LMPs at `TH_SP15_GEN-APND` and `TH_NP15_GEN-APND`, pulled with `gridstatus` from CAISO OASIS.
- **Date range.** Jul 1 2023 - Aug 31 2026: 38 months, 1,158 days per hub. OASIS retains only about the trailing 3 years of LMP data, so earlier years could not be pulled.
- **Quality.** No missing hours or duplicates. Every Pacific day has 24 hours except the 6 DST-transition days (23 or 25).
- **Unused.** Real-time 15-minute prices were downloaded but are not used. No load, net load, or renewable data was used.
- **Time handling.** Timestamps are stored in UTC. Day boundaries and hour-of-day use Pacific time.

## 2. Methodology

### Battery model (identical for every strategy)

- 100 MW charge/discharge limit, 400 MWh capacity, state-of-charge bounds 0-400 MWh.
- 85% round-trip efficiency, split evenly between charging and discharging.
- Max 1 equivalent full cycle per day (400 MWh discharged).
- $15/MWh degradation cost on MWh discharged, subtracted from revenue.
- State of charge pinned to 50% (200 MWh) at every Pacific midnight, so each day is independent.
- Revenue is settled at the hub's actual day-ahead price.

### Strategies

- **Perfect foresight (PF).** A linear program (`cvxpy`) maximizes the day's net profit on that day's actual prices. This is the upper bound.
- **Rule-based.** One charge block and one discharge block per day, chosen from the average hour-of-day price profile over the trailing 30 days (excluding the current day).
  - The battery idles if the profile doesn't cover efficiency losses plus degradation.
  - With SoC pinned at 50%, one charge block plus one discharge block can move at most 200 MWh through storage per day: half the LP's cycle allowance.
- **Forecast-driven.** The same LP as PF, solved on a forecast of the day's prices, with the resulting schedule settled at actual prices.
- **Persistence LP (diagnostic only).** The same LP solved on the previous day's actual prices. It isolates what the forecast model adds over the most naive forecast. It is not one of the three compared strategies.

### Forecast model

- **Decision time.** Day-ahead bids for operating day D close around 10:00 PT on D-1. DA prices for D-1 were published around 13:00 PT on D-2. The forecast for D therefore uses day-ahead prices through D-1 and nothing from D.
  - Real-time prices are not used: at bid time they are known for only part of D-1.
- **Features, per Pacific hour h of D:**
  - price at h on D-1 and on D-7;
  - mean price at h over D-1 to D-7;
  - D-1's daily mean and max-min spread;
  - day-of-week dummies;
  - day of year encoded as sine and cosine. Month dummies weren't used because the training window starts in July, so most months have no training data in the first year.
- **Model.** One ridge regression per hour of day (24 models), with a ridge penalty of 10 on standardized features.
- **Training.** The model is refit from scratch for every target day on all earlier days (expanding window). Standardization statistics come from the training rows only.
- **No tuning.** The penalty and the feature list were set before seeing any results and were not tuned afterwards.
- **First forecast.** Aug 7 2023, after 7 days of lag history and a 30-day minimum training set.

### Lookahead controls

- Every feature for day D is built from days strictly before D. The training set for D contains only days before D.
- A unit test (`tests/test_forecast.py`) replaces every price on and after D with random values and asserts that D's forecast is unchanged. It runs for the expanding window and for a rolling window.
- A second test checks that, with a rolling window, data older than the window plus the 7 lag days doesn't affect D's forecast.

### Comparison window and capture rate

- Strategy comparisons use only dates where every strategy has a result: **Aug 7 2023 - Aug 31 2026, 1,121 days per hub.**
- Rule-based has no result for Jul 1-7 2023. Forecast-driven has none for Jul 1 - Aug 6 2023.
- Capture rate = sum of a strategy's net profit / sum of PF net profit over the same days (a ratio of sums, not an average of daily ratios).
- Seasons are by calendar month (Winter = Dec, Jan, Feb), and December is grouped with its own calendar year.

## 3. Three-strategy comparison

### Headline (1,121 matched days)

| | SP15 | NP15 | Both |
|---|---:|---:|---:|
| Perfect foresight | $11.70M | $5.73M | $17.43M |
| Rule-based | $6.64M | $3.30M | $9.94M |
| Forecast-driven | $10.66M | $4.55M | $15.21M |
| Persistence LP (diagnostic) | $10.76M | $4.82M | $15.58M |
| **Rule-based capture** | **56.8%** | **57.6%** | **57.0%** |
| **Forecast-driven capture** | **91.1%** | **79.3%** | **87.2%** |
| Persistence LP capture | 92.0% | 84.0% | 89.4% |

### Annual profit per battery

Each strategy's full date range, not the matched window. Full calendar years show $/kW-yr. Partial years show actual dollars and are not annualized, because profit is strongly seasonal.

| Year | PF SP15 | Rule SP15 | Fcst SP15 | PF NP15 | Rule NP15 | Fcst NP15 |
|---|---:|---:|---:|---:|---:|---:|
| 2023 (partial) | $3.33M | $2.22M | $2.14M | $1.29M | $0.89M | $0.67M |
| 2024 | $4.77M ($47.7) | $2.83M ($28.3) | $4.45M ($44.5) | $2.16M ($21.6) | $1.38M ($13.8) | $1.76M ($17.6) |
| 2025 | $3.13M ($31.3) | $1.58M ($15.8) | $2.88M ($28.8) | $1.65M ($16.5) | $0.79M ($7.9) | $1.41M ($14.1) |
| 2026 Jan-Aug | $1.35M | $0.69M | $1.19M | $0.89M | $0.45M | $0.72M |
| Trailing 12 mo (Sep 2025 - Aug 2026) | $1.94M ($19.4) | $0.94M ($9.4) | $1.65M ($16.5) | $1.04M ($10.4) | $0.44M ($4.4) | $0.77M ($7.7) |

2023 days: PF 184 (from Jul 1), rule 177 (from Jul 8), forecast-driven 147 (from Aug 7).

**These figures cover day-ahead energy arbitrage only.** They are at hub prices, with a $15/MWh degradation charge deducted. They exclude real-time market revenue, ancillary services, and resource adequacy payments, which make up a large share of actual CAISO battery revenue. Real batteries also settle at their own node, not the hub. Compare them to the energy-arbitrage component of published battery revenue figures (e.g. CAISO Department of Market Monitoring reports), not to total revenue.

### Capture by year (matched days)

| Year | Rule both | Rule SP15 | Rule NP15 | Fcst both | Fcst SP15 | Fcst NP15 |
|---|---:|---:|---:|---:|---:|---:|
| 2023 (Aug 7 - Dec) | 64.0% | 63.0% | 66.5% | 80.6% | 87.3% | 64.8% |
| 2024 | 60.8% | 59.4% | 63.8% | 89.6% | 93.3% | 81.4% |
| 2025 | 49.5% | 50.4% | 47.7% | 89.7% | 92.0% | 85.3% |
| 2026 (Jan - Aug) | 50.8% | 51.0% | 50.4% | 84.8% | 88.1% | 80.0% |

### Capture by season (matched days, all years pooled)

| Season | Rule both | Rule SP15 | Rule NP15 | Fcst both | Fcst SP15 | Fcst NP15 |
|---|---:|---:|---:|---:|---:|---:|
| Winter (DJF) | 43.2% | 47.5% | 28.0% | 76.3% | 85.3% | 44.7% |
| Spring (MAM) | 50.3% | 50.0% | 50.8% | 93.3% | 95.9% | 88.5% |
| Summer (JJA) | 71.2% | 69.7% | 73.2% | 90.2% | 91.3% | 88.7% |
| Fall (SON) | 59.4% | 60.0% | 57.6% | 81.4% | 88.6% | 59.6% |

Season mixes are uneven. Summer covers Jun-Aug of 2024-2026 plus Aug 2023. 2023 contributes only Aug-Dec, so its high-spread months weigh more heavily on summer and fall.

### Trading activity (matched days)

| | Mean MWh discharged/day | Idle days | Days with negative profit |
|---|---:|---:|---:|
| SP15 Perfect foresight | 320 | 66 | 0 |
| SP15 Forecast-driven | 351 | 4 | 97 |
| SP15 Rule-based | 180 | 25 | 70 |
| SP15 Persistence LP | 320 | 66 | 59 |
| NP15 Perfect foresight | 233 | 199 | 0 |
| NP15 Forecast-driven | 255 | 82 | 235 |
| NP15 Rule-based | 150 | 208 | 137 |
| NP15 Persistence LP | 233 | 199 | 139 |

No day's rule-based or forecast-driven result exceeded PF. The persistence LP replays PF's schedule from the previous day, which is why its mean discharge and idle-day count match PF's exactly.

### Volume handicap of the rule-based strategy

With SoC pinned at 50% at midnight, the rule can deliver at most 184 MWh per day. PF reaches a full cycle by discharging in the morning, recharging midday, and discharging again in the evening. Over the full period, PF delivers more than 184 MWh on 85% of SP15 days and 55% of NP15 days.

Against PF capped at the rule's daily volume, the rule captures 89.5% (SP15) and 79.5% (NP15) on the matched days, compared with 56.8% / 57.6% against uncapped PF. So the rule-based headline mostly measures half the trading volume, not lack of foresight. The forecast strategies are not volume-limited: they capture 144% (SP15) and 110% (NP15) of capped PF, so that baseline doesn't apply to them.

## 4. Forecast accuracy

Accuracy is reported separately from profit. Hourly forecasts over the 1,121 forecast days, in $/MWh. "Persistence" is the previous day's price at the same hour; "mean7" is the trailing 7-day mean at that hour.

| | MAE | RMSE | Bias | Demeaned MAE (shape) | Within-day rank corr. | Mean forecast daily spread | Actual daily spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| SP15 ridge | 7.8 | 19.7 | +1.6 | 6.2 | 0.894 | 65.7 | 55.3 |
| SP15 persistence | 7.3 | 17.8 | 0.0 | 5.4 | 0.905 | 55.3 | 55.3 |
| SP15 mean7 | 9.6 | 22.0 | +0.1 | 6.4 | 0.918 | 54.0 | 55.3 |
| NP15 ridge | 6.7 | 17.6 | +1.5 | 5.1 | 0.864 | 46.4 | 41.0 |
| NP15 persistence | 6.4 | 15.6 | 0.0 | 4.7 | 0.869 | 41.1 | 41.0 |
| NP15 mean7 | 8.5 | 19.6 | 0.0 | 5.3 | 0.886 | 39.4 | 41.0 |

- Demeaned MAE subtracts each day's mean from both forecast and actual, so it measures shape error only.
- Rank correlation is the Spearman correlation between a day's 24 forecast and 24 actual prices, averaged over days.
- Mean7 was used only as an accuracy baseline; it was not run through the LP.

Observations:
- **Where ridge wins and loses.** Ridge has lower MAE than persistence at hours 5-15 at both hubs, and higher MAE at hours 16-23 and 0-4. At hours 17-20, ridge MAE is 1.2-1.7x persistence's. Full hourly tables are in the [appendix](#appendix-mae-and-rmse-by-hour-of-day).
- **Evening bias.** Ridge errors at hours 17-20 are biased upward in every year at both hubs: SP15 +7.4, +8.9, +4.6, +1.9 $/MWh for 2023-2026; NP15 +7.3, +5.1, +3.7, +1.9.
- **Over-trading.** The forecast therefore overstates the daily spread (SP15 $65.7 vs $55.3 actual). The LP acts on it: at SP15 it trades more than PF (351 vs 320 MWh/day) and almost never idles (4 days vs 66). At NP15 it loses money on 235 days vs 139 for the persistence LP.
- **MAE by year (ridge / persistence).** SP15: 2023 13.3 / 10.8, 2024 9.1 / 8.3, 2025 6.0 / 6.2, 2026 5.4 / 5.4. NP15: 2023 10.9 / 8.5, 2024 7.8 / 7.1, 2025 5.1 / 5.5, 2026 5.0 / 5.5.
- **Accuracy and profit diverge.** Ridge has worse MAE than persistence and still captures 91% of PF at SP15. All three forecasts rank the day's hours with correlations of 0.86-0.92. The LP mainly needs the order of hours (cheap midday, expensive evening), which every forecast mostly gets right. Mean7 has the best rank correlation of the three and the worst MAE.

## 5. Rolling-window test (post hoc)

**This test was designed after seeing the results above.** It was motivated by the evening-peak bias in the expanding-window model. Four windows were run on the same backtest period that showed the bias, so this is a diagnostic, not out-of-sample evidence. The forecast-driven results in section 3 use the expanding window, and no other window has been adopted.

**Hypothesis.** The expanding window keeps old, higher-priced evenings (including the Aug 15-16 2023 price spike) while prices fall, and that inflates evening forecasts. If so, a rolling window should shrink the evening bias, reduce the overstated spread, and cut over-trading.

**Setup.** The only change is the training set: for a window of N days, day D trains on [D-N, D) only. Features, decision time, ridge penalty, minimum training set, LP, and battery config are identical, and the lookahead tests cover rolling windows. Windows tested: 90, 180, and 365 days.

**Windows only differ from expanding once the history is longer than the window.** Data starts in Jul 2023, so:
- 90-day diverges from Oct 7 2023;
- 180-day from Jan 5 2024;
- 365-day from Jul 8 2024.

Before those dates each window is the same model as expanding. Results are therefore also shown for **Jul 8 2024 - Aug 31 2026 (785 days), when all four models differ.**

### Evening-peak bias (mean forecast - actual, hours 17-20, $/MWh)

| Hub, year | Expanding | 90-day | 180-day | 365-day | Persistence |
|---|---:|---:|---:|---:|---:|
| SP15 2023 | +7.4 | +10.6 | +7.4 | +7.4 | +0.5 |
| SP15 2024 | +8.9 | +1.1 | +1.9 | +8.3 | 0.0 |
| SP15 2025 | +4.6 | +0.5 | 0.0 | +1.5 | 0.0 |
| SP15 2026 | +1.9 | 0.0 | 0.0 | +0.2 | 0.0 |
| NP15 2023 | +7.3 | +9.6 | +7.3 | +7.3 | +0.5 |
| NP15 2024 | +5.1 | +1.2 | +1.6 | +4.4 | 0.0 |
| NP15 2025 | +3.7 | +0.3 | +0.2 | +1.2 | 0.0 |
| NP15 2026 | +1.9 | 0.0 | 0.0 | +0.2 | 0.0 |

The 2023 values for the 180- and 365-day windows are identical to expanding by construction. The 365-day 2024 value is identical for the first half of the year.

### Accuracy, spread, and dispatch by window

| | Expanding | 90-day | 180-day | 365-day | Persistence |
|---|---:|---:|---:|---:|---:|
| SP15 evening bias, Jul 2024+ | +4.2 | +0.7 | +0.5 | +2.0 | 0.0 |
| NP15 evening bias, Jul 2024+ | +3.3 | +0.7 | +0.7 | +1.3 | 0.0 |
| SP15 evening MAE, Jul 2024+ | 10.8 | 8.9 | 8.9 | 8.8 | 8.0 |
| NP15 evening MAE, Jul 2024+ | 9.5 | 8.6 | 8.4 | 8.2 | 7.6 |
| SP15 MAE, full | 7.8 | 7.5 | 7.6 | 7.6 | 7.3 |
| SP15 RMSE, full | 19.7 | 19.9 | 19.7 | 19.6 | 17.8 |
| NP15 MAE, full | 6.7 | 6.8 | 6.6 | 6.6 | 6.4 |
| NP15 RMSE, full | 17.6 | 18.1 | 17.8 | 17.5 | 15.6 |
| SP15 MAE, Jul 2024+ | 6.2 | 6.0 | 6.1 | 5.8 | 6.3 |
| SP15 RMSE, Jul 2024+ | 11.0 | 11.7 | 11.5 | 10.8 | 11.7 |
| NP15 MAE, Jul 2024+ | 5.4 | 5.4 | 5.3 | 5.2 | 5.7 |
| NP15 RMSE, Jul 2024+ | 9.6 | 10.5 | 10.3 | 9.4 | 10.2 |
| SP15 forecast daily spread, full (actual 55.3) | 65.7 | 58.5 | 58.9 | 61.7 | 55.3 |
| SP15 forecast daily spread, Jul 2024+ (actual 47.1) | 55.2 | 48.4 | 48.5 | 49.4 | 47.1 |
| NP15 forecast daily spread, full (actual 41.0) | 46.4 | 43.7 | 43.9 | 44.1 | 41.1 |
| NP15 forecast daily spread, Jul 2024+ (actual 38.4) | 41.4 | 39.1 | 39.0 | 38.0 | 38.5 |
| SP15 capture, full | 91.1% | 91.8% | 90.8% | 91.2% | 92.0% |
| NP15 capture, full | 79.3% | 81.9% | 79.8% | 80.3% | 84.0% |
| SP15 capture, Jul 2024+ | 91.5% | 91.5% | 90.6% | 91.6% | 90.3% |
| NP15 capture, Jul 2024+ | 84.6% | 84.7% | 84.0% | 86.1% | 83.8% |
| SP15 loss days, full | 97 | 74 | 82 | 84 | 59 |
| NP15 loss days, full | 235 | 179 | 219 | 206 | 139 |
| SP15 loss days, Jul 2024+ | 73 | 53 | 59 | 60 | 47 |
| NP15 loss days, Jul 2024+ | 136 | 105 | 119 | 107 | 93 |

In 2023 every window overstates the SP15 spread about as much as expanding does ($104-107 forecast vs $87 actual).

### Result

1. **Bias: hypothesis supported.** From Jul 2024, evening bias falls from +4.2 (SP15) and +3.3 (NP15) under expanding to +0.5-0.7 under the 90- and 180-day windows. The overstated daily spread largely disappears. The 365-day window removes about half the bias. Trading volume and loss days fall with it.
2. **Profit: hypothesis mostly not supported.** From Jul 2024, capture changed by -0.9 to +1.5 points vs expanding. The largest gain was the 365-day window at NP15 (84.6% → 86.1%); the 180-day window did worse than expanding at both hubs. Fewer losing days did not turn into more profit.
3. **Most of the evening error is not bias.** With bias near zero, evening MAE is still above persistence's (SP15 8.9 vs 8.0). RMSE is *worse* with the 90- and 180-day windows than with expanding (SP15 11.7 / 11.5 vs 11.0), and the large errors on spike evenings remain.
4. **The data doesn't pick a window.** The best and worst windows differ by at most 2.6 capture points, and the ranking changes with hub and period. Picking one from these results would be fitting to the test period.

## 6. Cold-start qualification

The full-period finding that the ridge forecast doesn't beat persistence comes entirely from the first 11 months. Splitting at Jul 8 2024 (the first day all windows differ):

| Period | Hub | MAE ridge | MAE persistence | Capture ridge | Capture persistence | Profit gap (persistence - ridge) |
|---|---|---:|---:|---:|---:|---:|
| Aug 7 2023 - Jul 7 2024 (336 days) | SP15 | 11.64 | 9.83 | 90.7% | 93.9% | +$175k |
| | NP15 | 9.77 | 8.19 | 70.5% | 84.4% | +$297k |
| Jul 8 2024 - Aug 31 2026 (785 days) | SP15 | 6.2 | 6.3 | 91.5% | 90.3% | -$72k |
| | NP15 | 5.4 | 5.7 | 84.6% | 83.8% | -$26k |

- **Early period: little training data.** Training data was weeks to months old and included the Aug 15-16 2023 spike. Through Oct 6 2023 a rolling window can't change anything, because the history is shorter than any window tested. In Aug 7 - Dec 31 2023 at NP15, every ridge variant captured 64.8-68.4% vs 89.1% for persistence; the rule-based strategy captured 66.5%.
- **Later period: ridge wins narrowly.** From Jul 2024 the expanding ridge model beats persistence on MAE and capture at both hubs. All three rolling windows also beat it on both, though the 180-day window at NP15 only barely (84.0% vs 83.8% capture).
- **The margins are small** (0.8-1.2 capture points for expanding). This split was chosen after seeing the results, which is another reason to read it as a qualification, not a claim of skill.

## 7. Market price trends (day-ahead)

Average daily spread (mean of the 4 highest-priced hours minus mean of the 4 lowest, $/MWh). Comparisons use the same calendar months in each year so partial years don't distort them:

| Months | Year | SP15 | NP15 |
|---|---|---:|---:|
| Jul-Dec | 2023 | 75.0 | 42.5 |
| | 2024 | 47.5 | 36.8 |
| | 2025 | 30.2 | 21.6 |
| Jan-Aug | 2024 | 58.4 | 39.0 |
| | 2025 | 45.6 | 35.5 |
| | 2026 | 32.5 | 27.7 |

- Daily max-minus-min spread by calendar year (SP15 / NP15): 2023 H2 $100.1 / $58.2, 2024 $65.5 / $46.9, 2025 $44.8 / $35.1, 2026 Jan-Aug $36.3 / $32.0.
- Average day-ahead price (SP15 / NP15): 2023 H2 $55.9 / $57.2, 2024 $32.7 / $40.7, 2025 $32.2 / $38.2, 2026 Jan-Aug $24.8 / $28.0.

Negative-price hours, same calendar months:

| Months | Year | SP15 | NP15 |
|---|---|---:|---:|
| Jul-Dec | 2023 | 58 | 0 |
| | 2024 | 144 | 6 |
| | 2025 | 122 | 1 |
| Jan-Aug | 2024 | 1,041 | 267 |
| | 2025 | 864 | 287 |
| | 2026 | 1,008 | 385 |

Negative hours cluster in Feb-Jun. SP15's peak months are Apr-May, with about 240-300 negative hours in each of 2024, 2025, and 2026.

## 8. Other observations

1. **Arbitrage value fell steeply.** SP15 PF went from $47.7/kW-yr (2024) to $31.3 (2025) to $19.4 (trailing 12 months), and the SP15 top-4/bottom-4 spread from $75 to $30/MWh (Jul-Dec 2023 vs 2025).
   - Negative-price hours did not fall with the spread: Jan-Aug 2026 had about as many as Jan-Aug 2024. Lower, flatter evening peaks would be consistent with this; that has not been checked directly.
   - Rule-based capture fell with spreads (about 64% in 2023 to about 50% in 2025-26). Forecast-driven capture fell less (SP15 93.3% in 2024, 92.0% in 2025, 88.1% in Jan-Aug 2026).
2. **Profit is concentrated in a few days.**
   - Aug 15-16 2023 contributed $510k of SP15 PF profit, 15% of the 2023 SP15 total. Evening DA peaks were $900-934/MWh on Aug 15 and $1,091-1,248/MWh on Aug 16 across the two hubs; checked in the raw data.
   - The top 10% of days produce 34% (SP15) and 45% (NP15) of PF profit, and 35% / 54% of forecast-driven profit.
   - Aug 16 2023 is the best day for every strategy at both hubs. Both forecast LPs earned within $2k of PF that day: Aug 15 had the same evening spike shape, and the forecasts carried it forward.
3. **SP15 earns about twice what NP15 does** under PF ($12.6M vs $6.0M over the full period), despite NP15's higher average price. SP15 has the wider daily spread and many more negative-price hours.
4. **NP15 in winter and fall is the weakest segment for both non-PF strategies.**
   - Forecast-driven captured 44.7% (winter) and 59.6% (fall), vs 63.3% and 75.4% for the persistence LP.
   - Rule-based captured 28.0% in NP15 winter. It idled on 208 NP15 days vs 25 at SP15, almost all in Q4/Q1. PF itself idles on 199 NP15 days, when no trade covers losses plus degradation.
   - Small spreads leave little margin: a few dollars of forecast error turns an idle day into a losing trade.

## 9. Not covered

- **Real-time market.** The 15-minute RT data is downloaded but unused. All results are day-ahead only.
- **Other revenue streams.** Ancillary services, resource adequacy, and nodal (vs hub) settlement are not modeled.
- **Fundamentals features.** No load, net load, or solar/wind forecasts were used. CAISO publishes day-ahead load and renewable forecasts before the bid deadline, so they would be legitimate features; they have not been pulled.
- **Sensitivities.** Degradation cost, the 50% midnight SoC pin, the 1-cycle limit, and the 30-day rule lookback were each fixed at one value.
- **Bias and cold-start fixes.** A direct evening bias correction, and a cold-start fallback (e.g. using persistence until enough history exists), have not been tested. Any choice made now would be informed by these results and would need later data to evaluate fairly.
- **Uncertainty.** No confidence intervals or significance tests were computed for capture-rate differences. Several of the differences above are within 1-2 points.

## Reproducing

```bash
python src/load_prices.py --start 2023-07-01 --end 2026-08-31   # raw prices -> data/raw
python src/backtest.py        # all strategies -> data/processed/daily_results.parquet, forecasts_hourly.parquet
python src/analysis.py        # summary tables -> data/processed/analysis_*.csv
python src/window_test.py     # rolling-window test -> data/processed/window_test_*.csv
python src/tableau_export.py  # dashboard CSVs -> data/processed/{daily_pnl,hourly_dispatch,capture_rates,price_trends,forecast_accuracy}.csv
pytest
```

## Appendix: MAE and RMSE by hour of day

Full forecast period (Aug 7 2023 - Aug 31 2026), $/MWh, Pacific hour. Exp. = expanding window (headline); R90/R180/R365 = rolling windows; Persist. = previous day's price.

SP15:

|   Hour |   MAE Exp. |   MAE R90 |   MAE R180 |   MAE R365 |   MAE Persist. |   RMSE Exp. |   RMSE R90 |   RMSE R180 |   RMSE R365 |   RMSE Persist. |
|-------:|-----------:|----------:|-----------:|-----------:|---------------:|------------:|-----------:|------------:|------------:|----------------:|
|      0 |        5.0 |       4.9 |        4.9 |        4.8 |            4.8 |         8.2 |        8.5 |         8.4 |         8.1 |             8.1 |
|      1 |        4.5 |       4.5 |        4.5 |        4.4 |            4.3 |         7.6 |        7.8 |         7.9 |         7.5 |             7.4 |
|      2 |        4.3 |       4.3 |        4.3 |        4.2 |            4.0 |         7.5 |        7.7 |         7.8 |         7.4 |             7.2 |
|      3 |        4.2 |       4.2 |        4.3 |        4.1 |            4.0 |         7.4 |        7.7 |         7.7 |         7.4 |             7.1 |
|      4 |        4.3 |       4.4 |        4.4 |        4.2 |            4.2 |         7.5 |        7.8 |         7.8 |         7.5 |             7.2 |
|      5 |        4.9 |       4.9 |        4.9 |        4.7 |            5.1 |         8.1 |        8.3 |         8.4 |         8.0 |             8.2 |
|      6 |        5.2 |       5.3 |        5.2 |        5.1 |            6.2 |         8.6 |        8.9 |         8.9 |         8.6 |             9.5 |
|      7 |        5.7 |       6.1 |        5.9 |        5.7 |            7.1 |         9.0 |        9.6 |         9.3 |         9.1 |            10.5 |
|      8 |        7.3 |       7.7 |        7.5 |        7.3 |            8.6 |        10.0 |       10.6 |        10.3 |        10.1 |            12.2 |
|      9 |        7.7 |       7.9 |        7.9 |        7.7 |            8.9 |        10.7 |       10.9 |        10.9 |        10.7 |            12.8 |
|     10 |        7.6 |       7.7 |        7.9 |        7.5 |            8.7 |        10.5 |       10.6 |        10.7 |        10.4 |            12.4 |
|     11 |        7.6 |       7.7 |        7.9 |        7.5 |            8.7 |        10.5 |       10.7 |        10.8 |        10.5 |            12.3 |
|     12 |        7.6 |       7.8 |        7.9 |        7.6 |            8.6 |        10.5 |       10.7 |        10.8 |        10.5 |            12.2 |
|     13 |        7.8 |       8.0 |        8.1 |        7.8 |            8.9 |        10.9 |       11.1 |        11.1 |        10.9 |            12.4 |
|     14 |        8.2 |       8.4 |        8.5 |        8.2 |            9.1 |        11.5 |       11.7 |        11.8 |        11.5 |            12.8 |
|     15 |        8.7 |       8.9 |        9.2 |        8.7 |            9.3 |        13.3 |       13.4 |        13.6 |        13.3 |            13.7 |
|     16 |        9.3 |       9.4 |        9.8 |        9.4 |            9.1 |        16.5 |       16.6 |        16.7 |        16.5 |            15.4 |
|     17 |       12.5 |      11.1 |       11.1 |       11.6 |            9.6 |        32.3 |       31.8 |        31.7 |        32.2 |            26.5 |
|     18 |       17.4 |      14.9 |       15.2 |       16.0 |           11.4 |        51.6 |       51.6 |        51.4 |        51.6 |            44.5 |
|     19 |       20.5 |      15.9 |       16.5 |       18.2 |           12.4 |        51.8 |       53.2 |        52.4 |        51.5 |            45.8 |
|     20 |       11.4 |       9.5 |        9.5 |       10.4 |            7.9 |        28.3 |       28.5 |        28.1 |        28.0 |            23.3 |
|     21 |        6.8 |       6.6 |        6.5 |        6.6 |            5.8 |        13.6 |       13.9 |        13.7 |        13.5 |            11.7 |
|     22 |        4.9 |       5.1 |        5.1 |        5.0 |            4.7 |         8.3 |        8.9 |         8.5 |         8.3 |             7.8 |
|     23 |        4.6 |       4.9 |        4.7 |        4.6 |            4.3 |         7.9 |        8.6 |         8.2 |         7.9 |             7.5 |

NP15:

|   Hour |   MAE Exp. |   MAE R90 |   MAE R180 |   MAE R365 |   MAE Persist. |   RMSE Exp. |   RMSE R90 |   RMSE R180 |   RMSE R365 |   RMSE Persist. |
|-------:|-----------:|----------:|-----------:|-----------:|---------------:|------------:|-----------:|------------:|------------:|----------------:|
|      0 |        4.4 |       4.6 |        4.5 |        4.4 |            4.4 |         7.9 |        8.4 |         8.1 |         7.8 |             7.6 |
|      1 |        4.0 |       4.1 |        4.0 |        3.9 |            3.8 |         7.2 |        7.7 |         7.4 |         7.2 |             6.9 |
|      2 |        3.8 |       3.9 |        3.9 |        3.7 |            3.6 |         7.1 |        7.5 |         7.3 |         7.1 |             6.8 |
|      3 |        3.7 |       3.9 |        3.8 |        3.7 |            3.5 |         7.0 |        7.5 |         7.3 |         7.0 |             6.7 |
|      4 |        3.9 |       4.1 |        4.0 |        3.8 |            3.7 |         7.2 |        7.7 |         7.4 |         7.1 |             6.9 |
|      5 |        4.4 |       4.5 |        4.4 |        4.3 |            4.7 |         7.8 |        8.3 |         8.0 |         7.7 |             7.8 |
|      6 |        4.8 |       5.0 |        4.8 |        4.7 |            5.9 |         8.1 |        8.6 |         8.2 |         8.1 |             9.0 |
|      7 |        5.0 |       5.3 |        5.1 |        5.0 |            6.5 |         8.4 |        9.0 |         8.7 |         8.4 |             9.7 |
|      8 |        5.4 |       5.9 |        5.6 |        5.4 |            6.7 |         9.2 |        9.8 |         9.4 |         9.2 |            10.4 |
|      9 |        5.8 |       6.3 |        6.0 |        5.8 |            6.9 |         9.4 |       10.1 |         9.6 |         9.4 |            10.6 |
|     10 |        6.0 |       6.5 |        6.3 |        6.1 |            7.1 |         9.6 |       10.3 |         9.8 |         9.6 |            10.7 |
|     11 |        6.3 |       6.8 |        6.5 |        6.4 |            7.3 |         9.9 |       10.6 |        10.1 |         9.9 |            11.0 |
|     12 |        6.6 |       7.0 |        6.8 |        6.7 |            7.5 |        10.2 |       10.9 |        10.4 |        10.2 |            11.2 |
|     13 |        7.0 |       7.3 |        7.2 |        7.0 |            7.6 |        10.6 |       11.2 |        10.8 |        10.6 |            11.5 |
|     14 |        7.3 |       7.7 |        7.6 |        7.3 |            7.9 |        11.1 |       11.8 |        11.3 |        11.1 |            11.7 |
|     15 |        7.9 |       8.3 |        8.2 |        8.0 |            8.3 |        12.5 |       13.1 |        12.7 |        12.5 |            12.7 |
|     16 |        8.5 |       8.5 |        8.6 |        8.5 |            8.2 |        15.4 |       15.8 |        15.6 |        15.4 |            14.4 |
|     17 |       10.7 |       9.9 |        9.6 |       10.0 |            8.8 |        26.7 |       26.1 |        25.9 |        26.5 |            22.5 |
|     18 |       14.3 |      13.2 |       13.0 |       13.2 |           10.1 |        42.7 |       42.8 |        42.5 |        42.5 |            35.1 |
|     19 |       17.2 |      15.2 |       15.4 |       16.0 |           11.2 |        49.1 |       51.6 |        51.1 |        49.0 |            41.4 |
|     20 |        9.8 |       9.0 |        8.9 |        9.0 |            7.2 |        25.6 |       26.1 |        25.8 |        25.4 |            21.7 |
|     21 |        5.8 |       6.1 |        5.8 |        5.7 |            5.2 |        11.5 |       12.2 |        11.7 |        11.5 |            10.3 |
|     22 |        4.5 |       4.8 |        4.6 |        4.5 |            4.3 |         7.9 |        8.6 |         8.0 |         7.9 |             7.4 |
|     23 |        4.1 |       4.4 |        4.2 |        4.1 |            3.8 |         7.6 |        8.3 |         7.7 |         7.6 |             7.1 |
