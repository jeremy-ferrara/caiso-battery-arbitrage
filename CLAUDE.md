# CLAUDE.md

## Project goal

Backtest battery storage arbitrage in CAISO. Simulate a 100 MW / 400 MWh battery against real CAISO day-ahead and real-time prices (2021-2025, pulled via `gridstatus`) and compare three strategies:

1. **Perfect foresight**: optimal dispatch on known prices (upper bound)
2. **Rule-based**: simple charge-low / discharge-high schedule
3. **Forecast-driven**: optimal dispatch on forecast prices, settled at actual prices

Deliverables: a Tableau Public dashboard and a one-page market note. This is a portfolio project.

## Tech stack

- Python, virtualenv in `.venv`
- `gridstatus` for CAISO price data
- `pandas`, `numpy` for data handling
- `cvxpy` for the linear program
- `matplotlib` for plots, `jupyter` for notebooks
- Tableau Public for the final dashboard

## Physical model (LP constraints)

- Power limit: 100 MW charge / discharge
- Energy capacity: 400 MWh, with state-of-charge (SoC) min/max bounds
- Round-trip efficiency losses
- Cycle limit (max equivalent full cycles per day or year)
- Degradation cost per MWh throughput, subtracted from revenue

## Layout

- `data/raw`, `data/processed`: price data, gitignored (only `.gitkeep` is tracked)
- `src/`: reusable code (loading, optimizer, strategies)
- `notebooks/`: exploration and result write-ups
- `tableau/`: dashboard extracts and notes
- `reports/`: market note

## Conventions

- Put reusable logic in `src/`, not in notebooks; notebooks import from `src`.
- Never commit raw or processed data, `.env`, or credentials.
- Units: power in MW, energy in MWh, prices in $/MWh. Put units in variable or column names where ambiguous.
- Timestamps are timezone-aware. Store in UTC; convert to `America/Los_Angeles` only for display or day-boundary logic.
- Strategies share one interface and one battery config so results are comparable.
- Keep the same constraint parameters across all three strategies.
