from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

HISTORICAL_WINDOW_DAYS = 100

# Five COB (close-of-business) snapshots. Each COB's VaR uses a 100-day rolling window
# of historical scenario dates ending the day BEFORE that COB (T-1 back 100 days) — the
# window shifts by one day between consecutive COBs.
COB_DATES: tuple[str, ...] = (
    "2026-07-06",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
    "2026-07-10",
)


@dataclass(frozen=True)
class TradeConfig:
    trade_id: str
    desk: str
    product: str
    risk_factors: tuple[str, ...]
    booking_cob: str
    # Last COB the trade is still on the book for. None = alive through the final COB.
    maturity_cob: str | None = None


RISK_FACTOR_TYPES = {
    "SOFR": ("IR Delta", "bps"),
    "SONIA": ("IR Delta", "bps"),
    "ESTER": ("IR Delta", "bps"),
    "USD GBP": ("FX Delta", "pct"),
    "USD EUR": ("FX Delta", "pct"),
    "USD JPY": ("FX Delta", "pct"),
    "Vodafone Spot": ("Equity Delta", "pct"),
    "HSBC Spot": ("Equity Delta", "pct"),
    "SPX": ("Equity Delta", "pct"),
    "GBP Swaption Vol": ("Vega", "vol_pt"),
    "EUR Credit Spread": ("CS01", "bps"),
    "Brent": ("Commodity Delta", "pct"),
}

# The book evolves day to day as trades mature and new trades are booked:
#   COB 1 (07-06): 20 trades (T1-T20, all booked)
#   COB 2 (07-07): 17 trades (T5, T13, T19 matured overnight)
#   COB 3 (07-08): 22 trades (T21-T25 newly booked)
#   COB 4 (07-09): 20 trades (T2, T21 matured overnight)
#   COB 5 (07-10): 22 trades (T8 matured overnight; T26-T28 newly booked)
TRADES: tuple[TradeConfig, ...] = (
    TradeConfig("T1", "D1", "IR Swap", ("SOFR", "USD GBP"), booking_cob="2026-07-06"),
    TradeConfig("T2", "D1", "IR Swap", ("SONIA", "USD GBP"), booking_cob="2026-07-06", maturity_cob="2026-07-08"),
    TradeConfig("T3", "D1", "FX Swap", ("ESTER", "USD EUR"), booking_cob="2026-07-06"),
    TradeConfig("T4", "D1", "Equity Option", ("Vodafone Spot", "USD GBP", "SOFR"), booking_cob="2026-07-06"),
    TradeConfig("T5", "D1", "FX Forward", ("USD JPY",), booking_cob="2026-07-06", maturity_cob="2026-07-06"),
    TradeConfig("T6", "D2", "FX Forward", ("USD GBP",), booking_cob="2026-07-06"),
    TradeConfig("T7", "D2", "FX Swap", ("USD EUR", "ESTER"), booking_cob="2026-07-06"),
    TradeConfig("T8", "D2", "Equity Total Return Swap", ("HSBC Spot", "USD GBP", "ESTER"), booking_cob="2026-07-06", maturity_cob="2026-07-09"),
    TradeConfig("T9", "D2", "FX Forward", ("USD EUR",), booking_cob="2026-07-06"),
    TradeConfig("T10", "D2", "IR Forward", ("ESTER", "USD EUR"), booking_cob="2026-07-06"),
    TradeConfig("T11", "D3", "Cash Equity", ("Vodafone Spot",), booking_cob="2026-07-06"),
    TradeConfig("T12", "D3", "Equity Total Return Swap", ("Vodafone Spot", "USD GBP", "SONIA"), booking_cob="2026-07-06"),
    TradeConfig("T13", "D3", "FX Forward", ("USD JPY",), booking_cob="2026-07-06", maturity_cob="2026-07-06"),
    TradeConfig("T14", "D3", "IR Swap", ("SONIA", "USD GBP"), booking_cob="2026-07-06"),
    TradeConfig("T15", "D3", "Cash Equity", ("HSBC Spot",), booking_cob="2026-07-06"),
    TradeConfig("T16", "D4", "Rates Option", ("SOFR", "GBP Swaption Vol"), booking_cob="2026-07-06"),
    TradeConfig("T17", "D4", "Credit Index Swap", ("EUR Credit Spread", "USD EUR"), booking_cob="2026-07-06"),
    TradeConfig("T18", "D4", "Equity Option", ("SPX", "USD GBP"), booking_cob="2026-07-06"),
    TradeConfig("T19", "D5", "Commodity Forward", ("Brent", "USD EUR"), booking_cob="2026-07-06", maturity_cob="2026-07-06"),
    TradeConfig("T20", "D5", "Macro Hedge", ("SOFR", "SPX", "USD JPY"), booking_cob="2026-07-06"),
    TradeConfig("T21", "D2", "FX Forward", ("USD EUR",), booking_cob="2026-07-08", maturity_cob="2026-07-08"),
    TradeConfig("T22", "D1", "IR Swap", ("SOFR", "ESTER"), booking_cob="2026-07-08"),
    TradeConfig("T23", "D3", "Equity Option", ("SPX", "USD GBP"), booking_cob="2026-07-08"),
    TradeConfig("T24", "D4", "FX Swap", ("USD JPY", "ESTER"), booking_cob="2026-07-08"),
    TradeConfig("T25", "D5", "Commodity Forward", ("Brent", "USD GBP"), booking_cob="2026-07-08"),
    TradeConfig("T26", "D1", "FX Forward", ("USD EUR",), booking_cob="2026-07-10"),
    TradeConfig("T27", "D2", "Cash Equity", ("HSBC Spot",), booking_cob="2026-07-10"),
    TradeConfig("T28", "D3", "IR Swap", ("SONIA", "ESTER"), booking_cob="2026-07-10"),
)


def _stable_index(value: str) -> int:
    """Deterministic hash. Uses a real digest (not a positional weighted sum) so that near-identical
    inputs — e.g. COB dates that differ only in the last digit — don't produce correlated outputs."""
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def _trades_on_cob(cob_date: str) -> list[TradeConfig]:
    cob_idx = COB_DATES.index(cob_date)
    live = []
    for trade in TRADES:
        booking_idx = COB_DATES.index(trade.booking_cob)
        maturity_idx = COB_DATES.index(trade.maturity_cob) if trade.maturity_cob else len(COB_DATES) - 1
        if booking_idx <= cob_idx <= maturity_idx:
            live.append(trade)
    return live


def _historical_window(cob_date: str) -> pd.DatetimeIndex:
    """The 100-day rolling window for a COB: ends the day before COB, rolls back from there."""
    window_end = pd.Timestamp(cob_date) - pd.Timedelta(days=1)
    return pd.date_range(end=window_end, periods=HISTORICAL_WINDOW_DAYS, freq="D")


def _historical_union() -> pd.DatetimeIndex:
    """Every historical date needed across all 5 COBs' rolling windows, deduplicated."""
    all_dates = pd.DatetimeIndex([])
    for cob_date in COB_DATES:
        all_dates = all_dates.union(_historical_window(cob_date))
    return all_dates.sort_values()


def _sensitivity_value(cob_date: str, trade_id: str, risk_factor: str) -> float:
    """Re-marked every COB: a stable base per (trade, risk_factor) plus a daily drift of up
    to +/-10%, so the same trade's exposure moves day to day like a real book being re-risked,
    rather than staying frozen for as long as the trade is on the book."""
    seed = _stable_index(f"{trade_id}-{risk_factor}")
    magnitude = 1 + (seed % 95)
    sign = -1 if seed % 3 == 0 else 1
    base = float(sign * magnitude)
    drift_seed = _stable_index(f"{trade_id}-{risk_factor}-{cob_date}-drift")
    drift_pct = ((drift_seed % 21) - 10) * 0.01
    return round(base * (1 + drift_pct), 4)


def _shock_value(historical_date: str, risk_factor: str) -> float:
    """Pure function of (date, risk_factor) — the same historical date must produce the same
    shock regardless of which COB's rolling window it's viewed from, since it's the same market data."""
    sensitivity_type, unit = RISK_FACTOR_TYPES[risk_factor]
    seed = _stable_index(f"{risk_factor}-{historical_date}")
    direction = (seed % 21) - 10
    if unit == "bps":
        return float(direction * 8 + ((_stable_index(historical_date) % 5) - 2) * 3)
    if unit == "vol_pt":
        return round(direction * 0.35, 4)
    return round(float(direction), 4)


def _residual_pnl(cob_date: str, trade_id: str, risk_factor: str, historical_date: str, linear_pnl: float) -> float:
    """Non-linear/convexity effect a linear sensitivity model can't capture, bounded to +/-5% of the linear estimate."""
    seed = _stable_index(f"{trade_id}-{risk_factor}-{cob_date}-{historical_date}-residual")
    pct = ((seed % 11) - 5) * 0.01
    return round(linear_pnl * pct, 4)


def generate_sample_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)

    trade_rows: list[dict[str, object]] = []
    for cob_date in COB_DATES:
        for trade in _trades_on_cob(cob_date):
            for risk_factor in trade.risk_factors:
                sensitivity_type, _ = RISK_FACTOR_TYPES[risk_factor]
                trade_rows.append(
                    {
                        "cob_date": cob_date,
                        "trade_id": trade.trade_id,
                        "risk_factor": risk_factor,
                        "desk": trade.desk,
                        "sensitivity_type": sensitivity_type,
                        "sensitivity_value": _sensitivity_value(cob_date, trade.trade_id, risk_factor),
                        "product": trade.product,
                    }
                )

    scenario_rows: list[dict[str, object]] = []
    for date in _historical_union():
        historical_date = date.strftime("%Y-%m-%d")
        scenario_name = f"Historical {historical_date}"
        for risk_factor, (_, unit) in RISK_FACTOR_TYPES.items():
            scenario_rows.append(
                {
                    "historical_date": historical_date,
                    "scenario_name": scenario_name,
                    "risk_factor": risk_factor,
                    "shock_value": _shock_value(historical_date, risk_factor),
                    "shock_unit": unit,
                }
            )

    pnl_rows: list[dict[str, object]] = []
    for cob_date in COB_DATES:
        window = _historical_window(cob_date)
        for trade in _trades_on_cob(cob_date):
            for risk_factor in trade.risk_factors:
                sensitivity_value = _sensitivity_value(cob_date, trade.trade_id, risk_factor)
                for date in window:
                    historical_date = date.strftime("%Y-%m-%d")
                    scenario_name = f"Historical {historical_date}"
                    linear_pnl = sensitivity_value * _shock_value(historical_date, risk_factor)
                    residual = _residual_pnl(cob_date, trade.trade_id, risk_factor, historical_date, linear_pnl)
                    pnl_rows.append(
                        {
                            "cob_date": cob_date,
                            "trade_id": trade.trade_id,
                            "desk": trade.desk,
                            "product": trade.product,
                            "risk_factor": risk_factor,
                            "historical_date": historical_date,
                            "scenario_name": scenario_name,
                            "pnl": round(linear_pnl + residual, 4),
                        }
                    )

    frames = {
        "trade_sensitivities": pd.DataFrame(trade_rows),
        "risk_factor_scenarios": pd.DataFrame(scenario_rows),
        "trade_pnl": pd.DataFrame(pnl_rows),
    }

    for name, frame in frames.items():
        frame.to_csv(data_dir / f"{name}.csv", index=False)

    return frames


def ensure_sample_data(data_dir: Path = DATA_DIR) -> None:
    expected = {"trade_sensitivities.csv", "risk_factor_scenarios.csv", "trade_pnl.csv"}
    if not expected.issubset({path.name for path in data_dir.glob("*.csv")}):
        generate_sample_data(data_dir)


if __name__ == "__main__":
    generate_sample_data()
