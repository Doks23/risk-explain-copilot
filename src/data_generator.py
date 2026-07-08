from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class TradeConfig:
    trade_id: str
    desk: str
    product: str
    risk_factors: tuple[str, ...]


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

TRADES: tuple[TradeConfig, ...] = (
    TradeConfig("T1", "D1", "IR Swap", ("SOFR", "USD GBP")),
    TradeConfig("T2", "D1", "IR Swap", ("SONIA", "USD GBP")),
    TradeConfig("T3", "D1", "FX Swap", ("ESTER", "USD EUR")),
    TradeConfig("T4", "D1", "Equity Option", ("Vodafone Spot", "USD GBP", "SOFR")),
    TradeConfig("T5", "D1", "FX Forward", ("USD JPY",)),
    TradeConfig("T6", "D2", "FX Forward", ("USD GBP",)),
    TradeConfig("T7", "D2", "FX Swap", ("USD EUR", "ESTER")),
    TradeConfig("T8", "D2", "Equity Total Return Swap", ("HSBC Spot", "USD GBP", "ESTER")),
    TradeConfig("T9", "D2", "FX Forward", ("USD EUR",)),
    TradeConfig("T10", "D2", "IR Forward", ("ESTER", "USD EUR")),
    TradeConfig("T11", "D3", "Cash Equity", ("Vodafone Spot",)),
    TradeConfig("T12", "D3", "Equity Total Return Swap", ("Vodafone Spot", "USD GBP", "SONIA")),
    TradeConfig("T13", "D3", "FX Forward", ("USD JPY",)),
    TradeConfig("T14", "D3", "IR Swap", ("SONIA", "USD GBP")),
    TradeConfig("T15", "D3", "Cash Equity", ("HSBC Spot",)),
    TradeConfig("T16", "D4", "Rates Option", ("SOFR", "GBP Swaption Vol")),
    TradeConfig("T17", "D4", "Credit Index Swap", ("EUR Credit Spread", "USD EUR")),
    TradeConfig("T18", "D4", "Equity Option", ("SPX", "USD GBP")),
    TradeConfig("T19", "D5", "Commodity Forward", ("Brent", "USD EUR")),
    TradeConfig("T20", "D5", "Macro Hedge", ("SOFR", "SPX", "USD JPY")),
)


def _historical_dates() -> pd.DatetimeIndex:
    return pd.date_range("2025-07-01", periods=50, freq="D")


def _stable_index(value: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(value))


def _sensitivity_value(trade_id: str, risk_factor: str) -> float:
    seed = _stable_index(f"{trade_id}-{risk_factor}")
    magnitude = 1 + (seed % 95)
    sign = -1 if seed % 3 == 0 else 1
    return float(sign * magnitude)


def _shock_value(date_idx: int, risk_factor: str) -> float:
    sensitivity_type, unit = RISK_FACTOR_TYPES[risk_factor]
    seed = _stable_index(risk_factor) + date_idx * 17
    direction = (seed % 21) - 10
    if unit == "bps":
        return float(direction * 8 + ((date_idx % 5) - 2) * 3)
    if unit == "vol_pt":
        return round(direction * 0.35, 4)
    return round(float(direction), 4)


def generate_sample_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)

    trade_rows: list[dict[str, object]] = []
    for trade in TRADES:
        for risk_factor in trade.risk_factors:
            sensitivity_type, _ = RISK_FACTOR_TYPES[risk_factor]
            trade_rows.append(
                {
                    "trade_id": trade.trade_id,
                    "risk_factor": risk_factor,
                    "desk": trade.desk,
                    "sensitivity_type": sensitivity_type,
                    "sensitivity_value": _sensitivity_value(trade.trade_id, risk_factor),
                    "product": trade.product,
                }
            )

    scenario_rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(_historical_dates()):
        historical_date = date.strftime("%Y-%m-%d")
        scenario_name = f"Historical {historical_date}"
        for risk_factor, (_, unit) in RISK_FACTOR_TYPES.items():
            scenario_rows.append(
                {
                    "historical_date": historical_date,
                    "scenario_name": scenario_name,
                    "risk_factor": risk_factor,
                    "shock_value": _shock_value(date_idx, risk_factor),
                    "shock_unit": unit,
                }
            )

    frames = {
        "trade_sensitivities": pd.DataFrame(trade_rows),
        "risk_factor_scenarios": pd.DataFrame(scenario_rows),
    }

    for name, frame in frames.items():
        frame.to_csv(data_dir / f"{name}.csv", index=False)

    return frames


def ensure_sample_data(data_dir: Path = DATA_DIR) -> None:
    expected = {"trade_sensitivities.csv", "risk_factor_scenarios.csv"}
    if not expected.issubset({path.name for path in data_dir.glob("*.csv")}):
        generate_sample_data(data_dir)


if __name__ == "__main__":
    generate_sample_data()
