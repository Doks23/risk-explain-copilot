from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class PortfolioConfig:
    desk: str
    book: str
    portfolio: str
    product: str
    currency: str
    risk_factors: tuple[str, ...]


PORTFOLIOS: tuple[PortfolioConfig, ...] = (
    PortfolioConfig("London Rates", "GBP Swaps", "GBP Linear", "Interest Rate Swap", "GBP", ("GBP 2Y", "GBP 5Y", "GBP 10Y", "GBP Swaption Vol")),
    PortfolioConfig("London Rates", "EUR Swaps", "EUR Linear", "Interest Rate Swap", "EUR", ("EUR 2Y", "EUR 5Y", "EUR 10Y", "EUR Swaption Vol")),
    PortfolioConfig("London Rates", "Rates Options", "GBP Vol", "Swaption", "GBP", ("GBP 5Y", "GBP 10Y", "GBP Swaption Vol")),
    PortfolioConfig("FX Options", "G10 FX Options", "EURUSD Options", "FX Option", "USD", ("EURUSD", "EURUSD Vol", "USD 10Y")),
    PortfolioConfig("FX Options", "GBP FX Options", "GBPUSD Options", "FX Option", "USD", ("GBPUSD", "GBPUSD Vol", "GBP 10Y")),
    PortfolioConfig("FX Options", "EM FX Options", "EM Asia Options", "FX Option", "USD", ("USDJPY", "USDCNH", "USDINR")),
    PortfolioConfig("Credit Trading", "IG Credit", "IG Index", "Credit Index Swap", "USD", ("EUR Credit Spread", "USD Credit Spread", "USD 10Y")),
    PortfolioConfig("Credit Trading", "HY Credit", "HY Index", "Credit Index Swap", "USD", ("HY Credit Spread", "Energy Credit Spread", "USD 10Y")),
    PortfolioConfig("Credit Trading", "Credit Options", "Credit Vol", "Credit Option", "USD", ("EUR Credit Spread", "Credit Index Vol", "USD Credit Spread")),
    PortfolioConfig("Equity Derivatives", "Index Options", "US Index Options", "Equity Option", "USD", ("SPX", "SPX Vol", "USD 10Y")),
    PortfolioConfig("Equity Derivatives", "UK Index Options", "UK Index Options", "Equity Option", "GBP", ("FTSE", "FTSE Vol", "GBP 10Y")),
    PortfolioConfig("Equity Derivatives", "Single Stock Options", "Global Single Names", "Equity Option", "USD", ("AAPL", "MSFT", "Single Name Vol")),
    PortfolioConfig("Treasury", "Liquidity Buffer", "HQLA Bonds", "Government Bond", "USD", ("USD 2Y", "USD 10Y", "Funding Spread")),
    PortfolioConfig("Treasury", "Funding", "Term Funding", "Funding Swap", "USD", ("Funding Spread", "USD 5Y", "USD 10Y")),
    PortfolioConfig("Treasury", "Hedges", "Macro Hedges", "Macro Hedge", "USD", ("SPX", "EURUSD", "USD 10Y")),
)

SCENARIOS: tuple[str, ...] = (
    "Rates Up",
    "Rates Down",
    "Curve Steepener",
    "Credit Widening",
    "Credit Tightening",
    "USD Rally",
    "Equity Selloff",
    "Equity Rally",
    "Vol Spike",
    "Liquidity Squeeze",
)

MOVE_UNITS: dict[str, str] = {
    "DV01": "bp",
    "CS01": "bp",
    "Delta": "pct",
    "Vega": "vol_pt",
}


def _business_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2026-06-15", periods=15)


def _all_risk_factors() -> tuple[str, ...]:
    return tuple(sorted({risk_factor for portfolio in PORTFOLIOS for risk_factor in portfolio.risk_factors}))


def _risk_factor_index(risk_factor: str) -> int:
    return sum(ord(char) for char in risk_factor)


def _sensitivity_type(risk_factor: str) -> str:
    if any(token in risk_factor for token in ("2Y", "5Y", "10Y")):
        return "DV01"
    if "Credit Spread" in risk_factor or "Funding Spread" in risk_factor:
        return "CS01"
    if "Vol" in risk_factor:
        return "Vega"
    return "Delta"


def _market_level(date_idx: int, risk_factor: str) -> float:
    base = 100.0 + (_risk_factor_index(risk_factor) % 40)
    drift = date_idx * (0.15 + (_risk_factor_index(risk_factor) % 5) * 0.02)
    return round(base + drift, 4)


def _market_move(date_idx: int, risk_factor: str) -> float:
    sensitivity_type = _sensitivity_type(risk_factor)
    scale = {"DV01": 1.0, "CS01": 2.0, "Vega": 0.45, "Delta": 0.35}[sensitivity_type]
    direction = ((_risk_factor_index(risk_factor) + date_idx * 3) % 9) - 4
    event = 0.0
    if date_idx == 8 and risk_factor in {"GBP 10Y", "EUR Credit Spread", "SPX", "GBP Swaption Vol"}:
        event = 3.0
    if date_idx == 12 and risk_factor in {"EURUSD", "GBPUSD", "Funding Spread", "HY Credit Spread"}:
        event = -2.5
    return round((direction * scale) + event, 4)


def _scenario_shock(date_idx: int, scenario: str, risk_factor: str) -> tuple[float, str]:
    sensitivity_type = _sensitivity_type(risk_factor)
    unit = MOVE_UNITS[sensitivity_type]
    base = {
        "Rates Up": 10.0 if sensitivity_type == "DV01" else 1.5,
        "Rates Down": -9.0 if sensitivity_type == "DV01" else -1.0,
        "Curve Steepener": 7.0 if any(token in risk_factor for token in ("10Y", "5Y")) else -3.0,
        "Credit Widening": 15.0 if sensitivity_type == "CS01" else 1.0,
        "Credit Tightening": -10.0 if sensitivity_type == "CS01" else -0.8,
        "USD Rally": -1.5 if risk_factor in {"EURUSD", "GBPUSD"} else 0.8,
        "Equity Selloff": -2.0 if risk_factor in {"SPX", "FTSE", "AAPL", "MSFT"} else 0.6,
        "Equity Rally": 1.8 if risk_factor in {"SPX", "FTSE", "AAPL", "MSFT"} else 0.5,
        "Vol Spike": 4.0 if sensitivity_type == "Vega" else 0.7,
        "Liquidity Squeeze": 12.0 if risk_factor == "Funding Spread" else 1.2,
    }[scenario]
    time_variation = ((date_idx % 4) - 1.5) * 0.2
    return round(base + time_variation, 4), unit


def _var_multiplier(scenario: str, risk_factor: str) -> float:
    sensitivity_type = _sensitivity_type(risk_factor)
    if scenario in {"Rates Up", "Rates Down", "Curve Steepener"} and sensitivity_type == "DV01":
        return 1.35
    if scenario in {"Credit Widening", "Credit Tightening"} and sensitivity_type == "CS01":
        return 1.45
    if scenario in {"Vol Spike"} and sensitivity_type == "Vega":
        return 1.5
    if scenario in {"Equity Selloff", "Equity Rally"} and sensitivity_type == "Delta":
        return 1.25
    if scenario == "Liquidity Squeeze" and risk_factor == "Funding Spread":
        return 1.6
    return 0.65


def generate_sample_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)
    dates = _business_dates()
    risk_factors = _all_risk_factors()

    hierarchy_rows: list[dict[str, object]] = []
    pnl_rows: list[dict[str, object]] = []
    var_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    market_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    pnl_level: dict[tuple[str, str, str, str], float] = {}
    prior_sensitivities: dict[tuple[str, str, str, str, str], float] = {}

    for date_idx, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")

        for risk_factor in risk_factors:
            sensitivity_type = _sensitivity_type(risk_factor)
            market_rows.append(
                {
                    "date": date_str,
                    "risk_factor": risk_factor,
                    "market_level": _market_level(date_idx, risk_factor),
                    "market_move": _market_move(date_idx, risk_factor),
                    "move_unit": MOVE_UNITS[sensitivity_type],
                }
            )
            for scenario in SCENARIOS:
                shock_value, shock_unit = _scenario_shock(date_idx, scenario, risk_factor)
                scenario_rows.append(
                    {
                        "date": date_str,
                        "scenario": scenario,
                        "risk_factor": risk_factor,
                        "shock_value": shock_value,
                        "shock_unit": shock_unit,
                    }
                )

        for portfolio_idx, portfolio in enumerate(PORTFOLIOS, start=1):
            hierarchy_rows.append(
                {
                    "date": date_str,
                    "desk": portfolio.desk,
                    "book": portfolio.book,
                    "portfolio": portfolio.portfolio,
                    "product": portfolio.product,
                    "currency": portfolio.currency,
                }
            )
            portfolio_key = (portfolio.desk, portfolio.book, portfolio.portfolio, portfolio.product)
            if portfolio_key not in pnl_level:
                pnl_level[portfolio_key] = 1_000_000 + portfolio_idx * 75_000

            daily_explained = 0.0
            for risk_idx, risk_factor in enumerate(portfolio.risk_factors, start=1):
                sensitivity_type = _sensitivity_type(risk_factor)
                raw_sensitivity = 8_000 + portfolio_idx * 1_200 + risk_idx * 650 + date_idx * 95
                if sensitivity_type in {"DV01", "CS01"}:
                    sensitivity_value = round(raw_sensitivity, 2)
                elif sensitivity_type == "Vega":
                    sensitivity_value = round(raw_sensitivity * 0.65, 2)
                else:
                    sensitivity_value = round(raw_sensitivity * 0.9, 2)

                sensitivity_rows.append(
                    {
                        "date": date_str,
                        "desk": portfolio.desk,
                        "book": portfolio.book,
                        "portfolio": portfolio.portfolio,
                        "product": portfolio.product,
                        "risk_factor": risk_factor,
                        "sensitivity_type": sensitivity_type,
                        "sensitivity_value": sensitivity_value,
                    }
                )

                prev_sensitivity = prior_sensitivities.get((*portfolio_key, risk_factor), sensitivity_value)
                move = _market_move(date_idx, risk_factor)
                if date_idx > 0:
                    daily_explained += prev_sensitivity * move
                prior_sensitivities[(*portfolio_key, risk_factor)] = sensitivity_value

                for scenario_idx, scenario in enumerate(SCENARIOS, start=1):
                    shock_value, _ = _scenario_shock(date_idx, scenario, risk_factor)
                    position_effect = 1.0 + date_idx * 0.015 + portfolio_idx * 0.025
                    var_contribution = abs(sensitivity_value * shock_value * _var_multiplier(scenario, risk_factor) * position_effect / 1_000)
                    if date_idx == len(dates) - 1 and risk_factor in {"GBP 10Y", "EUR Credit Spread", "SPX", "Funding Spread"}:
                        var_contribution *= 1.2
                    var_rows.append(
                        {
                            "date": date_str,
                            "desk": portfolio.desk,
                            "book": portfolio.book,
                            "portfolio": portfolio.portfolio,
                            "scenario": scenario,
                            "risk_factor": risk_factor,
                            "product": portfolio.product,
                            "var_contribution": round(var_contribution + scenario_idx * 0.15, 4),
                        }
                    )

            residual = (portfolio_idx - 8) * 2_500 + ((date_idx % 5) - 2) * 1_250
            if date_idx > 0:
                pnl_level[portfolio_key] += daily_explained + residual
            pnl_rows.append(
                {
                    "date": date_str,
                    "desk": portfolio.desk,
                    "book": portfolio.book,
                    "portfolio": portfolio.portfolio,
                    "product": portfolio.product,
                    "pnl_value": round(pnl_level[portfolio_key], 2),
                }
            )

    frames = {
        "hierarchy": pd.DataFrame(hierarchy_rows).drop_duplicates(),
        "pnl_results": pd.DataFrame(pnl_rows),
        "var_results": pd.DataFrame(var_rows),
        "sensitivities": pd.DataFrame(sensitivity_rows),
        "market_data": pd.DataFrame(market_rows),
        "scenario_data": pd.DataFrame(scenario_rows),
    }

    for name, frame in frames.items():
        frame.to_csv(data_dir / f"{name}.csv", index=False)

    return frames


def ensure_sample_data(data_dir: Path = DATA_DIR) -> None:
    expected = {
        "hierarchy.csv",
        "pnl_results.csv",
        "var_results.csv",
        "sensitivities.csv",
        "market_data.csv",
        "scenario_data.csv",
    }
    if not expected.issubset({path.name for path in data_dir.glob("*.csv")}):
        generate_sample_data(data_dir)


if __name__ == "__main__":
    generate_sample_data()
