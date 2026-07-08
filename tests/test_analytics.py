from __future__ import annotations

from pathlib import Path

import pytest

from src.analytics import aggregate_scenario_pnl, calculate_var, coverage, explain_trade_pnl, explain_var_by_risk_factor, trend_analysis
from src.data_generator import generate_sample_data
from src.db import load_csvs_to_sqlite


def _db(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    db_path = tmp_path / "risk.db"
    generate_sample_data(data_dir)
    load_csvs_to_sqlite(data_dir=data_dir, db_path=db_path, reset=True)
    return db_path


def test_trade_pnl_uses_sensitivity_times_shock(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    result = explain_trade_pnl(desk="D1", db_path=db_path)
    row = result.iloc[0]

    assert not result.empty
    assert row["scenario_pnl"] == pytest.approx(row["sensitivity_value"] * row["shock_value"], abs=0.01)


def test_aggregate_scenario_pnl_returns_distribution(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    result = aggregate_scenario_pnl(desk="D1", db_path=db_path)

    assert len(result) == 50
    assert {"historical_date", "scenario_pnl", "loss_amount"}.issubset(result.columns)


def test_var_is_percentile_loss_from_aggregated_distribution(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    var = calculate_var(desk="D1", db_path=db_path)
    row = var.iloc[0]

    assert {"confidence_level", "historical_date", "scenario_pnl", "var_95", "scenario_count"}.issubset(var.columns)
    assert row["confidence_level"] == pytest.approx(0.95)
    assert row["scenario_count"] == 50
    assert row["var_95"] >= 0


def test_var_drivers_return_risk_factor_contributions_for_var_scenario(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    drivers = explain_var_by_risk_factor(desk="D1", db_path=db_path)

    assert not drivers.empty
    assert {"risk_factor", "sensitivity_type", "shock_value", "driver_pnl"}.issubset(drivers.columns)


def test_trend_and_coverage(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    trend = trend_analysis("PNL", desk="D1", days=10, db_path=db_path)
    covered = coverage(db_path=db_path)

    assert len(trend) == 10
    assert {"historical_date", "value"}.issubset(trend.columns)
    assert {"desk", "trade_count", "product_count", "risk_factor_count"}.issubset(covered.columns)
