from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.analytics import (
    compare_pnl,
    compare_var,
    drilldown,
    explain_pnl_by_risk_factor,
    explain_var_by_risk_factor,
    explain_var_by_scenario,
    trend_analysis,
)
from src.data_generator import generate_sample_data
from src.db import load_csvs_to_sqlite


def _db(tmp_path: Path) -> tuple[Path, str, str]:
    data_dir = tmp_path / "data"
    db_path = tmp_path / "risk.db"
    generate_sample_data(data_dir)
    load_csvs_to_sqlite(data_dir=data_dir, db_path=db_path, reset=True)
    with (data_dir / "hierarchy.csv").open() as handle:
        dates = sorted({row["date"] for row in csv.DictReader(handle)})
    return db_path, dates[-2], dates[-1]


def test_compare_pnl_reconciles_explained_and_residual(tmp_path: Path) -> None:
    db_path, date1, date2 = _db(tmp_path)

    result = compare_pnl(date1, date2, desk="London Rates", db_path=db_path)
    row = result.iloc[0]

    assert {"pnl_change", "explained_pnl", "residual_pnl"}.issubset(result.columns)
    assert row["pnl_change"] == pytest.approx(row["explained_pnl"] + row["residual_pnl"], abs=0.02)


def test_explain_pnl_by_risk_factor_uses_sensitivity_times_market_move(tmp_path: Path) -> None:
    db_path, date1, date2 = _db(tmp_path)

    result = explain_pnl_by_risk_factor(date1, date2, desk="London Rates", db_path=db_path)
    row = result.iloc[0]

    assert not result.empty
    assert row["estimated_pnl_impact"] == pytest.approx(row["sensitivity_value"] * row["market_move"], abs=0.02)


def test_var_functions_return_scenario_and_risk_factor_contribution_changes(tmp_path: Path) -> None:
    db_path, date1, date2 = _db(tmp_path)

    movement = compare_var(date1, date2, desk="London Rates", db_path=db_path)
    scenarios = explain_var_by_scenario(date1, date2, desk="London Rates", db_path=db_path)
    risk_factors = explain_var_by_risk_factor(date1, date2, desk="London Rates", db_path=db_path)

    assert {"var_date_1", "var_date_2", "var_change", "percentage_change"}.issubset(movement.columns)
    assert {"scenario", "previous_value", "current_value", "delta"}.issubset(scenarios.columns)
    assert {"risk_factor", "previous_value", "current_value", "delta"}.issubset(risk_factors.columns)


def test_trend_and_drilldown_return_expected_grain(tmp_path: Path) -> None:
    db_path, date1, date2 = _db(tmp_path)

    trend = trend_analysis("VAR", desk="FX Options", days=10, db_path=db_path)
    drill = drilldown("VAR", date1, date2, desk="London Rates", db_path=db_path)

    assert len(trend) == 10
    assert {"date", "value"}.issubset(trend.columns)
    assert {"desk", "book", "portfolio", "product", "scenario", "risk_factor", "delta"}.issubset(drill.columns)
