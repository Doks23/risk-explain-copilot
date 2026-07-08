from __future__ import annotations

from pathlib import Path

import pytest

from src.data_generator import generate_sample_data
from src.db import load_csvs_to_sqlite
from src.query_engine import execute_query_plan, generate_query_plan, generate_response, get_llm_config, validate_sql


def _db(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    db_path = tmp_path / "risk.db"
    generate_sample_data(data_dir)
    load_csvs_to_sqlite(data_dir=data_dir, db_path=db_path, reset=True)
    return db_path


def test_validate_sql_rejects_unsafe_or_full_table_patterns() -> None:
    with pytest.raises(ValueError):
        validate_sql("DELETE FROM trade_sensitivities")
    with pytest.raises(ValueError):
        validate_sql("SELECT * FROM trade_sensitivities")
    with pytest.raises(ValueError):
        validate_sql("SELECT trade_id FROM trade_sensitivities; SELECT risk_factor FROM risk_factor_scenarios")


def test_llm_config_can_use_gemini_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")

    config = get_llm_config()

    assert config is not None
    assert config.provider == "Gemini"
    assert config.api_key == "test-gemini-key"
    assert config.model == "gemini-test-model"
    assert config.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_var_query_computes_percentile_loss_from_stored_trade_pnl(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    plan = generate_query_plan("Calculate 95% VaR for D1", db_path=db_path)
    result = execute_query_plan(plan, db_path=db_path)
    answer = generate_response(plan.question, plan, result)

    assert plan.intent == "var"
    assert plan.metric == "VAR"
    assert {"confidence_level", "historical_date", "scenario_pnl", "var_95", "scenario_count"}.issubset(result.columns)
    assert "trade_pnl" in plan.sql
    assert "Data trace" not in answer


def test_trade_pnl_query_returns_actual_stored_pnl(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    plan = generate_query_plan("Show trade level PNL for D1", db_path=db_path)
    result = execute_query_plan(plan, db_path=db_path)

    assert plan.intent == "trade_pnl"
    assert {"trade_id", "desk", "product", "historical_date", "pnl"}.issubset(result.columns)
    assert "FROM trade_pnl" in plan.sql


def test_var_driver_query_attributes_by_risk_factor_with_residual(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    driver_plan = generate_query_plan("Explain VaR risk factor drivers for D1", db_path=db_path)
    drivers = execute_query_plan(driver_plan, db_path=db_path)

    assert driver_plan.intent == "var_risk_factor_drivers"
    assert {"risk_factor", "driver_pnl", "var_scenario_date"}.issubset(drivers.columns)
    assert "trade_sensitivities" in driver_plan.sql
    assert "risk_factor_scenarios" in driver_plan.sql


def test_trend_query_uses_stored_trade_pnl(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    trend_plan = generate_query_plan("Show 10-day PNL trend for D2", db_path=db_path)
    trend = execute_query_plan(trend_plan, db_path=db_path)

    assert trend_plan.intent == "trend"
    assert {"date", "value"}.issubset(trend.columns)
    assert len(trend) == 10
    assert "trade_pnl" in trend_plan.sql


def test_coverage_question_uses_trade_table(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    plan = generate_query_plan("What desks are we covering?", db_path=db_path)
    result = execute_query_plan(plan, db_path=db_path)
    answer = generate_response(plan.question, plan, result)

    assert plan.intent == "coverage"
    assert plan.metric == "METADATA"
    assert {"desk", "trade_count", "product_count", "risk_factor_count", "products"}.issubset(result.columns)
    assert "We cover" in answer
    assert "D1" in answer


def test_follow_up_query_uses_prior_desk_scope(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    context = "User: Calculate 95% VaR for D1\nAssistant: intent=var; metric=VAR; visualization=metric"

    plan = generate_query_plan("show risk factor drivers", db_path=db_path, conversation_context=context)
    result = execute_query_plan(plan, db_path=db_path)

    assert plan.intent == "var_risk_factor_drivers"
    assert "ts.desk = 'D1'" in plan.sql
    assert not result.empty
