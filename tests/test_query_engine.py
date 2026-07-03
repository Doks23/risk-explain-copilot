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
        validate_sql("DELETE FROM var_results")
    with pytest.raises(ValueError):
        validate_sql("SELECT * FROM var_results")
    with pytest.raises(ValueError):
        validate_sql("SELECT date FROM var_results; SELECT date FROM pnl_results")


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


def test_var_movement_uses_var_contribution_not_pnl(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    plan = generate_query_plan("Why did VAR move for London Rates?", db_path=db_path)
    result = execute_query_plan(plan, db_path=db_path)
    answer = generate_response(plan.question, plan, result)

    assert plan.intent == "var_movement"
    assert plan.metric == "VAR"
    assert not plan.llm_used
    assert len(result) == 1
    assert {"var_date_1", "var_date_2", "var_change", "percentage_change"}.issubset(result.columns)
    assert "var_contribution" in plan.sql
    assert "pnl_value" not in plan.sql
    assert "Data trace" not in answer


def test_pnl_movement_returns_explained_and_residual_pnl(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    plan = generate_query_plan("Explain PNL movement for London Rates", db_path=db_path)
    result = execute_query_plan(plan, db_path=db_path)

    assert plan.intent == "pnl_movement"
    assert plan.metric == "PNL"
    assert len(result) == 1
    assert {"pnl_date_1", "pnl_date_2", "pnl_change", "explained_pnl", "residual_pnl"}.issubset(result.columns)
    assert "sensitivity_value" in plan.sql
    assert "market_move" in plan.sql


def test_driver_and_trend_queries_use_correct_business_logic(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    scenario_plan = generate_query_plan("What are the top 5 scenario drivers for VAR?", db_path=db_path)
    pnl_driver_plan = generate_query_plan("What market moves explain the PNL change?", db_path=db_path)
    trend_plan = generate_query_plan("Show 10-day VAR trend for FX Options", db_path=db_path)

    scenario_driver = execute_query_plan(scenario_plan, db_path=db_path)
    pnl_driver = execute_query_plan(pnl_driver_plan, db_path=db_path)
    trend = execute_query_plan(trend_plan, db_path=db_path)

    assert len(scenario_driver) == 5
    assert {"scenario", "delta"}.issubset(scenario_driver.columns)
    assert "var_contribution" in scenario_plan.sql
    assert len(pnl_driver) == 5
    assert {"risk_factor", "sensitivity_value", "market_move", "estimated_pnl_impact"}.issubset(pnl_driver.columns)
    assert "sensitivity_value" in pnl_driver_plan.sql
    assert len(trend) == 10
    assert {"date", "value"}.issubset(trend.columns)
    assert "desk = 'FX Options'" in trend_plan.sql


def test_fallback_trend_response_is_not_driver_framed(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    plan = generate_query_plan("Show 10-day VAR trend for FX Options", db_path=db_path)
    result = execute_query_plan(plan, db_path=db_path)
    answer = generate_response(plan.question, plan, result)

    assert plan.intent == "trend"
    assert "The returned trend" in answer
    assert "### Evidence from SQL" not in answer
    assert "### Drivers" not in answer


def test_query_plan_handles_desk_coverage_question(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    plan = generate_query_plan("What desks are we covering?", db_path=db_path)
    result = execute_query_plan(plan, db_path=db_path)
    answer = generate_response(plan.question, plan, result)

    assert plan.intent == "desk_coverage"
    assert plan.metric == "METADATA"
    assert plan.retrieved_context
    assert {"desk", "book_count", "portfolio_count", "books", "products", "currencies"}.issubset(result.columns)
    assert "We cover 5 desks" in answer
    assert "London Rates" in answer


def test_follow_up_query_uses_prior_scope_for_sql(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    context = "User: Show 10-day VAR trend for FX Options\nAssistant: intent=trend; metric=VAR; visualization=line"

    plan = generate_query_plan("same for PNL", db_path=db_path, conversation_context=context)
    result = execute_query_plan(plan, db_path=db_path)

    assert plan.intent == "trend"
    assert plan.metric == "PNL"
    assert "pnl_results" in plan.sql
    assert "desk = 'FX Options'" in plan.sql
    assert len(result) == 10


def test_follow_up_query_uses_prior_desk_for_movement(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    context = "User: Why did VAR move for London Rates?\nAssistant: intent=var_movement; metric=VAR; visualization=metric"

    plan = generate_query_plan("what about PNL?", db_path=db_path, conversation_context=context)
    result = execute_query_plan(plan, db_path=db_path)

    assert plan.intent == "pnl_movement"
    assert plan.metric == "PNL"
    assert "pnl_results" in plan.sql
    assert "desk = 'London Rates'" in plan.sql
    assert len(result) == 1
