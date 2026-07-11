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


def test_var_query_defaults_to_latest_cob_with_full_100_day_window(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    plan = generate_query_plan("Calculate 95% VaR for D1", db_path=db_path)
    result = execute_query_plan(plan, db_path=db_path)

    assert "cob_date" in plan.sql
    # Not 104 (the union of all 5 COBs' overlapping windows) -- exactly one COB's own window.
    assert result.iloc[0]["scenario_count"] == 100


def test_var_query_for_an_explicit_cob_scopes_to_that_cob_only(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    latest_plan = generate_query_plan("Calculate 95% VaR for D1", db_path=db_path)
    latest_result = execute_query_plan(latest_plan, db_path=db_path)

    earlier_plan = generate_query_plan("Calculate 95% VaR for D1 as of COB 2026-07-08", db_path=db_path)
    earlier_result = execute_query_plan(earlier_plan, db_path=db_path)

    assert "2026-07-08" in earlier_plan.sql
    assert earlier_result.iloc[0]["scenario_count"] == 100
    # Different COBs have different books/sensitivities, so the two VaR figures should differ.
    assert earlier_result.iloc[0]["var_95"] != latest_result.iloc[0]["var_95"]


def test_trade_pnl_query_returns_actual_stored_pnl(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    plan = generate_query_plan("Show trade level PNL for D1", db_path=db_path)
    result = execute_query_plan(plan, db_path=db_path)

    assert plan.intent == "trade_pnl"
    assert {"trade_id", "desk", "product", "historical_date", "pnl"}.issubset(result.columns)
    assert "FROM trade_pnl" in plan.sql


def test_var_is_literally_the_5th_worst_day_of_100(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    plan = generate_query_plan("Calculate 95% VaR for D1", db_path=db_path)
    result = execute_query_plan(plan, db_path=db_path)

    from src.db import get_connection

    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT historical_date, SUM(pnl) AS pnl FROM trade_pnl "
            "WHERE desk = 'D1' AND cob_date = (SELECT MAX(cob_date) FROM trade_sensitivities) "
            "GROUP BY historical_date"
        ).fetchall()
    losses = sorted((-r["pnl"] if r["pnl"] < 0 else 0.0) for r in rows)
    fifth_worst = losses[-5]  # 5th worst = 5th from the top of a 100-day window

    assert result.iloc[0]["var_95"] == pytest.approx(fifth_worst, abs=0.01)


@pytest.mark.parametrize("phrasing", ["What is expected shortfall for D1", "Show CVaR for D1", "What is the tail loss for D1", "conditional VaR for D1"])
def test_expected_shortfall_is_mean_of_the_5_worst_days(phrasing: str, tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    plan = generate_query_plan(phrasing, db_path=db_path)
    result = execute_query_plan(plan, db_path=db_path)

    from src.db import get_connection

    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT historical_date, SUM(pnl) AS pnl FROM trade_pnl "
            "WHERE desk = 'D1' AND cob_date = (SELECT MAX(cob_date) FROM trade_sensitivities) "
            "GROUP BY historical_date"
        ).fetchall()
    losses = sorted((-r["pnl"] if r["pnl"] < 0 else 0.0) for r in rows)
    worst_5 = losses[-5:]

    assert plan.intent == "expected_shortfall"
    row = result.iloc[0]
    assert row["tail_days"] == 5
    assert row["es_95"] == pytest.approx(sum(worst_5) / 5, abs=0.01)
    assert row["var_95"] == pytest.approx(worst_5[0], abs=0.01)
    # ES sits at or beyond VaR, since VaR is the least-severe day in the same tail.
    assert row["es_95"] >= row["var_95"]


def test_var_risk_factor_breakdown_is_exact_and_reconciles_with_no_residual(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    var_plan = generate_query_plan("Calculate 95% VaR for D1", db_path=db_path)
    var_result = execute_query_plan(var_plan, db_path=db_path)

    driver_plan = generate_query_plan("Explain VaR risk factor drivers for D1", db_path=db_path)
    drivers = execute_query_plan(driver_plan, db_path=db_path)

    assert driver_plan.intent == "var_risk_factor_drivers"
    assert {"risk_factor", "driver_pnl", "var_scenario_date"}.issubset(drivers.columns)
    assert "Unexplained (non-linear residual)" not in drivers["risk_factor"].values
    # Exact, straight from trade_pnl -- no approximation, so it reconciles perfectly, not "closely".
    assert drivers["driver_pnl"].sum() == pytest.approx(var_result.iloc[0]["scenario_pnl"], abs=0.01)


def test_entity_var_explained_by_desk_is_additive(tmp_path: Path) -> None:
    db_path = _db(tmp_path)

    entity_plan = generate_query_plan("Calculate 95% VaR", db_path=db_path)
    entity_result = execute_query_plan(entity_plan, db_path=db_path)

    breakdown_plan = generate_query_plan("Explain entity VaR by desk", db_path=db_path)
    breakdown = execute_query_plan(breakdown_plan, db_path=db_path)

    assert breakdown_plan.intent == "var_desk_drivers"
    assert {"desk", "driver_pnl", "var_scenario_date"}.issubset(breakdown.columns)
    assert breakdown.iloc[0]["var_scenario_date"] == entity_result.iloc[0]["historical_date"]
    # Same historical date, just split by desk -- unlike desk-level VaR, this MUST sum exactly.
    assert breakdown["driver_pnl"].sum() == pytest.approx(entity_result.iloc[0]["scenario_pnl"], abs=0.01)


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


def test_conversation_context_sql_does_not_falsely_match_trade_id(tmp_path: Path) -> None:
    """Regression test: prior turns' generated SQL is embedded in conversation_context, and SQL syntax
    can accidentally spell out a trade_id-like substring once whitespace/punctuation is stripped
    (e.g. "...AS INT) + 1..." compacts to "...asint1...", falsely matching trade_id "T1"). The fuzzy
    compact-match pass must never run against conversation context, only the current question."""
    db_path = _db(tmp_path)
    entity_plan = generate_query_plan("What VAR for entity", db_path=db_path)
    context = f"User: What VAR for entity\nAssistant: intent={entity_plan.intent}; sql={entity_plan.sql}"

    follow_up = generate_query_plan("Explain the contribution by Desk", db_path=db_path, conversation_context=context)

    assert "trade_id" not in follow_up.sql
    assert follow_up.intent == "var_desk_drivers"


def test_bare_trend_question_inherits_var_from_prior_turn_not_pnl(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    entity_plan = generate_query_plan("What VAR for entity", db_path=db_path)
    context = f"User: What VAR for entity\nAssistant: intent={entity_plan.intent}; metric={entity_plan.metric}; sql={entity_plan.sql}"

    trend_plan = generate_query_plan("show trend", db_path=db_path, conversation_context=context)

    assert trend_plan.metric == "VAR"

    override_plan = generate_query_plan("show pnl trend", db_path=db_path, conversation_context=context)
    assert override_plan.metric == "PNL"
