from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .db import DB_PATH, distinct_values, get_connection, latest_cob_date
from .document_store import retrieve_context
from .llm_config import get_llm_config


MAX_RESULT_ROWS = 50

SCHEMA_CONTEXT = """
SQLite schema:
- trade_sensitivities(cob_date, trade_id, risk_factor, desk, sensitivity_type, sensitivity_value, product)
- risk_factor_scenarios(historical_date, scenario_name, risk_factor, shock_value, shock_unit)
- trade_pnl(cob_date, trade_id, desk, product, risk_factor, historical_date, scenario_name, pnl)

Business rules:
- These are the three stored input tables. trade_pnl holds the actual trade-level P&L for every
  trade on every historical scenario date (as if from full revaluation) and is the source of truth
  for aggregation and VaR.
- cob_date is the close-of-business snapshot date. The trade book changes across COBs (trades book
  and mature), and sensitivities are re-marked every COB even for surviving trades, so the same
  trade can have a different sensitivity_value on different cob_date rows.
- Every query against trade_pnl or trade_sensitivities MUST filter to exactly one cob_date. If the
  question does not name a COB, use the single most recent cob_date present in trade_sensitivities
  (SELECT MAX(cob_date) FROM trade_sensitivities). NEVER aggregate or join across more than one
  cob_date in the same query — that silently blends different trade books together and produces a
  meaningless number.
- Each cob_date's historical scenario window is already exactly the 100 rolling days that belong to
  it in trade_pnl (ending the day before that cob_date). Filtering to one cob_date is sufficient by
  itself to get the correct 100-day window; do not additionally filter historical_date by a range.
- Portfolio/desk/entity scenario P&L = SUM(trade_pnl.pnl) grouped by historical_date for the scope
  (cob_date + desk/product/trade filters). Never derive portfolio P&L from trade_sensitivities x
  risk_factor_scenarios; use trade_pnl for that.
- 95% VaR is literally the k-th worst day, where k = ROUND(N x (1 - 0.95)) (e.g. the 5th worst day
  out of a 100-day window). Rank loss_amount = max(-scenario_pnl, 0) descending (worst first) and
  take the k-th row. This is NOT the nearest-rank/ceiling convention and NOT linear interpolation.
- VaR is NOT additive: entity-level VaR is not the sum of desk-level VaRs, because each scope's
  95th-percentile scenario date can differ. Only P&L is additive across scopes, VaR is not.
- To explain WHAT drives VaR at the risk-factor level, find the scope's own VaR historical date,
  then GROUP BY risk_factor directly on trade_pnl for that single (cob_date, historical_date, scope)
  — trade_pnl already stores real P&L per risk factor, so this is exact and always reconciles to the
  scope's total P&L on that date; no residual or approximation needed. Optionally LEFT JOIN
  trade_sensitivities/risk_factor_scenarios (same cob_date and date) purely to show the underlying
  sensitivity_value/shock_value as supporting color, but driver_pnl itself must come from trade_pnl.
- To explain entity (or any scope) VaR by DESK, take that scope's own VaR historical date, then
  GROUP BY desk directly on trade_pnl for that single (cob_date, historical_date) with no desk
  filter. Because every desk's number comes from the exact same historical date, they ARE additive
  and sum exactly to the scope's total P&L on that date — unlike desk-level VaR figures, which are
  each computed on a different desk's own worst-case date and are never additive.
- Do not treat sensitivity as P&L. Do not treat shocks as P&L.
- If the question asks to see a series over time -- "trend", "graph", "chart", "plot", "timeline",
  "visualize", "over time", "history", or "last N days" -- set visualization to "line", and pick ONE
  of these two shapes depending on which metric:
  - PNL trend: one COB's own scenario_pnl/loss_amount spread across its historical_date window (the
    existing single-COB pattern). Still filter to exactly one cob_date.
  - VAR trend: VaR is one number per COB, so a VaR trend means recomputing the 95% VaR independently
    for EACH cob_date and plotting one point per COB (NOT spreading one COB's distribution across
    historical_date, and NOT filtering to a single cob_date -- rank historical_date within each
    cob_date separately, e.g. with ROW_NUMBER() OVER (PARTITION BY cob_date ORDER BY loss_amount),
    keep only the 5th-worst row per cob_date, and return one (cob_date, var_95) row per COB).
  Do not confuse either with "movement" used to mean "explain what drove this VaR" (e.g. "Desk D1 Var
  movement"), which is a single-date breakdown, not a time series.

Conversation rules:
- If the current question does not say VAR or PNL, use whichever one the previous turn was about
  (from the conversation context). Only switch metric when the current question explicitly says the
  other one. E.g. after "What VAR for entity", a follow-up "explain the contribution by desk" stays
  VAR and reuses entity VaR's own historical date; "show trend" stays VAR too, not PNL.
- Likewise inherit scope (desk/product/trade/cob_date) from the previous turn when the current
  question doesn't name one, and only change it when the current question explicitly names a
  different one (e.g. "Explain the Desk D1 Var movement" switches scope to D1 and uses D1's own VaR
  date, even if the prior turn was about the entity).

SQL rules:
- Generate one read-only SQLite SELECT or WITH statement.
- Do not use SELECT *.
- Always aggregate or filter to the user's question.
- Always filter trade_pnl and trade_sensitivities to exactly one cob_date.
- Always include LIMIT 50 or lower.
- Do not expose full database contents.
"""

BLOCKED_SQL_TOKENS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "pragma",
    "attach",
    "detach",
    "vacuum",
    "sqlite_master",
    "sqlite_schema",
    "load_extension",
)


@dataclass(frozen=True)
class QueryPlan:
    question: str
    sql: str
    intent: str
    metric: str
    visualization: str
    llm_used: bool
    notes: str
    retrieved_context: tuple[dict[str, Any], ...] = ()
    conversation_context: str = ""


def generate_query_plan(question: str, db_path: Path = DB_PATH, conversation_context: str | None = None) -> QueryPlan:
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Please enter a question.")

    conversation_context = (conversation_context or "").strip()
    retrieval_query = f"{conversation_context}\nCurrent question: {cleaned}" if conversation_context else cleaned
    context = retrieve_context(retrieval_query)
    if get_llm_config():
        try:
            plan = _llm_query_plan(cleaned, context, conversation_context)
            validate_sql(plan.sql)
            if not _llm_sql_is_correctly_scoped(cleaned, plan.sql):
                raise ValueError("LLM SQL was not correctly scoped to trade_pnl and a single cob_date.")
            return plan
        except Exception:
            pass

    return replace(
        _fallback_query_plan(cleaned, db_path, conversation_context=conversation_context),
        retrieved_context=context,
        conversation_context=conversation_context,
    )


def _llm_sql_is_correctly_scoped(question: str, sql: str) -> bool:
    """Guard against two ways the LLM's SQL can be silently wrong even though it executes fine.

    1. Deriving portfolio P&L from sensitivity x shock instead of the real trade_pnl table.
    2. Aggregating trade_pnl/trade_sensitivities without a cob_date filter, which blends
       multiple COBs' different trade books and rolling windows into one meaningless number.

    Only VaR/P&L/trend/driver questions require either; coverage and raw shock-lookup questions
    legitimately don't touch trade_pnl at all.
    """
    lowered = question.lower()
    if _is_coverage_question(lowered) and not _mentions_pnl(lowered) and not _mentions_var(lowered):
        return True
    if "risk factor" in lowered and ("scenario" in lowered or "shock" in lowered) and not _mentions_pnl(lowered) and not _mentions_var(lowered):
        return True
    if _mentions_var(lowered) or _mentions_pnl(lowered) or "trend" in lowered or "driver" in lowered or "explain" in lowered or re.search(r"\b\d+\s*[- ]?day\b", lowered):
        sql_lower = sql.lower()
        return "trade_pnl" in sql_lower and "cob_date" in sql_lower
    return True


def generate_fallback_query_plan(question: str, db_path: Path = DB_PATH, conversation_context: str | None = None) -> QueryPlan:
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Please enter a question.")
    conversation_context = (conversation_context or "").strip()
    context = retrieve_context(f"{conversation_context}\nCurrent question: {cleaned}" if conversation_context else cleaned)
    return replace(
        _fallback_query_plan(cleaned, db_path, conversation_context=conversation_context),
        retrieved_context=context,
        conversation_context=conversation_context,
    )


def execute_query_plan(plan: QueryPlan, db_path: Path = DB_PATH, row_limit: int = MAX_RESULT_ROWS) -> pd.DataFrame:
    sql = enforce_limit(validate_sql(plan.sql), row_limit=row_limit)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def generate_response(question: str, plan: QueryPlan, result: pd.DataFrame) -> str:
    if get_llm_config():
        try:
            return _llm_response(question, plan, result)
        except Exception:
            pass
    return _fallback_response(question, plan, result)


def validate_sql(sql: str) -> str:
    normalized = sql.strip().rstrip(";").strip()
    lowered = normalized.lower()
    if not normalized:
        raise ValueError("Generated SQL is empty.")
    if not re.match(r"^(select|with)\b", lowered):
        raise ValueError("Only SELECT statements are allowed.")
    if ";" in normalized:
        raise ValueError("Only one SQL statement is allowed.")
    if re.search(r"\bselect\s+\*", lowered):
        raise ValueError("SELECT * is not allowed.")
    for token in BLOCKED_SQL_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            raise ValueError(f"SQL token is not allowed: {token}")
    return normalized


def enforce_limit(sql: str, row_limit: int = MAX_RESULT_ROWS) -> str:
    if re.search(r"\blimit\s+\d+\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql}\nLIMIT {row_limit}"


def _llm_query_plan(question: str, context: tuple[dict[str, Any], ...], conversation_context: str = "") -> QueryPlan:
    from openai import OpenAI

    config = get_llm_config()
    if not config:
        raise RuntimeError("No LLM provider configured.")
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a SQL planner for a market risk copilot. "
                    "Return strict JSON with keys: sql, intent, metric, visualization, notes. "
                    "The SQL must obey the schema, business rules, and SQL rules exactly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{SCHEMA_CONTEXT}\n\n"
                    f"Recent conversation context:\n{conversation_context or 'No prior conversation context.'}\n\n"
                    f"Retrieved business context:\n{_context_for_prompt(context)}\n\n"
                    f"Current question: {question}"
                ),
            },
        ],
        temperature=0,
    )
    data = _parse_json_object(response.choices[0].message.content or "{}")
    return QueryPlan(
        question=question,
        sql=str(data["sql"]),
        intent=str(data.get("intent", "query")),
        metric=str(data.get("metric", "UNKNOWN")).upper(),
        visualization=str(data.get("visualization", "table")),
        llm_used=True,
        notes=str(data.get("notes", f"Generated by {config.provider}.")),
        retrieved_context=context,
        conversation_context=conversation_context,
    )


def _llm_response(question: str, plan: QueryPlan, result: pd.DataFrame) -> str:
    from openai import OpenAI

    config = get_llm_config()
    if not config:
        raise RuntimeError("No LLM provider configured.")
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    payload = {
        "question": question,
        "sql": plan.sql,
        "query_intent": plan.intent,
        "metric": plan.metric,
        "retrieved_context": plan.retrieved_context,
        "result_profile": _result_profile(result),
        "result_rows": result.to_dict(orient="records"),
        "row_count": len(result),
    }
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the SQL result rows. Do not invent numbers. "
                    "trade_pnl holds actual trade-level P&L; 95% VaR is literally the 5th-worst day of the scope's own "
                    "100-day rolling window and is NOT additive across scopes. Risk-factor and desk breakdowns of VaR "
                    "are read directly from trade_pnl on the VaR's own historical date, so they are exact (a desk "
                    "breakdown is additive back to the total; a risk-factor breakdown may include supporting "
                    "sensitivity/shock context but the P&L itself is real, not approximated). "
                    "If the result rows include a cob_date, state it explicitly (e.g. 'As of COB 2026-07-08, ...') "
                    "since the book and its VaR both change from one COB to the next. "
                    "Be concise. Write like a market risk analyst briefing a colleague on a trading floor, not "
                    "writing a report: at most 2-3 short sentences, plain prose, no markdown headers, no "
                    "bullet-point walls, no restating every row, no filler ('it's worth noting', 'in summary'). "
                    "Lead with the number, then the one-line reason, stop there. "
                    "Do not include SQL, raw JSON, or mention 'trace' in the visible answer."
                ),
            },
            {"role": "user", "content": json.dumps(payload, default=str)},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _fallback_query_plan(question: str, db_path: Path, conversation_context: str = "") -> QueryPlan:
    lowered = question.lower()
    context_lower = conversation_context.lower()
    planning_lower = lowered if _has_explicit_intent(lowered) else f"{lowered}\n{context_lower}"

    desk = _match_choice(lowered, context_lower, distinct_values("desk", table="trade_sensitivities", db_path=db_path))
    product = _match_choice(lowered, context_lower, distinct_values("product", table="trade_sensitivities", db_path=db_path))
    trade_id = _match_choice(lowered, context_lower, distinct_values("trade_id", table="trade_sensitivities", db_path=db_path))
    risk_factor = _match_choice(lowered, context_lower, distinct_values("risk_factor", table="trade_sensitivities", db_path=db_path))
    historical_date = _match_choice(lowered, context_lower, distinct_values("historical_date", table="risk_factor_scenarios", db_path=db_path))
    cob_date = _resolve_cob_date(lowered, context_lower, db_path)
    top_n = _extract_top_n(lowered)
    days = _extract_days(lowered)
    pnl_filters = _scope_filters(cob_date=cob_date, desk=desk, product=product, trade_id=trade_id)
    ts_filters = _scope_filters(cob_date=cob_date, desk=desk, product=product, trade_id=trade_id, risk_factor=risk_factor, alias="ts")
    pnl_filters_no_cob = _scope_filters(desk=desk, product=product, trade_id=trade_id)
    cob_note = f" (COB {cob_date})" if cob_date else ""

    metric_context = planning_lower
    if not _mentions_var(lowered) and not _mentions_pnl(lowered) and (_mentions_var(context_lower) or _mentions_pnl(context_lower)):
        metric_context = f"{planning_lower}\n{context_lower}"

    if _is_coverage_question(lowered) and not _mentions_pnl(lowered) and not _mentions_var(lowered):
        return QueryPlan(question, _coverage_sql(cob_date), "coverage", "METADATA", "table", False, f"Rule fallback generated coverage SQL{cob_note}.")

    if "risk factor" in planning_lower and ("scenario" in planning_lower or "shock" in planning_lower) and "pnl" not in planning_lower and "p&l" not in planning_lower:
        return QueryPlan(question, _scenario_shock_sql(risk_factor, historical_date, top_n), "scenario_shocks", "SHOCK", "table", False, "Rule fallback generated RF scenario SQL.")

    if _mentions_timeline(planning_lower) or re.search(r"\b\d+\s*[- ]?day\b", planning_lower):
        # Use metric_context, not planning_lower: timeline words alone count as an explicit intent,
        # which would otherwise block inheriting VAR/PNL from the prior question (e.g. a bare "show
        # trend" after "What VAR for entity" must stay VAR, not silently default to PNL).
        if _mentions_var(metric_context):
            # A true VaR trend: the 95% VaR figure itself, recomputed once per COB, across COBs --
            # not one COB's own loss distribution spread across historical_date (that's PNL trend).
            return QueryPlan(question, _var_trend_across_cobs_sql(pnl_filters_no_cob, days), "var_trend", "VAR", "line", False, "Rule fallback generated VaR-per-COB trend SQL, recomputed independently for each COB.")
        return QueryPlan(question, _trend_sql(pnl_filters, days, "PNL", cob_date), "trend", "PNL", "line", False, f"Rule fallback generated PNL trend SQL from trade_pnl{cob_note}.")

    if _mentions_var(metric_context):
        wants_breakdown = "driver" in planning_lower or "explain" in planning_lower or "break" in planning_lower
        if wants_breakdown and desk is None and _mentions_desk_breakdown(planning_lower):
            return QueryPlan(question, _var_desk_breakdown_sql(pnl_filters, top_n, cob_date), "var_desk_drivers", "VAR", "bar", False, f"Rule fallback generated entity VaR desk breakdown SQL, additive on the VaR date{cob_note}.")
        if wants_breakdown and ("risk factor" in planning_lower or not _mentions_desk_breakdown(planning_lower)):
            return QueryPlan(question, _var_driver_sql(pnl_filters, ts_filters, top_n, cob_date), "var_risk_factor_drivers", "VAR", "bar", False, f"Rule fallback generated exact VaR risk-factor breakdown from trade_pnl{cob_note}.")
        return QueryPlan(question, _var_sql(pnl_filters, cob_date), "var", "VAR", "metric", False, f"Rule fallback generated VaR percentile SQL from trade_pnl{cob_note}.")

    return QueryPlan(question, _trade_pnl_sql(pnl_filters, historical_date, top_n), "trade_pnl", "PNL", "bar", False, f"Rule fallback generated trade-level P&L SQL from trade_pnl{cob_note}.")


def _coverage_sql(cob_date: str | None) -> str:
    where = f"WHERE cob_date = {_quote(cob_date)}" if cob_date else ""
    return f"""
SELECT
  desk,
  COUNT(DISTINCT trade_id) AS trade_count,
  COUNT(DISTINCT product) AS product_count,
  COUNT(DISTINCT risk_factor) AS risk_factor_count,
  GROUP_CONCAT(DISTINCT product) AS products
FROM trade_sensitivities
{where}
GROUP BY desk
ORDER BY desk
LIMIT 50
"""


def _scenario_shock_sql(risk_factor: str | None, historical_date: str | None, top_n: int) -> str:
    filters = []
    if risk_factor:
        filters.append(f"risk_factor = {_quote(risk_factor)}")
    if historical_date:
        filters.append(f"historical_date = {_quote(historical_date)}")
    where = "WHERE " + " AND ".join(filters) if filters else ""
    return f"""
SELECT
  historical_date,
  scenario_name,
  risk_factor,
  ROUND(shock_value, 4) AS shock_value,
  shock_unit
FROM risk_factor_scenarios
{where}
ORDER BY historical_date, risk_factor
LIMIT {top_n}
"""


def _trade_pnl_sql(pnl_filters: str, historical_date: str | None, top_n: int) -> str:
    date_filter = f" AND historical_date = {_quote(historical_date)}" if historical_date else ""
    return f"""
SELECT
  cob_date,
  trade_id,
  desk,
  product,
  historical_date,
  scenario_name,
  ROUND(pnl, 4) AS pnl
FROM trade_pnl
WHERE 1 = 1{pnl_filters}{date_filter}
ORDER BY ABS(pnl) DESC
LIMIT {top_n}
"""


def _var_pnl_cte(pnl_filters: str) -> str:
    return f"""
pnl_by_scenario AS (
  SELECT
    historical_date,
    scenario_name,
    SUM(pnl) AS scenario_pnl
  FROM trade_pnl
  WHERE 1 = 1{pnl_filters}
  GROUP BY historical_date, scenario_name
),
losses AS (
  SELECT
    historical_date,
    scenario_name,
    scenario_pnl,
    CASE WHEN scenario_pnl < 0 THEN -scenario_pnl ELSE 0 END AS loss_amount
  FROM pnl_by_scenario
),
ranked AS (
  SELECT
    historical_date,
    scenario_name,
    scenario_pnl,
    loss_amount,
    ROW_NUMBER() OVER (ORDER BY loss_amount) AS rn,
    COUNT(*) OVER () AS scenario_count
  FROM losses
),
target AS (
  -- VaR is literally the k-th worst day, k = ROUND(N x (1 - 0.95)) -- e.g. the 5th worst of 100.
  -- target_rank counts ascending (rn=1 is the smallest loss), so the k-th worst is rn = N - k + 1.
  SELECT scenario_count - CAST(ROUND(MAX(scenario_count * 0.05, 1)) AS INT) + 1 AS target_rank
  FROM ranked
  LIMIT 1
),
var_scenario AS (
  SELECT
    ranked.historical_date,
    ranked.scenario_name,
    ranked.scenario_pnl,
    ranked.loss_amount,
    ranked.scenario_count
  FROM ranked
  JOIN target ON ranked.rn = target.target_rank
)
"""


def _var_sql(pnl_filters: str, cob_date: str | None) -> str:
    cob_col = f"{_quote(cob_date)} AS cob_date," if cob_date else "NULL AS cob_date,"
    return f"""
WITH {_var_pnl_cte(pnl_filters)}
SELECT
  'VAR' AS metric,
  {cob_col}
  0.95 AS confidence_level,
  historical_date,
  scenario_name,
  ROUND(scenario_pnl, 4) AS scenario_pnl,
  ROUND(loss_amount, 4) AS var_95,
  scenario_count
FROM var_scenario
LIMIT 1
"""


def _var_driver_sql(pnl_filters: str, ts_filters: str, top_n: int, cob_date: str | None) -> str:
    """Risk-factor breakdown of a scope's VaR, on that scope's own VaR historical date.

    driver_pnl is exact -- summed directly from trade_pnl, which already stores real P&L per
    risk factor -- so it always reconciles to the scope's total P&L on that date with no residual.
    sensitivity_value/shock_value are joined in purely as supporting color, not as the driver_pnl source.
    """
    cob_col = f"{_quote(cob_date)} AS cob_date," if cob_date else "NULL AS cob_date,"
    return f"""
WITH {_var_pnl_cte(pnl_filters)},
drivers AS (
  SELECT
    tp.risk_factor,
    SUM(tp.pnl) AS driver_pnl
  FROM trade_pnl tp
  JOIN var_scenario
    ON var_scenario.historical_date = tp.historical_date
  WHERE 1 = 1{pnl_filters}
  GROUP BY tp.risk_factor
),
context AS (
  SELECT
    ts.risk_factor,
    ts.sensitivity_type,
    SUM(ts.sensitivity_value) AS sensitivity_value,
    MAX(rs.shock_value) AS shock_value,
    MAX(rs.shock_unit) AS shock_unit
  FROM trade_sensitivities ts
  JOIN risk_factor_scenarios rs
    ON rs.risk_factor = ts.risk_factor
  JOIN var_scenario
    ON var_scenario.historical_date = rs.historical_date
  WHERE 1 = 1{ts_filters}
  GROUP BY ts.risk_factor, ts.sensitivity_type
)
SELECT
  {cob_col}
  d.risk_factor,
  c.sensitivity_type,
  ROUND(c.sensitivity_value, 4) AS sensitivity_value,
  ROUND(c.shock_value, 4) AS shock_value,
  c.shock_unit,
  ROUND(d.driver_pnl, 4) AS driver_pnl,
  (SELECT historical_date FROM var_scenario) AS var_scenario_date
FROM drivers d
LEFT JOIN context c ON c.risk_factor = d.risk_factor
ORDER BY ABS(d.driver_pnl) DESC
LIMIT {top_n}
"""


def _var_desk_breakdown_sql(pnl_filters: str, top_n: int, cob_date: str | None) -> str:
    """Desk breakdown of a scope's (typically entity-level) VaR, on that scope's own VaR historical date.

    Every desk's number comes from the exact same historical date, so unlike desk-level VaR figures
    (each on their own worst-case date) these ARE additive and sum exactly to the scope's total P&L
    on that date.
    """
    cob_col = f"{_quote(cob_date)} AS cob_date," if cob_date else "NULL AS cob_date,"
    return f"""
WITH {_var_pnl_cte(pnl_filters)}
SELECT
  {cob_col}
  tp.desk,
  ROUND(SUM(tp.pnl), 4) AS driver_pnl,
  (SELECT historical_date FROM var_scenario) AS var_scenario_date
FROM trade_pnl tp
JOIN var_scenario
  ON var_scenario.historical_date = tp.historical_date
WHERE 1 = 1{pnl_filters}
GROUP BY tp.desk
ORDER BY ABS(driver_pnl) DESC
LIMIT {top_n}
"""


def _trend_sql(pnl_filters: str, days: int, metric: str, cob_date: str | None) -> str:
    value_expr = "loss_amount" if metric == "VAR" else "scenario_pnl"
    cob_col = f"{_quote(cob_date)} AS cob_date," if cob_date else "NULL AS cob_date,"
    return f"""
WITH {_var_pnl_cte(pnl_filters)}
SELECT
  {cob_col}
  historical_date AS date,
  ROUND({value_expr}, 4) AS value
FROM losses
ORDER BY historical_date DESC
LIMIT {days}
"""


def _var_trend_across_cobs_sql(filters_no_cob: str, days: int) -> str:
    """A true VaR trend: the 95% VaR figure itself, recomputed once per COB, plotted across COBs.

    Different from _trend_sql (which spreads one COB's own loss distribution across historical_date) --
    this ranks each COB's window independently (PARTITION BY cob_date) and keeps only the k-th-worst
    row per COB, so the result is one VaR point per COB, e.g. how VaR moved day over day.
    """
    return f"""
WITH pnl_by_scenario AS (
  SELECT
    cob_date,
    historical_date,
    SUM(pnl) AS scenario_pnl
  FROM trade_pnl
  WHERE 1 = 1{filters_no_cob}
  GROUP BY cob_date, historical_date
),
losses AS (
  SELECT
    cob_date,
    historical_date,
    CASE WHEN scenario_pnl < 0 THEN -scenario_pnl ELSE 0 END AS loss_amount
  FROM pnl_by_scenario
),
ranked AS (
  SELECT
    cob_date,
    historical_date,
    loss_amount,
    ROW_NUMBER() OVER (PARTITION BY cob_date ORDER BY loss_amount) AS rn,
    COUNT(*) OVER (PARTITION BY cob_date) AS scenario_count
  FROM losses
),
target AS (
  SELECT
    cob_date,
    scenario_count - CAST(ROUND(MAX(scenario_count * 0.05, 1)) AS INT) + 1 AS target_rank
  FROM ranked
  GROUP BY cob_date, scenario_count
)
SELECT
  ranked.cob_date AS date,
  ROUND(ranked.loss_amount, 4) AS value,
  ranked.historical_date AS var_scenario_date
FROM ranked
JOIN target
  ON target.cob_date = ranked.cob_date AND target.target_rank = ranked.rn
ORDER BY ranked.cob_date DESC
LIMIT {days}
"""


def _cob_prefix(row: dict[str, Any]) -> str:
    cob_date = row.get("cob_date")
    return f"As of COB {cob_date}, " if cob_date else ""


def _fallback_response(question: str, plan: QueryPlan, result: pd.DataFrame) -> str:
    if result.empty:
        return "No rows matched this query, so there isn't enough data to answer."

    first = result.iloc[0].to_dict()
    cob_prefix = _cob_prefix(first)

    if plan.intent == "coverage" and "desk" in result.columns:
        return f"We cover {result['desk'].nunique()} desks, {int(result['trade_count'].sum())} trades: " + "; ".join(
            f"{row['desk']} ({int(row['trade_count'])} trades, {row['products']})" for _, row in result.iterrows()
        ) + "."

    if {"var_95", "scenario_pnl", "confidence_level"}.issubset(result.columns):
        return (
            f"{cob_prefix}{_fmt(first['confidence_level'])} VaR is {_fmt(first['var_95'])}, the 5th-worst day, {first['historical_date']} "
            f"(aggregated P&L {_fmt(first['scenario_pnl'])}). This is the scope's own worst-case day over its "
            f"100-day rolling window, not a sum of narrower scopes' VaR."
        )

    if "driver_pnl" in result.columns and "desk" in result.columns:
        return (
            f"{cob_prefix}by desk on the VaR date ({first.get('var_scenario_date')}), the largest contributor is "
            f"{first.get('desk')}: {_fmt(first['driver_pnl'])}. These are additive across desks — same date, just split out — "
            f"unlike each desk's own independently-computed VaR."
        )

    if "driver_pnl" in result.columns:
        return (
            f"{cob_prefix}largest driver is {first.get('risk_factor')}: {_fmt(first['driver_pnl'])}"
            + (f" (sensitivity {_fmt(first['sensitivity_value'])} x shock {_fmt(first['shock_value'])} {first.get('shock_unit', '')})" if first.get("sensitivity_value") is not None else "")
            + "."
        )

    if "pnl" in result.columns and "trade_id" in result.columns:
        return f"{cob_prefix}largest P&L is trade {first.get('trade_id')} ({first.get('desk')}/{first.get('product')}) on {first.get('historical_date')}: {_fmt(first['pnl'])}."

    if {"date", "value"}.issubset(result.columns):
        ordered = result.sort_values("date")
        start, end = ordered.iloc[0], ordered.iloc[-1]
        change = float(end["value"]) - float(start["value"])
        return f"{cob_prefix}moved {_fmt(change)}, from {_fmt(start['value'])} on {start['date']} to {_fmt(end['value'])} on {end['date']}."

    if "shock_value" in result.columns:
        return f"{len(result)} historical shock rows; first is {first.get('risk_factor')} on {first.get('historical_date')}: {_fmt(first.get('shock_value'))} {first.get('shock_unit')}."

    return f"{len(result)} row(s) returned. Top row: {first}."


def _result_profile(result: pd.DataFrame) -> dict[str, Any]:
    numeric_summary: dict[str, dict[str, float]] = {}
    for col in result.select_dtypes(include=["float", "int"]).columns:
        series = result[col]
        numeric_summary[col] = {
            "min": float(series.min()) if not series.empty else 0.0,
            "max": float(series.max()) if not series.empty else 0.0,
            "sum": float(series.sum()) if not series.empty else 0.0,
            "mean": float(series.mean()) if not series.empty else 0.0,
        }
    return {"columns": list(result.columns), "row_count": len(result), "numeric_summary": numeric_summary}


def _context_for_prompt(context: tuple[dict[str, Any], ...]) -> str:
    if not context:
        return "No retrieved context."
    return "\n".join(f"- {item['title']} (score {item['score']}): {item['text']}" for item in context)


def _is_coverage_question(lowered: str) -> bool:
    return any(term in lowered for term in ("cover", "covering", "covered", "available", "list", "show", "what")) and any(
        term in lowered for term in ("desk", "desks", "trade", "trades", "product", "products")
    )


def _mentions_var(text: str) -> bool:
    return bool(re.search(r"\bvar\b|value at risk", text))


def _mentions_pnl(text: str) -> bool:
    return bool(re.search(r"\bpnl\b|\bp&l\b|\bpl\b|profit|loss", text))


def _mentions_desk_breakdown(text: str) -> bool:
    return bool(re.search(r"by desk|desk level|desk breakdown|per desk|across desks|each desk|by book", text))


def _mentions_timeline(text: str) -> bool:
    """Any phrasing that asks to see a series over time, plotted as a line chart.

    Deliberately excludes words like "movement" alone, which is already used to mean "explain what
    drove this VaR" (e.g. "Explain the Desk D1 Var movement") -- a single-date breakdown, not a
    time series -- so it must not be redirected into the trend branch.
    """
    return bool(re.search(r"\btrend\b|\bgraph\b|\bchart\b|\bplot\b|\btimeline\b|visuali[sz]e|over time|\bhistory\b", text))


def _has_explicit_intent(lowered: str) -> bool:
    return _mentions_timeline(lowered) or any(term in lowered for term in ("scenario", "shock", "day", "risk factor", "driver", "var", "pnl", "p&l", "pl", "trade"))


def _extract_top_n(lowered: str) -> int:
    top_match = re.search(r"\btop\s+(\d+)\b", lowered)
    if top_match:
        return max(1, min(25, int(top_match.group(1))))
    show_match = re.search(r"\bshow\s+(\d+)\s+(?![- ]?day\b)", lowered)
    if show_match:
        return max(1, min(25, int(show_match.group(1))))
    return 10


def _extract_days(lowered: str) -> int:
    match = re.search(r"\b(\d+)\s*[- ]?day\b", lowered)
    if not match:
        return 10
    return max(2, min(60, int(match.group(1))))


def _literal_match(text: str, choices: list[str]) -> str | None:
    for choice in sorted(choices, key=len, reverse=True):
        if choice.lower() in text:
            return choice
    return None


def _compact_match(text: str, choices: list[str]) -> str | None:
    compact_text = re.sub(r"[^a-z0-9]", "", text)
    for choice in sorted(choices, key=len, reverse=True):
        if re.sub(r"[^a-z0-9]", "", choice.lower()) in compact_text:
            return choice
    return None


def _match_choice(question_lower: str, context_lower: str, choices: list[str]) -> str | None:
    """Resolve a scope value (desk/product/trade_id/...), preferring an explicit mention in the
    current question, then falling back to prior conversation turns for follow-ups.

    The fuzzy "compact" pass (strips spaces/punctuation, for phrasing like "T-1" vs "T1") only ever
    runs against the CURRENT question, never against conversation context. Context includes raw
    generated SQL for cob_date continuity, and compact-matching SQL is unsafe: e.g. "...AS INT) + 1"
    compacts to "...asint1...", which falsely substring-matches trade_id "T1".
    """
    return _literal_match(question_lower, choices) or _literal_match(context_lower, choices) or _compact_match(question_lower, choices)


def _resolve_cob_date(question_lower: str, context_lower: str, db_path: Path) -> str | None:
    """The single cob_date every trade_pnl/trade_sensitivities query must be scoped to.

    Matches an explicit date mention, "latest"/"today"/"current"/"now", or falls back to the
    most recent cob_date on the book — the same default the LLM path is instructed to use.
    """
    cob_dates = distinct_values("cob_date", table="trade_sensitivities", db_path=db_path)
    matched = _match_choice(question_lower, context_lower, cob_dates)
    if matched:
        return matched
    if re.search(r"\b(latest|today|current|now|most recent)\b", f"{question_lower}\n{context_lower}"):
        return latest_cob_date(db_path)
    return latest_cob_date(db_path)


def _scope_filters(
    cob_date: str | None = None,
    desk: str | None = None,
    product: str | None = None,
    trade_id: str | None = None,
    risk_factor: str | None = None,
    alias: str = "",
) -> str:
    prefix = f"{alias}." if alias else ""
    filters = []
    if cob_date:
        filters.append(f"{prefix}cob_date = {_quote(cob_date)}")
    if desk:
        filters.append(f"{prefix}desk = {_quote(desk)}")
    if product:
        filters.append(f"{prefix}product = {_quote(product)}")
    if trade_id:
        filters.append(f"{prefix}trade_id = {_quote(trade_id)}")
    if risk_factor:
        filters.append(f"{prefix}risk_factor = {_quote(risk_factor)}")
    return " AND " + " AND ".join(filters) if filters else ""


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)
