from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .db import DB_PATH, distinct_values, get_connection
from .document_store import retrieve_context
from .llm_config import get_llm_config


MAX_RESULT_ROWS = 50

SCHEMA_CONTEXT = """
SQLite schema:
- trade_sensitivities(trade_id, risk_factor, desk, sensitivity_type, sensitivity_value, product)
- risk_factor_scenarios(historical_date, scenario_name, risk_factor, shock_value, shock_unit)
- trade_pnl(trade_id, desk, product, historical_date, scenario_name, pnl)

Business rules:
- These are the three stored input tables. trade_pnl holds the actual trade-level P&L for every
  trade on every historical scenario date (as if from full revaluation) and is the source of truth
  for aggregation and VaR.
- Portfolio/desk/entity scenario P&L = SUM(trade_pnl.pnl) grouped by historical_date for the scope
  (desk/product/trade filters). Never derive portfolio P&L from trade_sensitivities x
  risk_factor_scenarios; use trade_pnl for that.
- 95% VaR uses the nearest-rank method: rank the scope's historical loss_amounts ascending
  (loss_amount = max(-scenario_pnl, 0)) and take the ceil(N x 0.95)-th smallest, i.e. the 3rd-worst
  day out of 50 historical scenarios. Not linear-interpolated percentile.
- VaR is NOT additive: entity-level VaR is not the sum of desk-level VaRs, because each scope's
  95th-percentile scenario date can differ. Only P&L is additive across scopes, VaR is not.
- To explain WHAT drives VaR on its worst-case scenario date, join trade_sensitivities to
  risk_factor_scenarios on risk_factor (sensitivity_value * shock_value) for that single date only.
  This is a linear risk-factor attribution of the actual trade_pnl and may leave a small unexplained
  residual versus the actual stored P&L (non-linear/convexity effects a static sensitivity can't capture).
- Do not treat sensitivity as P&L. Do not treat shocks as P&L. Attribution P&L requires sensitivity
  times shock, and only approximates the real trade_pnl.

SQL rules:
- Generate one read-only SQLite SELECT or WITH statement.
- Do not use SELECT *.
- Always aggregate or filter to the user's question.
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
            if not _llm_sql_uses_trade_pnl_where_required(cleaned, plan.sql):
                raise ValueError("LLM SQL aggregated P&L without sourcing trade_pnl.")
            return plan
        except Exception:
            pass

    return replace(
        _fallback_query_plan(cleaned, db_path, conversation_context=conversation_context),
        retrieved_context=context,
        conversation_context=conversation_context,
    )


def _llm_sql_uses_trade_pnl_where_required(question: str, sql: str) -> bool:
    """Guard against the LLM silently deriving portfolio P&L from sensitivity x shock instead of trade_pnl.

    That SQL still executes successfully (so execute_query_plan won't catch it), but the VaR-scenario
    date it finds is only a linear approximation of the real trade_pnl distribution. Only VaR/P&L/trend/
    driver questions require trade_pnl; coverage and raw shock-lookup questions legitimately don't touch it.
    """
    lowered = question.lower()
    if _is_coverage_question(lowered) and not _mentions_pnl(lowered) and not _mentions_var(lowered):
        return True
    if "risk factor" in lowered and ("scenario" in lowered or "shock" in lowered) and not _mentions_pnl(lowered) and not _mentions_var(lowered):
        return True
    if _mentions_var(lowered) or _mentions_pnl(lowered) or "trend" in lowered or "driver" in lowered or "explain" in lowered or re.search(r"\b\d+\s*[- ]?day\b", lowered):
        return "trade_pnl" in sql.lower()
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
                    "trade_pnl holds actual trade-level P&L; VaR is the nearest-rank percentile loss from that "
                    "scope's aggregated distribution and is NOT additive across scopes. Sensitivity x shock only "
                    "explains what drove VaR on its worst-case date and may leave a small residual versus actual P&L. "
                    "Write like a market risk analyst briefing a colleague: 2-4 sentences, plain prose, no markdown "
                    "headers, no bullet-point walls, no restating every row. Lead with the number, then the one-line "
                    "reason. Do not include SQL, raw JSON, or mention 'trace' in the visible answer."
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
    scope_lower = f"{lowered}\n{context_lower}"

    desk = _match_choice(scope_lower, distinct_values("desk", table="trade_sensitivities", db_path=db_path))
    product = _match_choice(scope_lower, distinct_values("product", table="trade_sensitivities", db_path=db_path))
    trade_id = _match_choice(scope_lower, distinct_values("trade_id", table="trade_sensitivities", db_path=db_path))
    risk_factor = _match_choice(scope_lower, distinct_values("risk_factor", table="trade_sensitivities", db_path=db_path))
    historical_date = _match_choice(scope_lower, distinct_values("historical_date", table="risk_factor_scenarios", db_path=db_path))
    top_n = _extract_top_n(lowered)
    days = _extract_days(lowered)
    pnl_filters = _scope_filters(desk=desk, product=product, trade_id=trade_id)
    ts_filters = _scope_filters(desk=desk, product=product, trade_id=trade_id, risk_factor=risk_factor, alias="ts")

    metric_context = planning_lower
    if not _mentions_var(lowered) and not _mentions_pnl(lowered) and (_mentions_var(context_lower) or _mentions_pnl(context_lower)):
        metric_context = f"{planning_lower}\n{context_lower}"

    if _is_coverage_question(lowered) and not _mentions_pnl(lowered) and not _mentions_var(lowered):
        return QueryPlan(question, _coverage_sql(), "coverage", "METADATA", "table", False, "Rule fallback generated coverage SQL.")

    if "risk factor" in planning_lower and ("scenario" in planning_lower or "shock" in planning_lower) and "pnl" not in planning_lower and "p&l" not in planning_lower:
        return QueryPlan(question, _scenario_shock_sql(risk_factor, historical_date, top_n), "scenario_shocks", "SHOCK", "table", False, "Rule fallback generated RF scenario SQL.")

    if "trend" in planning_lower or re.search(r"\b\d+\s*[- ]?day\b", planning_lower):
        metric = "VAR" if _mentions_var(planning_lower) else "PNL"
        return QueryPlan(question, _trend_sql(pnl_filters, days, metric), "trend", metric, "line", False, "Rule fallback generated P&L trend SQL from trade_pnl.")

    if _mentions_var(metric_context):
        if "risk factor" in planning_lower or "driver" in planning_lower or "explain" in planning_lower:
            return QueryPlan(question, _var_driver_sql(pnl_filters, ts_filters, top_n), "var_risk_factor_drivers", "VAR", "bar", False, "Rule fallback generated VaR risk-factor attribution SQL with residual.")
        return QueryPlan(question, _var_sql(pnl_filters), "var", "VAR", "metric", False, "Rule fallback generated VaR percentile SQL from trade_pnl.")

    return QueryPlan(question, _trade_pnl_sql(pnl_filters, historical_date, top_n), "trade_pnl", "PNL", "bar", False, "Rule fallback generated trade-level P&L SQL from trade_pnl.")


def _coverage_sql() -> str:
    return """
SELECT
  desk,
  COUNT(DISTINCT trade_id) AS trade_count,
  COUNT(DISTINCT product) AS product_count,
  COUNT(DISTINCT risk_factor) AS risk_factor_count,
  GROUP_CONCAT(DISTINCT product) AS products
FROM trade_sensitivities
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
  SELECT CAST((scenario_count * 0.95) + 0.999999 AS INT) AS target_rank
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


def _var_sql(pnl_filters: str) -> str:
    return f"""
WITH {_var_pnl_cte(pnl_filters)}
SELECT
  'VAR' AS metric,
  0.95 AS confidence_level,
  historical_date,
  scenario_name,
  ROUND(scenario_pnl, 4) AS scenario_pnl,
  ROUND(loss_amount, 4) AS var_95,
  scenario_count
FROM var_scenario
LIMIT 1
"""


def _var_driver_sql(pnl_filters: str, ts_filters: str, top_n: int) -> str:
    return f"""
WITH {_var_pnl_cte(pnl_filters)},
attribution AS (
  SELECT
    ts.risk_factor,
    ts.sensitivity_type,
    SUM(ts.sensitivity_value) AS sensitivity_value,
    MAX(rs.shock_value) AS shock_value,
    MAX(rs.shock_unit) AS shock_unit,
    SUM(ts.sensitivity_value * rs.shock_value) AS driver_pnl
  FROM trade_sensitivities ts
  JOIN risk_factor_scenarios rs
    ON rs.risk_factor = ts.risk_factor
  JOIN var_scenario
    ON var_scenario.historical_date = rs.historical_date
  WHERE 1 = 1{ts_filters}
  GROUP BY ts.risk_factor, ts.sensitivity_type
),
drivers_with_residual AS (
  SELECT
    risk_factor,
    sensitivity_type,
    ROUND(sensitivity_value, 4) AS sensitivity_value,
    ROUND(shock_value, 4) AS shock_value,
    shock_unit,
    ROUND(driver_pnl, 4) AS driver_pnl,
    (SELECT historical_date FROM var_scenario) AS var_scenario_date
  FROM attribution
  UNION ALL
  SELECT
    'Unexplained (non-linear residual)' AS risk_factor,
    'Residual' AS sensitivity_type,
    NULL AS sensitivity_value,
    NULL AS shock_value,
    NULL AS shock_unit,
    ROUND((SELECT scenario_pnl FROM var_scenario) - (SELECT SUM(driver_pnl) FROM attribution), 4) AS driver_pnl,
    (SELECT historical_date FROM var_scenario) AS var_scenario_date
)
SELECT
  risk_factor,
  sensitivity_type,
  sensitivity_value,
  shock_value,
  shock_unit,
  driver_pnl,
  var_scenario_date
FROM drivers_with_residual
ORDER BY ABS(driver_pnl) DESC
LIMIT {top_n}
"""


def _trend_sql(pnl_filters: str, days: int, metric: str) -> str:
    value_expr = "loss_amount" if metric == "VAR" else "scenario_pnl"
    return f"""
WITH {_var_pnl_cte(pnl_filters)}
SELECT
  historical_date AS date,
  ROUND({value_expr}, 4) AS value
FROM losses
ORDER BY historical_date DESC
LIMIT {days}
"""


def _fallback_response(question: str, plan: QueryPlan, result: pd.DataFrame) -> str:
    if result.empty:
        return "No rows matched this query, so there isn't enough data to answer."

    first = result.iloc[0].to_dict()

    if plan.intent == "coverage" and "desk" in result.columns:
        return f"We cover {result['desk'].nunique()} desks, {int(result['trade_count'].sum())} trades: " + "; ".join(
            f"{row['desk']} ({int(row['trade_count'])} trades, {row['products']})" for _, row in result.iterrows()
        ) + "."

    if {"var_95", "scenario_pnl", "confidence_level"}.issubset(result.columns):
        return (
            f"{_fmt(first['confidence_level'])} VaR is {_fmt(first['var_95'])}, driven by the {first['historical_date']} "
            f"scenario (aggregated P&L {_fmt(first['scenario_pnl'])}). This is the scope's own worst-case percentile, "
            f"not a sum of narrower scopes' VaR."
        )

    if "driver_pnl" in result.columns:
        if first.get("risk_factor") == "Unexplained (non-linear residual)":
            return f"Unexplained residual of {_fmt(first['driver_pnl'])} — the gap between actual P&L on the VaR date and the linear risk-factor attribution."
        return (
            f"Largest driver is {first.get('risk_factor')}: {_fmt(first['driver_pnl'])} "
            f"(sensitivity {_fmt(first.get('sensitivity_value', 0))} x shock {_fmt(first.get('shock_value', 0))} {first.get('shock_unit', '')})."
        )

    if "pnl" in result.columns and "trade_id" in result.columns:
        return f"Largest P&L is trade {first.get('trade_id')} ({first.get('desk')}/{first.get('product')}) on {first.get('historical_date')}: {_fmt(first['pnl'])}."

    if {"date", "value"}.issubset(result.columns):
        ordered = result.sort_values("date")
        start, end = ordered.iloc[0], ordered.iloc[-1]
        change = float(end["value"]) - float(start["value"])
        return f"Moved {_fmt(change)}, from {_fmt(start['value'])} on {start['date']} to {_fmt(end['value'])} on {end['date']}."

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


def _has_explicit_intent(lowered: str) -> bool:
    return any(term in lowered for term in ("scenario", "shock", "trend", "day", "risk factor", "driver", "var", "pnl", "p&l", "pl", "trade"))


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


def _match_choice(lowered: str, choices: list[str]) -> str | None:
    for choice in sorted(choices, key=len, reverse=True):
        if choice.lower() in lowered:
            return choice
    compact_query = re.sub(r"[^a-z0-9]", "", lowered)
    for choice in sorted(choices, key=len, reverse=True):
        if re.sub(r"[^a-z0-9]", "", choice.lower()) in compact_query:
            return choice
    return None


def _scope_filters(
    desk: str | None = None,
    product: str | None = None,
    trade_id: str | None = None,
    risk_factor: str | None = None,
    alias: str = "",
) -> str:
    prefix = f"{alias}." if alias else ""
    filters = []
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
