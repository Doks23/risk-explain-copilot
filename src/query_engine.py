from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from .db import DB_PATH, distinct_values, get_connection
from .vector_store import retrieve_context


load_dotenv()
MAX_RESULT_ROWS = 50

SCHEMA_CONTEXT = """
SQLite schema:
- hierarchy(date, desk, book, portfolio, product, currency)
- pnl_results(date, desk, book, portfolio, product, pnl_value)
- var_results(date, desk, book, portfolio, scenario, risk_factor, product, var_contribution)
- sensitivities(date, desk, book, portfolio, product, risk_factor, sensitivity_type, sensitivity_value)
- market_data(date, risk_factor, market_level, market_move, move_unit)
- scenario_data(date, scenario, risk_factor, shock_value, shock_unit)

Business rules:
- PNL is actual or hypothetical profit/loss movement between two dates.
- Explain PNL with estimated_pnl_impact = sensitivity_value from date1 multiplied by market_move from date1 exclusive to date2 inclusive.
- Compare actual PNL movement from pnl_results with explained PNL and residual PNL.
- Sensitivity is exposure, not final PNL.
- Market move alone is not a PNL explanation unless combined with sensitivity.
- VaR is a risk estimate, not actual loss.
- Explain VaR using var_contribution movement in var_results, grouped by scenario or risk_factor.
- Do not explain VaR directly as PNL and do not explain PNL directly as VaR.

SQL rules:
- Generate one read-only SQLite SELECT or WITH statement.
- Do not use SELECT *.
- Always aggregate or filter to the user's question.
- Always include LIMIT 50 or lower.
- Do not expose full database contents.
- Prefer latest date versus previous business date for movement questions unless the user specifies dates.
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


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str | None
    model: str


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
            return plan
        except Exception:
            pass

    return replace(
        _fallback_query_plan(cleaned, db_path, conversation_context=conversation_context),
        retrieved_context=context,
        conversation_context=conversation_context,
    )


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


def get_llm_config() -> LLMConfig | None:
    if os.getenv("OPENAI_API_KEY"):
        return LLMConfig(
            provider="OpenAI-compatible",
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
    if os.getenv("GEMINI_API_KEY"):
        return LLMConfig(
            provider="Gemini",
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        )
    return None


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
    if re.search(r"\blimit\s+\d+\s*$", sql, flags=re.IGNORECASE):
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
                    f"Recent conversation context for resolving follow-up questions:\n"
                    f"{conversation_context or 'No prior conversation context.'}\n\n"
                    f"Retrieved business context:\n{_context_for_prompt(context)}\n\n"
                    f"Current question: {question}\n\n"
                    "Use recent conversation only to resolve missing scope, metric, or intent. "
                    "If the current question explicitly names a metric, desk, book, date, or intent, it overrides prior context."
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
                    "You are the answer-synthesis layer of a market risk copilot. "
                    "Answer the user's question directly using only the SQL result rows provided. "
                    "Do not invent numbers, dates, desks, books, drivers, or explanations. "
                    "If the result is insufficient, say what is missing. "
                    "For PNL, explain actual PNL change, explained PNL from sensitivity times market move, and residual when present. "
                    "For VaR, explain movement as change in scenario or risk-factor risk contribution; do not describe it as actual loss. "
                    "For trends, summarize start, end, change, high, and low visible in the rows. "
                    "Use concise markdown. Do not include SQL, raw JSON, Evidence from SQL, or Data trace in the visible answer."
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

    if _is_coverage_question(lowered):
        return _coverage_query_plan(question, lowered)

    metric = _metric_from_text(lowered, context_lower)
    date1, date2 = _latest_comparison_dates(db_path)
    top_n = _extract_top_n(lowered)
    days = _extract_days(lowered)
    desk = _match_choice(scope_lower, distinct_values("desk", table="hierarchy", db_path=db_path))
    book = _match_choice(scope_lower, distinct_values("book", table="hierarchy", db_path=db_path))
    filters = _scope_filters(desk, book)

    if "trend" in planning_lower or re.search(r"\b\d+\s*[- ]?day\b", planning_lower):
        return QueryPlan(
            question,
            _trend_sql(metric, filters, days),
            "trend",
            metric,
            "line",
            False,
            "Rule fallback generated trend SQL.",
        )

    if "drill" in planning_lower:
        return QueryPlan(
            question,
            _pnl_drilldown_sql(date1, date2, filters, top_n) if metric == "PNL" else _var_drilldown_sql(date1, date2, filters, top_n),
            "drilldown",
            metric,
            "table",
            False,
            "Rule fallback generated bounded drill-down SQL.",
        )

    if metric == "PNL" and ("risk factor" in planning_lower or "risk-factor" in planning_lower or "market move" in planning_lower or "market moves" in planning_lower or "driver" in planning_lower):
        return QueryPlan(
            question,
            _pnl_risk_factor_sql(date1, date2, filters, top_n),
            "pnl_risk_factor_attribution",
            "PNL",
            "bar",
            False,
            "Rule fallback generated PNL sensitivity-times-market-move SQL.",
        )

    if metric == "VAR" and "scenario" in planning_lower:
        return QueryPlan(
            question,
            _var_driver_sql("scenario", date1, date2, filters, top_n),
            "var_scenario_drivers",
            "VAR",
            "bar",
            False,
            "Rule fallback generated VaR scenario driver SQL.",
        )

    if metric == "VAR" and ("risk factor" in planning_lower or "risk-factor" in planning_lower or "driver" in planning_lower):
        return QueryPlan(
            question,
            _var_driver_sql("risk_factor", date1, date2, filters, top_n),
            "var_risk_factor_drivers",
            "VAR",
            "bar",
            False,
            "Rule fallback generated VaR risk-factor driver SQL.",
        )

    if metric == "PNL":
        return QueryPlan(
            question,
            _pnl_movement_sql(date1, date2, filters),
            "pnl_movement",
            "PNL",
            "metric",
            False,
            "Rule fallback generated PNL movement and attribution SQL.",
        )

    return QueryPlan(
        question,
        _var_movement_sql(date1, date2, filters),
        "var_movement",
        "VAR",
        "metric",
        False,
        "Rule fallback generated VaR movement SQL.",
    )


def _pnl_movement_sql(date1: str, date2: str, filters: str) -> str:
    return f"""
WITH actual AS (
  SELECT date, SUM(pnl_value) AS value
  FROM pnl_results
  WHERE date IN ({_quote(date1)}, {_quote(date2)}){filters}
  GROUP BY date
),
sens AS (
  SELECT risk_factor, SUM(sensitivity_value) AS sensitivity_value
  FROM sensitivities
  WHERE date = {_quote(date1)}{filters}
  GROUP BY risk_factor
),
moves AS (
  SELECT risk_factor, SUM(market_move) AS market_move
  FROM market_data
  WHERE date > {_quote(date1)} AND date <= {_quote(date2)}
  GROUP BY risk_factor
),
explained AS (
  SELECT COALESCE(SUM(sens.sensitivity_value * COALESCE(moves.market_move, 0)), 0) AS explained_pnl
  FROM sens
  LEFT JOIN moves ON moves.risk_factor = sens.risk_factor
),
totals AS (
  SELECT
    COALESCE(SUM(CASE WHEN date = {_quote(date1)} THEN value ELSE 0 END), 0) AS pnl_date_1,
    COALESCE(SUM(CASE WHEN date = {_quote(date2)} THEN value ELSE 0 END), 0) AS pnl_date_2
  FROM actual
)
SELECT
  'PNL' AS metric,
  {_quote(date1)} AS previous_date,
  ROUND(totals.pnl_date_1, 4) AS pnl_date_1,
  {_quote(date2)} AS current_date,
  ROUND(totals.pnl_date_2, 4) AS pnl_date_2,
  ROUND(totals.pnl_date_2 - totals.pnl_date_1, 4) AS pnl_change,
  ROUND(explained.explained_pnl, 4) AS explained_pnl,
  ROUND((totals.pnl_date_2 - totals.pnl_date_1) - explained.explained_pnl, 4) AS residual_pnl
FROM totals
CROSS JOIN explained
LIMIT 1
"""


def _var_movement_sql(date1: str, date2: str, filters: str) -> str:
    return f"""
WITH totals AS (
  SELECT date, SUM(var_contribution) AS value
  FROM var_results
  WHERE date IN ({_quote(date1)}, {_quote(date2)}){filters}
  GROUP BY date
),
pivot AS (
  SELECT
    COALESCE(SUM(CASE WHEN date = {_quote(date1)} THEN value ELSE 0 END), 0) AS var_date_1,
    COALESCE(SUM(CASE WHEN date = {_quote(date2)} THEN value ELSE 0 END), 0) AS var_date_2
  FROM totals
)
SELECT
  'VAR' AS metric,
  {_quote(date1)} AS previous_date,
  ROUND(var_date_1, 4) AS var_date_1,
  {_quote(date2)} AS current_date,
  ROUND(var_date_2, 4) AS var_date_2,
  ROUND(var_date_2 - var_date_1, 4) AS var_change,
  ROUND(CASE WHEN var_date_1 = 0 THEN 0 ELSE (var_date_2 - var_date_1) * 100.0 / var_date_1 END, 4) AS percentage_change
FROM pivot
LIMIT 1
"""


def _pnl_risk_factor_sql(date1: str, date2: str, filters: str, top_n: int) -> str:
    return f"""
WITH sens AS (
  SELECT
    risk_factor,
    sensitivity_type,
    SUM(sensitivity_value) AS sensitivity_value
  FROM sensitivities
  WHERE date = {_quote(date1)}{filters}
  GROUP BY risk_factor, sensitivity_type
),
moves AS (
  SELECT
    risk_factor,
    SUM(market_move) AS market_move,
    MAX(move_unit) AS move_unit
  FROM market_data
  WHERE date > {_quote(date1)} AND date <= {_quote(date2)}
  GROUP BY risk_factor
)
SELECT
  sens.risk_factor,
  sens.sensitivity_type,
  ROUND(sens.sensitivity_value, 4) AS sensitivity_value,
  ROUND(COALESCE(moves.market_move, 0), 4) AS market_move,
  COALESCE(moves.move_unit, '') AS move_unit,
  ROUND(sens.sensitivity_value * COALESCE(moves.market_move, 0), 4) AS estimated_pnl_impact
FROM sens
LEFT JOIN moves ON moves.risk_factor = sens.risk_factor
ORDER BY ABS(estimated_pnl_impact) DESC
LIMIT {top_n}
"""


def _var_driver_sql(group_col: str, date1: str, date2: str, filters: str, top_n: int) -> str:
    if group_col not in {"scenario", "risk_factor"}:
        raise ValueError("Unsupported VaR driver column")
    return f"""
WITH driver AS (
  SELECT date, {group_col}, SUM(var_contribution) AS value
  FROM var_results
  WHERE date IN ({_quote(date1)}, {_quote(date2)}){filters}
  GROUP BY date, {group_col}
),
pivot AS (
  SELECT
    {group_col},
    COALESCE(SUM(CASE WHEN date = {_quote(date1)} THEN value ELSE 0 END), 0) AS previous_value,
    COALESCE(SUM(CASE WHEN date = {_quote(date2)} THEN value ELSE 0 END), 0) AS current_value
  FROM driver
  GROUP BY {group_col}
)
SELECT
  {group_col},
  ROUND(previous_value, 4) AS previous_value,
  ROUND(current_value, 4) AS current_value,
  ROUND(current_value - previous_value, 4) AS delta
FROM pivot
ORDER BY ABS(delta) DESC
LIMIT {top_n}
"""


def _trend_sql(metric: str, filters: str, days: int) -> str:
    table = "pnl_results" if metric == "PNL" else "var_results"
    value_col = "pnl_value" if metric == "PNL" else "var_contribution"
    return f"""
SELECT date, ROUND(value, 4) AS value
FROM (
  SELECT date, SUM({value_col}) AS value
  FROM {table}
  WHERE 1 = 1{filters}
  GROUP BY date
  ORDER BY date DESC
  LIMIT {days}
)
ORDER BY date
"""


def _var_drilldown_sql(date1: str, date2: str, filters: str, top_n: int) -> str:
    return f"""
WITH driver AS (
  SELECT
    date,
    desk,
    book,
    portfolio,
    product,
    scenario,
    risk_factor,
    SUM(var_contribution) AS value
  FROM var_results
  WHERE date IN ({_quote(date1)}, {_quote(date2)}){filters}
  GROUP BY date, desk, book, portfolio, product, scenario, risk_factor
),
pivot AS (
  SELECT
    desk,
    book,
    portfolio,
    product,
    scenario,
    risk_factor,
    COALESCE(SUM(CASE WHEN date = {_quote(date1)} THEN value ELSE 0 END), 0) AS previous_value,
    COALESCE(SUM(CASE WHEN date = {_quote(date2)} THEN value ELSE 0 END), 0) AS current_value
  FROM driver
  GROUP BY desk, book, portfolio, product, scenario, risk_factor
)
SELECT
  desk,
  book,
  portfolio,
  product,
  scenario,
  risk_factor,
  ROUND(previous_value, 4) AS previous_value,
  ROUND(current_value, 4) AS current_value,
  ROUND(current_value - previous_value, 4) AS delta
FROM pivot
ORDER BY ABS(delta) DESC
LIMIT {top_n}
"""


def _pnl_drilldown_sql(date1: str, date2: str, filters: str, top_n: int) -> str:
    return f"""
WITH sens AS (
  SELECT
    desk,
    book,
    portfolio,
    product,
    risk_factor,
    sensitivity_type,
    SUM(sensitivity_value) AS sensitivity_value
  FROM sensitivities
  WHERE date = {_quote(date1)}{filters}
  GROUP BY desk, book, portfolio, product, risk_factor, sensitivity_type
),
moves AS (
  SELECT risk_factor, SUM(market_move) AS market_move, MAX(move_unit) AS move_unit
  FROM market_data
  WHERE date > {_quote(date1)} AND date <= {_quote(date2)}
  GROUP BY risk_factor
)
SELECT
  sens.desk,
  sens.book,
  sens.portfolio,
  sens.product,
  'Actual market move' AS scenario,
  sens.risk_factor,
  sens.sensitivity_type,
  ROUND(sens.sensitivity_value, 4) AS sensitivity_value,
  ROUND(COALESCE(moves.market_move, 0), 4) AS market_move,
  COALESCE(moves.move_unit, '') AS move_unit,
  ROUND(sens.sensitivity_value * COALESCE(moves.market_move, 0), 4) AS estimated_pnl_impact
FROM sens
LEFT JOIN moves ON moves.risk_factor = sens.risk_factor
ORDER BY ABS(estimated_pnl_impact) DESC
LIMIT {top_n}
"""


def _fallback_response(question: str, plan: QueryPlan, result: pd.DataFrame) -> str:
    if result.empty:
        return "### Answer\nThe query returned no rows, so there is not enough data to answer from the available result."

    first = result.iloc[0].to_dict()
    lines = ["### Answer"]

    if plan.intent == "desk_coverage" and "desk" in result.columns:
        total_desks = result["desk"].nunique()
        total_books = int(result["book_count"].sum()) if "book_count" in result.columns else len(result)
        lines.append(f"We cover {total_desks} desks across {total_books} books in the hierarchy dataset.")
        lines.append("\n### Covered desks")
        for _, row in result.iterrows():
            lines.append(f"- {row['desk']}: {int(row['book_count'])} books; books: {row['books']}; products: {row['products']}")
    elif plan.intent == "book_coverage" and {"desk", "book"}.issubset(result.columns):
        lines.append(f"The hierarchy query returned {len(result)} desk/book combinations.")
        lines.append("\n### Covered books")
        for _, row in result.iterrows():
            lines.append(f"- {row['desk']} / {row['book']}: {row.get('portfolio', 'n/a')} - {row.get('product', 'n/a')} ({row.get('currency', 'n/a')})")
    elif {"pnl_date_1", "pnl_date_2", "pnl_change", "explained_pnl", "residual_pnl"}.issubset(result.columns):
        direction = "increased" if float(first["pnl_change"]) >= 0 else "decreased"
        lines.append(
            f"PNL {direction} by {_fmt(first['pnl_change'])}, from {_fmt(first['pnl_date_1'])} on "
            f"{first['previous_date']} to {_fmt(first['pnl_date_2'])} on {first['current_date']}."
        )
        lines.append(
            f"The sensitivity x market move attribution explains {_fmt(first['explained_pnl'])}; "
            f"the residual is {_fmt(first['residual_pnl'])}."
        )
    elif {"var_date_1", "var_date_2", "var_change", "percentage_change"}.issubset(result.columns):
        direction = "increased" if float(first["var_change"]) >= 0 else "decreased"
        lines.append(
            f"VaR {direction} by {_fmt(first['var_change'])}, from {_fmt(first['var_date_1'])} on "
            f"{first['previous_date']} to {_fmt(first['var_date_2'])} on {first['current_date']} "
            f"({_fmt(first['percentage_change'])}%)."
        )
        lines.append("This is a risk estimate movement based on VaR contributions, not an actual PNL movement.")
    elif "estimated_pnl_impact" in result.columns:
        lines.append(
            f"The largest estimated PNL impact is {first.get('risk_factor', 'n/a')}: "
            f"{_fmt(first['estimated_pnl_impact'])}, calculated from sensitivity "
            f"{_fmt(first.get('sensitivity_value', 0))} x market move {_fmt(first.get('market_move', 0))} "
            f"{first.get('move_unit', '')}."
        )
    elif "delta" in result.columns:
        label = _row_label(first, result)
        lines.append(f"The largest returned VaR contribution movement is {label}, with delta {_fmt(first['delta'])}.")
    elif {"date", "value"}.issubset(result.columns):
        start = result.iloc[0]
        end = result.iloc[-1]
        change = float(end["value"]) - float(start["value"])
        pct_change = (change / float(start["value"]) * 100) if float(start["value"]) else 0.0
        high = result.loc[result["value"].idxmax()]
        low = result.loc[result["value"].idxmin()]
        direction = "increased" if change >= 0 else "decreased"
        lines.append(
            f"The returned trend {direction} by {_fmt(change)}, from {_fmt(start['value'])} on {start['date']} "
            f"to {_fmt(end['value'])} on {end['date']} ({_fmt(pct_change)}%)."
        )
        lines.append(f"The high is {_fmt(high['value'])} on {high['date']}; the low is {_fmt(low['value'])} on {low['date']}.")
    else:
        lines.append(f"The SQL returned {len(result)} row(s). The top row is: {first}.")

    numeric_cols = [col for col in result.columns if pd.api.types.is_numeric_dtype(result[col])]
    if plan.intent not in {"desk_coverage", "book_coverage", "trend"}:
        lines.append("\n### Key metrics")
        if numeric_cols:
            for col in numeric_cols[:6]:
                lines.append(f"- {col}: {_fmt(first[col])} on top returned row")
        else:
            lines.append("- No numeric metric columns were returned.")

        if len(result) > 1:
            lines.append("\n### Top returned rows")
            for _, row in result.head(5).iterrows():
                numeric_summary = ", ".join(f"{col}={_fmt(row[col])}" for col in numeric_cols[:3])
                lines.append(f"- {_row_label(row.to_dict(), result)}: {numeric_summary}")

    return "\n".join(lines)


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
    coverage_terms = ("cover", "covering", "covered", "available", "list", "show", "what")
    subject_terms = ("desk", "desks", "book", "books", "portfolio", "portfolios", "product", "products", "currency", "currencies")
    return any(term in lowered for term in coverage_terms) and any(term in lowered for term in subject_terms)


def _coverage_query_plan(question: str, lowered: str) -> QueryPlan:
    if "book" in lowered and "desk" not in lowered:
        sql = """
SELECT
  desk,
  book,
  portfolio,
  product,
  currency
FROM hierarchy
GROUP BY desk, book, portfolio, product, currency
ORDER BY desk, book, portfolio
LIMIT 50
"""
        return QueryPlan(question, sql, "book_coverage", "METADATA", "table", False, "Rule fallback generated book coverage SQL.")

    sql = """
SELECT
  desk,
  COUNT(DISTINCT book) AS book_count,
  COUNT(DISTINCT portfolio) AS portfolio_count,
  GROUP_CONCAT(DISTINCT book) AS books,
  GROUP_CONCAT(DISTINCT product) AS products,
  GROUP_CONCAT(DISTINCT currency) AS currencies
FROM hierarchy
GROUP BY desk
ORDER BY desk
LIMIT 50
"""
    return QueryPlan(question, sql, "desk_coverage", "METADATA", "table", False, "Rule fallback generated desk coverage SQL.")


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _extract_top_n(lowered: str) -> int:
    top_match = re.search(r"\btop\s+(\d+)\b", lowered)
    if top_match:
        return max(1, min(25, int(top_match.group(1))))
    show_match = re.search(r"\bshow\s+(\d+)\s+(?![- ]?day\b)", lowered)
    if show_match:
        return max(1, min(25, int(show_match.group(1))))
    return 10 if "drill" in lowered else 5


def _extract_days(lowered: str) -> int:
    match = re.search(r"\b(\d+)\s*[- ]?day\b", lowered)
    if not match:
        return 10
    return max(2, min(60, int(match.group(1))))


def _metric_from_text(current_lower: str, context_lower: str) -> str:
    if _mentions_pnl(current_lower):
        return "PNL"
    if _mentions_var(current_lower):
        return "VAR"
    if _mentions_pnl(context_lower):
        return "PNL"
    if _mentions_var(context_lower):
        return "VAR"
    return "VAR"


def _mentions_pnl(text: str) -> bool:
    return bool(re.search(r"\bp[\s&-]?n[\s&-]?l\b|\bpnl\b|profit", text))


def _mentions_var(text: str) -> bool:
    return bool(re.search(r"\bvar\b|value at risk", text))


def _has_explicit_intent(lowered: str) -> bool:
    intent_terms = (
        "market move",
        "market moves",
        "trend",
        "day",
        "risk factor",
        "risk-factor",
        "scenario",
        "drill",
        "move",
        "movement",
        "why",
        "explain",
        "driver",
    )
    return any(term in lowered for term in intent_terms)


def _latest_comparison_dates(db_path: Path) -> tuple[str, str]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT date FROM hierarchy ORDER BY date").fetchall()
    dates = [str(row[0]) for row in rows]
    if len(dates) < 2:
        raise ValueError("At least two dates are required for comparison questions")
    return dates[-2], dates[-1]


def _match_choice(lowered: str, choices: list[str]) -> str | None:
    for choice in sorted(choices, key=len, reverse=True):
        if choice.lower() in lowered:
            return choice
    compact_query = re.sub(r"[^a-z0-9]", "", lowered)
    for choice in sorted(choices, key=len, reverse=True):
        if re.sub(r"[^a-z0-9]", "", choice.lower()) in compact_query:
            return choice
    return None


def _scope_filters(desk: str | None, book: str | None) -> str:
    filters = []
    if desk:
        filters.append(f"desk = {_quote(desk)}")
    if book:
        filters.append(f"book = {_quote(book)}")
    return " AND " + " AND ".join(filters) if filters else ""


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _first_label_column(result: pd.DataFrame) -> str:
    for col in ("scenario", "risk_factor", "desk", "book", "portfolio", "date"):
        if col in result.columns:
            return col
    return result.columns[0]


def _row_label(row: dict[str, Any], result: pd.DataFrame) -> str:
    if {"risk_factor", "scenario"}.issubset(result.columns):
        return f"{row.get('risk_factor')} / {row.get('scenario')}"
    label_col = _first_label_column(result)
    return str(row.get(label_col, "row"))


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)
