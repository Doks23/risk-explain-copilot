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
- trade_sensitivities(trade_id, risk_factor, desk, sensitivity_type, sensitivity_value, product)
- risk_factor_scenarios(historical_date, scenario_name, risk_factor, shock_value, shock_unit)

Business rules:
- These are the two stored input tables.
- Trade-level scenario P&L is computed, not stored: scenario_pnl = sensitivity_value * shock_value.
- Aggregate P&L by historical_date to create the scenario P&L distribution.
- 95% VaR is the 95th percentile of scenario losses, where loss_amount = max(-scenario_pnl, 0).
- Aggregation is essential: desk/product VaR must aggregate scenario P&L by historical_date before calculating percentile.
- Risk-factor VaR drivers are the risk-factor P&L contributions in the VaR scenario date.
- Do not treat sensitivity as P&L. Do not treat shocks as P&L. P&L requires sensitivity times shock.

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
                    "Explain the chain Trade sensitivity x risk-factor shock = scenario P&L; "
                    "scenario P&Ls aggregate by historical date; 95% VaR is the percentile loss from that distribution. "
                    "Do not include SQL, raw JSON, Evidence from SQL, or Data trace in the visible answer."
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
    filters = _scope_filters(desk=desk, product=product, trade_id=trade_id, risk_factor=risk_factor)

    metric_context = planning_lower
    if not _mentions_var(lowered) and not _mentions_pnl(lowered) and (_mentions_var(context_lower) or _mentions_pnl(context_lower)):
        metric_context = f"{planning_lower}\n{context_lower}"

    if _is_coverage_question(lowered) and not _mentions_pnl(lowered) and not _mentions_var(lowered):
        return QueryPlan(question, _coverage_sql(), "coverage", "METADATA", "table", False, "Rule fallback generated coverage SQL.")

    if "risk factor" in planning_lower and ("scenario" in planning_lower or "shock" in planning_lower) and "pnl" not in planning_lower and "p&l" not in planning_lower:
        return QueryPlan(question, _scenario_shock_sql(risk_factor, historical_date, top_n), "scenario_shocks", "SHOCK", "table", False, "Rule fallback generated RF scenario SQL.")

    if "trend" in planning_lower or re.search(r"\b\d+\s*[- ]?day\b", planning_lower):
        metric = "VAR" if _mentions_var(planning_lower) else "PNL"
        return QueryPlan(question, _trend_sql(filters, days, metric), "trend", metric, "line", False, "Rule fallback generated scenario P&L trend SQL.")

    if _mentions_var(metric_context):
        if "risk factor" in planning_lower or "driver" in planning_lower or "explain" in planning_lower:
            return QueryPlan(question, _var_driver_sql(filters, top_n), "var_risk_factor_drivers", "VAR", "bar", False, "Rule fallback generated VaR driver SQL.")
        return QueryPlan(question, _var_sql(filters), "var", "VAR", "metric", False, "Rule fallback generated VaR percentile SQL.")

    return QueryPlan(question, _trade_pnl_sql(filters, historical_date, top_n), "trade_scenario_pnl", "PNL", "bar", False, "Rule fallback generated trade scenario P&L SQL.")


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


def _trade_pnl_sql(filters: str, historical_date: str | None, top_n: int) -> str:
    date_filter = f" AND rs.historical_date = {_quote(historical_date)}" if historical_date else ""
    return f"""
SELECT
  ts.trade_id,
  ts.desk,
  ts.product,
  ts.risk_factor,
  ts.sensitivity_type,
  ROUND(ts.sensitivity_value, 4) AS sensitivity_value,
  rs.historical_date,
  rs.scenario_name,
  ROUND(rs.shock_value, 4) AS shock_value,
  rs.shock_unit,
  ROUND(ts.sensitivity_value * rs.shock_value, 4) AS scenario_pnl
FROM trade_sensitivities ts
JOIN risk_factor_scenarios rs
  ON rs.risk_factor = ts.risk_factor
WHERE 1 = 1{filters}{date_filter}
ORDER BY ABS(scenario_pnl) DESC
LIMIT {top_n}
"""


def _portfolio_pnl_cte(filters: str) -> str:
    return f"""
trade_pnl AS (
  SELECT
    rs.historical_date,
    rs.scenario_name,
    ts.trade_id,
    ts.desk,
    ts.product,
    ts.risk_factor,
    ts.sensitivity_type,
    ts.sensitivity_value,
    rs.shock_value,
    rs.shock_unit,
    ts.sensitivity_value * rs.shock_value AS scenario_pnl
  FROM trade_sensitivities ts
  JOIN risk_factor_scenarios rs
    ON rs.risk_factor = ts.risk_factor
  WHERE 1 = 1{filters}
),
pnl_by_scenario AS (
  SELECT
    historical_date,
    scenario_name,
    SUM(scenario_pnl) AS scenario_pnl
  FROM trade_pnl
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


def _var_sql(filters: str) -> str:
    return f"""
WITH {_portfolio_pnl_cte(filters)}
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


def _var_driver_sql(filters: str, top_n: int) -> str:
    return f"""
WITH {_portfolio_pnl_cte(filters)}
SELECT
  trade_pnl.risk_factor,
  trade_pnl.sensitivity_type,
  ROUND(SUM(trade_pnl.sensitivity_value), 4) AS sensitivity_value,
  ROUND(MAX(trade_pnl.shock_value), 4) AS shock_value,
  MAX(trade_pnl.shock_unit) AS shock_unit,
  ROUND(SUM(trade_pnl.scenario_pnl), 4) AS driver_pnl,
  var_scenario.historical_date AS var_scenario_date
FROM trade_pnl
JOIN var_scenario
  ON var_scenario.historical_date = trade_pnl.historical_date
GROUP BY trade_pnl.risk_factor, trade_pnl.sensitivity_type, var_scenario.historical_date
ORDER BY ABS(driver_pnl) DESC
LIMIT {top_n}
"""


def _trend_sql(filters: str, days: int, metric: str) -> str:
    value_expr = "loss_amount" if metric == "VAR" else "scenario_pnl"
    return f"""
WITH {_portfolio_pnl_cte(filters)}
SELECT
  historical_date AS date,
  ROUND({value_expr}, 4) AS value
FROM losses
ORDER BY historical_date DESC
LIMIT {days}
"""


def _fallback_response(question: str, plan: QueryPlan, result: pd.DataFrame) -> str:
    if result.empty:
        return "### Answer\nThe query returned no rows, so there is not enough data to answer from the available result."

    first = result.iloc[0].to_dict()
    lines = ["### Answer"]

    if plan.intent == "coverage" and "desk" in result.columns:
        lines.append(f"We cover {result['desk'].nunique()} desks across {int(result['trade_count'].sum())} trades.")
        lines.append("\n### Covered desks")
        for _, row in result.iterrows():
            lines.append(f"- {row['desk']}: {int(row['trade_count'])} trades; products: {row['products']}")
    elif {"var_95", "scenario_pnl", "confidence_level"}.issubset(result.columns):
        lines.append(
            f"The {_fmt(first['confidence_level'])} confidence VaR is {_fmt(first['var_95'])}. "
            f"It comes from historical scenario {first['historical_date']} where aggregate scenario P&L was {_fmt(first['scenario_pnl'])}."
        )
        lines.append("This is computed from trade sensitivities times risk-factor shocks, aggregated by historical date, then percentile-ranked as a loss distribution.")
    elif "driver_pnl" in result.columns:
        lines.append(
            f"The largest VaR-scenario driver is {first.get('risk_factor')}: driver P&L {_fmt(first['driver_pnl'])} "
            f"from sensitivity {_fmt(first.get('sensitivity_value', 0))} and shock {_fmt(first.get('shock_value', 0))} {first.get('shock_unit', '')}."
        )
    elif "scenario_pnl" in result.columns:
        lines.append(
            f"The largest returned trade scenario P&L is trade {first.get('trade_id')} / {first.get('risk_factor')}: "
            f"{_fmt(first['scenario_pnl'])}, calculated as sensitivity {_fmt(first.get('sensitivity_value', 0))} "
            f"x shock {_fmt(first.get('shock_value', 0))} {first.get('shock_unit', '')}."
        )
    elif {"date", "value"}.issubset(result.columns):
        ordered = result.sort_values("date")
        start = ordered.iloc[0]
        end = ordered.iloc[-1]
        change = float(end["value"]) - float(start["value"])
        lines.append(f"The returned scenario trend moved by {_fmt(change)}, from {_fmt(start['value'])} on {start['date']} to {_fmt(end['value'])} on {end['date']}.")
    elif "shock_value" in result.columns:
        lines.append(f"The query returned {len(result)} historical risk-factor shock rows. The first row is {first.get('risk_factor')} on {first.get('historical_date')}: {_fmt(first.get('shock_value'))} {first.get('shock_unit')}.")
    else:
        lines.append(f"The SQL returned {len(result)} row(s). The top row is: {first}.")

    numeric_cols = [col for col in result.columns if pd.api.types.is_numeric_dtype(result[col])]
    if plan.intent not in {"coverage", "trend"}:
        lines.append("\n### Key metrics")
        for col in numeric_cols[:6]:
            lines.append(f"- {col}: {_fmt(first[col])} on top returned row")

        if len(result) > 1:
            lines.append("\n### Top returned rows")
            for _, row in result.head(5).iterrows():
                label = row.get("risk_factor", row.get("trade_id", row.get("historical_date", "row")))
                summary = ", ".join(f"{col}={_fmt(row[col])}" for col in numeric_cols[:3])
                lines.append(f"- {label}: {summary}")

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
) -> str:
    filters = []
    if desk:
        filters.append(f"ts.desk = {_quote(desk)}")
    if product:
        filters.append(f"ts.product = {_quote(product)}")
    if trade_id:
        filters.append(f"ts.trade_id = {_quote(trade_id)}")
    if risk_factor:
        filters.append(f"ts.risk_factor = {_quote(risk_factor)}")
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
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)
