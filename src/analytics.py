from __future__ import annotations

from pathlib import Path

import pandas as pd

from .db import DB_PATH, get_connection


def compare_pnl(date1: str, date2: str, desk: str | None = None, book: str | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    pnl_date_1 = _metric_total("pnl_results", "pnl_value", date1, desk, book, db_path)
    pnl_date_2 = _metric_total("pnl_results", "pnl_value", date2, desk, book, db_path)
    pnl_change = pnl_date_2 - pnl_date_1
    explained_pnl = float(explain_pnl_by_risk_factor(date1, date2, desk, book, db_path)["estimated_pnl_impact"].sum())
    residual_pnl = pnl_change - explained_pnl
    return pd.DataFrame(
        [
            {
                "previous_date": date1,
                "current_date": date2,
                "pnl_date_1": round(pnl_date_1, 4),
                "pnl_date_2": round(pnl_date_2, 4),
                "pnl_change": round(pnl_change, 4),
                "explained_pnl": round(explained_pnl, 4),
                "residual_pnl": round(residual_pnl, 4),
            }
        ]
    )


def explain_pnl_by_risk_factor(
    date1: str,
    date2: str,
    desk: str | None = None,
    book: str | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    filters, params = _scope_filter("s", desk, book)
    sql = f"""
    WITH sens AS (
      SELECT
        risk_factor,
        sensitivity_type,
        SUM(sensitivity_value) AS sensitivity_value
      FROM sensitivities s
      WHERE s.date = ?{filters}
      GROUP BY risk_factor, sensitivity_type
    ),
    moves AS (
      SELECT
        risk_factor,
        SUM(market_move) AS market_move,
        MAX(move_unit) AS move_unit
      FROM market_data
      WHERE date > ? AND date <= ?
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
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=[date1, *params, date1, date2])


def compare_var(date1: str, date2: str, desk: str | None = None, book: str | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    var_date_1 = _metric_total("var_results", "var_contribution", date1, desk, book, db_path)
    var_date_2 = _metric_total("var_results", "var_contribution", date2, desk, book, db_path)
    var_change = var_date_2 - var_date_1
    percentage_change = (var_change / var_date_1 * 100.0) if var_date_1 else 0.0
    return pd.DataFrame(
        [
            {
                "previous_date": date1,
                "current_date": date2,
                "var_date_1": round(var_date_1, 4),
                "var_date_2": round(var_date_2, 4),
                "var_change": round(var_change, 4),
                "percentage_change": round(percentage_change, 4),
            }
        ]
    )


def explain_var_by_scenario(
    date1: str,
    date2: str,
    desk: str | None = None,
    book: str | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    return _var_driver(date1, date2, "scenario", desk, book, db_path)


def explain_var_by_risk_factor(
    date1: str,
    date2: str,
    desk: str | None = None,
    book: str | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    return _var_driver(date1, date2, "risk_factor", desk, book, db_path)


def trend_analysis(
    metric: str,
    desk: str | None = None,
    book: str | None = None,
    days: int = 10,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    metric_upper = metric.upper()
    if metric_upper == "PNL":
        table, value_col = "pnl_results", "pnl_value"
    elif metric_upper == "VAR":
        table, value_col = "var_results", "var_contribution"
    else:
        raise ValueError("metric must be PNL or VAR")
    filters, params = _scope_filter("", desk, book)
    sql = f"""
    SELECT date, ROUND(value, 4) AS value
    FROM (
      SELECT date, SUM({value_col}) AS value
      FROM {table}
      WHERE 1 = 1{filters}
      GROUP BY date
      ORDER BY date DESC
      LIMIT ?
    )
    ORDER BY date
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=[*params, days])


def drilldown(
    metric: str,
    date1: str,
    date2: str,
    desk: str | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    metric_upper = metric.upper()
    if metric_upper == "PNL":
        filters, params = _scope_filter("s", desk, None)
        sql = f"""
        WITH sens AS (
          SELECT
            s.desk,
            s.book,
            s.portfolio,
            s.product,
            'Actual market move' AS scenario,
            s.risk_factor,
            SUM(s.sensitivity_value) AS sensitivity_value
          FROM sensitivities s
          WHERE s.date = ?{filters}
          GROUP BY s.desk, s.book, s.portfolio, s.product, s.risk_factor
        ),
        moves AS (
          SELECT risk_factor, SUM(market_move) AS market_move
          FROM market_data
          WHERE date > ? AND date <= ?
          GROUP BY risk_factor
        )
        SELECT
          sens.desk,
          sens.book,
          sens.portfolio,
          sens.product,
          sens.scenario,
          sens.risk_factor,
          ROUND(sens.sensitivity_value * COALESCE(moves.market_move, 0), 4) AS estimated_pnl_impact
        FROM sens
        LEFT JOIN moves ON moves.risk_factor = sens.risk_factor
        ORDER BY ABS(estimated_pnl_impact) DESC
        """
        with get_connection(db_path) as conn:
            return pd.read_sql_query(sql, conn, params=[date1, *params, date1, date2])

    if metric_upper != "VAR":
        raise ValueError("metric must be PNL or VAR")
    filters, params = _scope_filter("", desk, None)
    sql = f"""
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
      WHERE date IN (?, ?){filters}
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
        SUM(CASE WHEN date = ? THEN value ELSE 0 END) AS previous_value,
        SUM(CASE WHEN date = ? THEN value ELSE 0 END) AS current_value
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
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=[date1, date2, *params, date1, date2])


def _metric_total(
    table: str,
    value_col: str,
    date: str,
    desk: str | None,
    book: str | None,
    db_path: Path,
) -> float:
    filters, params = _scope_filter("", desk, book)
    with get_connection(db_path) as conn:
        value = conn.execute(
            f"SELECT COALESCE(SUM({value_col}), 0) FROM {table} WHERE date = ?{filters}",
            [date, *params],
        ).fetchone()[0]
    return float(value or 0)


def _var_driver(
    date1: str,
    date2: str,
    group_col: str,
    desk: str | None,
    book: str | None,
    db_path: Path,
) -> pd.DataFrame:
    if group_col not in {"scenario", "risk_factor"}:
        raise ValueError("Unsupported VaR driver column")
    filters, params = _scope_filter("", desk, book)
    sql = f"""
    WITH driver AS (
      SELECT date, {group_col}, SUM(var_contribution) AS value
      FROM var_results
      WHERE date IN (?, ?){filters}
      GROUP BY date, {group_col}
    ),
    pivot AS (
      SELECT
        {group_col},
        SUM(CASE WHEN date = ? THEN value ELSE 0 END) AS previous_value,
        SUM(CASE WHEN date = ? THEN value ELSE 0 END) AS current_value
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
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=[date1, date2, *params, date1, date2])


def _scope_filter(alias: str, desk: str | None, book: str | None) -> tuple[str, list[str]]:
    prefix = f"{alias}." if alias else ""
    filters: list[str] = []
    params: list[str] = []
    if desk:
        filters.append(f"{prefix}desk = ?")
        params.append(desk)
    if book:
        filters.append(f"{prefix}book = ?")
        params.append(book)
    return (" AND " + " AND ".join(filters) if filters else ""), params
