from __future__ import annotations

from pathlib import Path

import pandas as pd

from .db import DB_PATH, get_connection


CONFIDENCE_LEVEL = 0.95


def explain_trade_pnl(
    desk: str | None = None,
    trade_id: str | None = None,
    product: str | None = None,
    historical_date: str | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    filters, params = _scope_filter("ts", desk=desk, trade_id=trade_id, product=product)
    date_filter = ""
    if historical_date:
        date_filter = " AND rs.historical_date = ?"
        params.append(historical_date)
    sql = f"""
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
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def aggregate_scenario_pnl(
    desk: str | None = None,
    product: str | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    filters, params = _scope_filter("ts", desk=desk, product=product)
    sql = f"""
    SELECT
      rs.historical_date,
      rs.scenario_name,
      ROUND(SUM(ts.sensitivity_value * rs.shock_value), 4) AS scenario_pnl,
      ROUND(CASE WHEN SUM(ts.sensitivity_value * rs.shock_value) < 0
        THEN -SUM(ts.sensitivity_value * rs.shock_value)
        ELSE 0
      END, 4) AS loss_amount
    FROM trade_sensitivities ts
    JOIN risk_factor_scenarios rs
      ON rs.risk_factor = ts.risk_factor
    WHERE 1 = 1{filters}
    GROUP BY rs.historical_date, rs.scenario_name
    ORDER BY rs.historical_date
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def calculate_var(
    desk: str | None = None,
    product: str | None = None,
    confidence_level: float = CONFIDENCE_LEVEL,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    pnl = aggregate_scenario_pnl(desk=desk, product=product, db_path=db_path)
    if pnl.empty:
        return pd.DataFrame(columns=["metric", "confidence_level", "historical_date", "scenario_pnl", "var_95", "scenario_count"])
    ranked = pnl.sort_values("loss_amount", ascending=True).reset_index(drop=True)
    rank_idx = max(0, min(len(ranked) - 1, int((len(ranked) * confidence_level + 0.999999) - 1)))
    row = ranked.iloc[rank_idx]
    return pd.DataFrame(
        [
            {
                "metric": "VAR",
                "confidence_level": confidence_level,
                "historical_date": row["historical_date"],
                "scenario_pnl": round(float(row["scenario_pnl"]), 4),
                "var_95": round(float(row["loss_amount"]), 4),
                "scenario_count": len(ranked),
            }
        ]
    )


def explain_var_by_risk_factor(
    desk: str | None = None,
    product: str | None = None,
    confidence_level: float = CONFIDENCE_LEVEL,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    var = calculate_var(desk=desk, product=product, confidence_level=confidence_level, db_path=db_path)
    if var.empty:
        return pd.DataFrame()
    historical_date = str(var.iloc[0]["historical_date"])
    pnl = explain_trade_pnl(desk=desk, product=product, historical_date=historical_date, db_path=db_path)
    return (
        pnl.groupby(["risk_factor", "sensitivity_type", "shock_unit"], as_index=False)
        .agg(sensitivity_value=("sensitivity_value", "sum"), shock_value=("shock_value", "first"), driver_pnl=("scenario_pnl", "sum"))
        .assign(abs_driver=lambda frame: frame["driver_pnl"].abs())
        .sort_values("abs_driver", ascending=False)
        .drop(columns=["abs_driver"])
        .reset_index(drop=True)
    )


def explain_var_by_trade(
    desk: str | None = None,
    product: str | None = None,
    confidence_level: float = CONFIDENCE_LEVEL,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    var = calculate_var(desk=desk, product=product, confidence_level=confidence_level, db_path=db_path)
    if var.empty:
        return pd.DataFrame()
    historical_date = str(var.iloc[0]["historical_date"])
    pnl = explain_trade_pnl(desk=desk, product=product, historical_date=historical_date, db_path=db_path)
    return (
        pnl.groupby(["trade_id", "desk", "product"], as_index=False)
        .agg(driver_pnl=("scenario_pnl", "sum"))
        .assign(abs_driver=lambda frame: frame["driver_pnl"].abs())
        .sort_values("abs_driver", ascending=False)
        .drop(columns=["abs_driver"])
        .reset_index(drop=True)
    )


def trend_analysis(metric: str = "PNL", desk: str | None = None, product: str | None = None, days: int = 10, db_path: Path = DB_PATH) -> pd.DataFrame:
    pnl = aggregate_scenario_pnl(desk=desk, product=product, db_path=db_path)
    if pnl.empty:
        return pd.DataFrame(columns=["historical_date", "value"])
    value_col = "loss_amount" if metric.upper() == "VAR" else "scenario_pnl"
    return pnl.tail(days)[["historical_date", value_col]].rename(columns={value_col: "value"}).reset_index(drop=True)


def coverage(db_path: Path = DB_PATH) -> pd.DataFrame:
    sql = """
    SELECT
      desk,
      COUNT(DISTINCT trade_id) AS trade_count,
      COUNT(DISTINCT product) AS product_count,
      COUNT(DISTINCT risk_factor) AS risk_factor_count,
      GROUP_CONCAT(DISTINCT product) AS products
    FROM trade_sensitivities
    GROUP BY desk
    ORDER BY desk
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def _scope_filter(
    alias: str,
    desk: str | None = None,
    trade_id: str | None = None,
    product: str | None = None,
) -> tuple[str, list[str]]:
    prefix = f"{alias}." if alias else ""
    filters: list[str] = []
    params: list[str] = []
    if desk:
        filters.append(f"{prefix}desk = ?")
        params.append(desk)
    if trade_id:
        filters.append(f"{prefix}trade_id = ?")
        params.append(trade_id)
    if product:
        filters.append(f"{prefix}product = ?")
        params.append(product)
    return (" AND " + " AND ".join(filters) if filters else ""), params
