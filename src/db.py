from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .data_generator import DATA_DIR, ensure_sample_data


DB_PATH = DATA_DIR / "risk_explain.db"

EXPECTED_FILES = {
    "trade_sensitivities": "trade_sensitivities.csv",
    "risk_factor_scenarios": "risk_factor_scenarios.csv",
}

LEGACY_TABLES = (
    "hierarchy",
    "pnl_results",
    "var_results",
    "sensitivities",
    "market_data",
    "scenario_data",
)

SCHEMA = {
    "trade_sensitivities": """
        CREATE TABLE IF NOT EXISTS trade_sensitivities (
            trade_id TEXT NOT NULL,
            risk_factor TEXT NOT NULL,
            desk TEXT NOT NULL,
            sensitivity_type TEXT NOT NULL,
            sensitivity_value REAL NOT NULL,
            product TEXT NOT NULL
        )
    """,
    "risk_factor_scenarios": """
        CREATE TABLE IF NOT EXISTS risk_factor_scenarios (
            historical_date TEXT NOT NULL,
            scenario_name TEXT NOT NULL,
            risk_factor TEXT NOT NULL,
            shock_value REAL NOT NULL,
            shock_unit TEXT NOT NULL
        )
    """,
}

REQUIRED_COLUMNS = {
    "trade_sensitivities": ["trade_id", "risk_factor", "desk", "sensitivity_type", "sensitivity_value", "product"],
    "risk_factor_scenarios": ["historical_date", "scenario_name", "risk_factor", "shock_value", "shock_unit"],
}


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: Path = DB_PATH, reset: bool = False) -> None:
    with get_connection(db_path) as conn:
        if reset:
            for table in (*EXPECTED_FILES.keys(), *LEGACY_TABLES):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        for ddl in SCHEMA.values():
            conn.execute(ddl)
        _create_indexes(conn)
        conn.commit()


def load_csvs_to_sqlite(data_dir: Path = DATA_DIR, db_path: Path = DB_PATH, reset: bool = False) -> dict[str, int]:
    ensure_sample_data(data_dir)
    initialize_database(db_path, reset=reset)
    counts: dict[str, int] = {}

    with get_connection(db_path) as conn:
        for table, filename in EXPECTED_FILES.items():
            path = data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing required CSV: {path}")
            frame = pd.read_csv(path)
            missing = set(REQUIRED_COLUMNS[table]) - set(frame.columns)
            if missing:
                raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")
            frame = frame[REQUIRED_COLUMNS[table]]
            conn.execute(f"DELETE FROM {table}")
            frame.to_sql(table, conn, if_exists="append", index=False)
            counts[table] = len(frame)
        _create_indexes(conn)
        conn.commit()
    return counts


def bootstrap_database(data_dir: Path = DATA_DIR, db_path: Path = DB_PATH) -> dict[str, int]:
    ensure_sample_data(data_dir)
    if not db_path.exists():
        return load_csvs_to_sqlite(data_dir=data_dir, db_path=db_path, reset=True)
    counts = table_counts(db_path)
    if any(count == 0 for count in counts.values()):
        return load_csvs_to_sqlite(data_dir=data_dir, db_path=db_path, reset=True)
    return counts


def table_counts(db_path: Path = DB_PATH) -> dict[str, int]:
    initialize_database(db_path, reset=False)
    counts: dict[str, int] = {}
    with get_connection(db_path) as conn:
        for table in EXPECTED_FILES:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return counts


def distinct_values(column: str, table: str = "trade_sensitivities", db_path: Path = DB_PATH) -> list[str]:
    allowed = {"trade_id", "desk", "product", "risk_factor", "sensitivity_type", "historical_date", "scenario_name", "shock_unit"}
    if column not in allowed:
        raise ValueError(f"Unsupported distinct column: {column}")
    if table not in EXPECTED_FILES:
        raise ValueError(f"Unsupported table: {table}")
    with get_connection(db_path) as conn:
        rows = conn.execute(f"SELECT DISTINCT {column} FROM {table} ORDER BY {column}").fetchall()
    return [str(row[0]) for row in rows]


def save_uploaded_csv(filename: str, content: bytes, data_dir: Path = DATA_DIR) -> Path:
    allowed = set(EXPECTED_FILES.values())
    if filename not in allowed:
        raise ValueError(f"Unexpected file {filename}. Expected one of: {sorted(allowed)}")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    path.write_bytes(content)
    return path


def _create_indexes(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_scope ON trade_sensitivities(desk, product, trade_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_factor ON trade_sensitivities(risk_factor)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scenario_factor ON risk_factor_scenarios(historical_date, risk_factor)")
