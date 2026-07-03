from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    title: str
    text: str
    tags: tuple[str, ...]


KNOWLEDGE_CHUNKS: tuple[KnowledgeChunk, ...] = (
    KnowledgeChunk(
        "schema_var_results",
        "VaR results table",
        (
            "Use var_results for Value at Risk or VaR questions. "
            "The table is at date, desk, book, portfolio, product, scenario, and risk_factor grain. "
            "VaR movement is current date SUM(var_contribution) minus previous date SUM(var_contribution)."
        ),
        ("schema", "var", "movement"),
    ),
    KnowledgeChunk(
        "schema_pnl_results",
        "PNL results table",
        (
            "Use pnl_results for actual or hypothetical PNL level questions. "
            "The table is at date, desk, book, portfolio, and product grain. "
            "Actual PNL movement is current date SUM(pnl_value) minus previous date SUM(pnl_value)."
        ),
        ("schema", "pnl", "movement"),
    ),
    KnowledgeChunk(
        "schema_sensitivities",
        "Sensitivities table",
        (
            "Use sensitivities for exposure. Sensitivity is not final PNL. "
            "DV01 explains rate exposure, Delta explains equity or FX spot exposure, Vega explains volatility exposure, "
            "and CS01 explains credit spread exposure."
        ),
        ("schema", "sensitivities", "exposure"),
    ),
    KnowledgeChunk(
        "schema_market_data",
        "Market data table",
        (
            "Use market_data for actual market moves by risk_factor. "
            "Market move alone does not explain PNL; combine it with sensitivity. "
            "For date1 to date2 attribution, use moves where date is greater than date1 and less than or equal to date2."
        ),
        ("schema", "market_data", "market_moves"),
    ),
    KnowledgeChunk(
        "schema_scenario_data",
        "Scenario data table",
        (
            "Use scenario_data for scenario shocks by date, scenario, and risk_factor. "
            "It provides shock values and units that support scenario interpretation, while VaR contribution movement comes from var_results."
        ),
        ("schema", "scenario", "shock"),
    ),
    KnowledgeChunk(
        "schema_hierarchy",
        "Hierarchy table",
        (
            "Use hierarchy for coverage questions such as what desks, books, portfolios, products, or currencies are covered. "
            "The hierarchy table maps date, desk, book, portfolio, product, and currency. "
            "Coverage questions should not default to VaR or PNL movement."
        ),
        ("schema", "coverage", "desk", "book"),
    ),
    KnowledgeChunk(
        "pnl_attribution",
        "PNL attribution logic",
        (
            "PNL should be explained as estimated_pnl_impact = sensitivity_value from date1 multiplied by market_move between date1 and date2. "
            "Actual PNL change comes from pnl_results. Explained PNL is the sum of estimated_pnl_impact. "
            "Residual PNL equals actual PNL change minus explained PNL."
        ),
        ("pnl", "drivers", "sensitivities", "market_moves"),
    ),
    KnowledgeChunk(
        "market_move_explanation",
        "Market move explanation",
        (
            "For questions asking what market moves explain PNL, join sensitivities from date1 to market_data moves between date1 and date2. "
            "Rank risk factors by absolute estimated_pnl_impact. "
            "Do not use raw market moves as the final answer without the sensitivity exposure."
        ),
        ("market_moves", "pnl", "sensitivities", "drivers"),
    ),
    KnowledgeChunk(
        "var_attribution",
        "VaR attribution logic",
        (
            "VaR is a risk estimate, not an actual loss. "
            "Explain VaR movement by comparing var_contribution between two dates and ranking scenario or risk_factor deltas by absolute movement. "
            "Do not explain VaR directly using sensitivity times market move."
        ),
        ("var", "drivers", "scenario", "risk_factor"),
    ),
    KnowledgeChunk(
        "trend_analysis",
        "Trend analysis",
        (
            "Trend questions should aggregate SUM(var_contribution) for VaR or SUM(pnl_value) for PNL by date. "
            "For a 10-day trend, select the latest 10 business dates and order ascending."
        ),
        ("trend", "time_series", "var", "pnl"),
    ),
    KnowledgeChunk(
        "sql_safety",
        "SQL safety policy",
        (
            "Generated SQL must be read-only SQLite SELECT or WITH statements. "
            "Do not use SELECT star. Do not expose full database tables. "
            "Always aggregate or filter to the question and return a bounded result."
        ),
        ("safety", "sql"),
    ),
    KnowledgeChunk(
        "response_policy",
        "Response policy",
        (
            "Final answers must use only the rows returned by the executed SQL. "
            "Do not invent numbers. If a question needs data that was not returned, say what is missing."
        ),
        ("response", "grounding"),
    ),
    KnowledgeChunk(
        "desk_aliases",
        "Desk and book examples",
        (
            "Known desks include London Rates, FX Options, Credit Trading, Equity Derivatives, and Treasury. "
            "Each desk has three books and multiple portfolios/products."
        ),
        ("desk", "book", "aliases"),
    ),
)
