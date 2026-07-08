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
        "risk_chain",
        "Risk chain",
        (
            "The core chain is Risk Factor -> Trade Sensitivity -> Historical Shock/Scenario -> Scenario P&L -> P&L Distribution -> VaR. "
            "A risk factor is what can move; sensitivity is exposure; shock is how much it moves; P&L is sensitivity times shock; VaR is a percentile loss from aggregated scenario P&Ls."
        ),
        ("risk_factor", "sensitivity", "shock", "pnl", "var"),
    ),
    KnowledgeChunk(
        "schema_trade_sensitivities",
        "Trade sensitivity table",
        (
            "trade_sensitivities is the trade-level input table with trade_id, risk_factor, desk, sensitivity_type, sensitivity_value, and product. "
            "It stores exposure, not P&L."
        ),
        ("schema", "trade", "sensitivity"),
    ),
    KnowledgeChunk(
        "schema_risk_factor_scenarios",
        "Risk-factor scenario table",
        (
            "risk_factor_scenarios is the historical risk-factor shock table with historical_date, scenario_name, risk_factor, shock_value, and shock_unit. "
            "Each historical date is one scenario containing shocks for many risk factors."
        ),
        ("schema", "scenario", "shock"),
    ),
    KnowledgeChunk(
        "scenario_pnl_logic",
        "Scenario P&L logic",
        (
            "Trade scenario P&L is computed by joining trade_sensitivities to risk_factor_scenarios on risk_factor. "
            "scenario_pnl = sensitivity_value * shock_value. "
            "This can be shown by trade, risk factor, desk, product, and historical scenario date."
        ),
        ("pnl", "scenario", "calculation"),
    ),
    KnowledgeChunk(
        "aggregation_logic",
        "Aggregation logic",
        (
            "Aggregation is essential for VaR. First compute trade/risk-factor scenario P&L, then aggregate scenario_pnl by historical_date across the selected desk/product/trades. "
            "This creates the portfolio scenario P&L distribution."
        ),
        ("aggregation", "pnl", "var"),
    ),
    KnowledgeChunk(
        "var_logic",
        "VaR percentile logic",
        (
            "VaR is computed from the aggregated scenario P&L distribution. "
            "For this prototype, loss_amount = max(-scenario_pnl, 0), and 95% VaR is the 95th percentile loss across historical scenario dates. "
            "VaR is not actual P&L and is not a maximum possible loss."
        ),
        ("var", "percentile", "loss"),
    ),
    KnowledgeChunk(
        "var_drivers",
        "VaR driver logic",
        (
            "Risk-factor VaR drivers are the risk-factor P&L contributions on the historical scenario date selected by the 95% VaR percentile. "
            "Rank drivers by absolute driver_pnl."
        ),
        ("var", "drivers", "risk_factor"),
    ),
    KnowledgeChunk(
        "coverage",
        "Coverage questions",
        (
            "Coverage questions should use trade_sensitivities to list desks, trades, products, and risk factors. "
            "Do not default coverage questions to VaR or P&L calculations."
        ),
        ("coverage", "desk", "trade", "product"),
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
)
