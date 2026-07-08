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
            "The core chain is Risk Factor -> Trade Sensitivity -> Historical Shock/Scenario -> Trade P&L -> P&L Distribution -> VaR. "
            "A risk factor is what can move; sensitivity is exposure; shock is how much it moves; trade_pnl is the actual "
            "trade-level financial impact on a historical scenario date; VaR is a percentile loss from the aggregated P&L distribution. "
            "Sensitivity times shock is used separately to attribute/explain what drove P&L on a specific date, not to compute the P&L itself."
        ),
        ("risk_factor", "sensitivity", "shock", "pnl", "var"),
    ),
    KnowledgeChunk(
        "schema_trade_sensitivities",
        "Trade sensitivity table",
        (
            "trade_sensitivities is the trade-level input table with trade_id, risk_factor, desk, sensitivity_type, sensitivity_value, and product. "
            "It stores exposure, not P&L, and is used for risk-factor attribution/explain, not for aggregating VaR."
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
        "schema_trade_pnl",
        "Trade P&L table",
        (
            "trade_pnl is the stored, actual trade-level P&L table with trade_id, desk, product, historical_date, scenario_name, and pnl. "
            "It is the ground truth for every trade on every historical scenario date (as if from full revaluation), and is the "
            "table to aggregate for portfolio, desk, product, or entity-level P&L and VaR."
        ),
        ("schema", "trade", "pnl"),
    ),
    KnowledgeChunk(
        "scenario_pnl_logic",
        "Scenario P&L logic",
        (
            "Portfolio/desk/entity scenario P&L is SUM(trade_pnl.pnl) grouped by historical_date for the scope's trades. "
            "This is aggregation of stored actual P&L, not a computation from sensitivities and shocks."
        ),
        ("pnl", "scenario", "calculation"),
    ),
    KnowledgeChunk(
        "aggregation_logic",
        "Aggregation logic",
        (
            "Aggregation is essential for VaR. Sum trade_pnl.pnl by historical_date across the selected desk/product/trades "
            "to create that scope's portfolio P&L distribution before ranking it into a percentile."
        ),
        ("aggregation", "pnl", "var"),
    ),
    KnowledgeChunk(
        "var_logic",
        "VaR percentile logic",
        (
            "VaR is computed from a scope's own aggregated P&L distribution (from trade_pnl). "
            "loss_amount = max(-scenario_pnl, 0). 95% VaR uses the nearest-rank method: rank the N historical "
            "loss_amounts ascending and take the ceil(N x 0.95)-th smallest (e.g. the 3rd-worst day out of 50). "
            "VaR is NOT additive: entity-level VaR is not the sum of desk-level VaRs, because each scope's 95th-percentile "
            "scenario date can be a different historical date. Only P&L is additive across scopes; VaR is not. "
            "VaR is not actual P&L and is not a maximum possible loss."
        ),
        ("var", "percentile", "loss", "additivity"),
    ),
    KnowledgeChunk(
        "var_drivers",
        "VaR driver logic",
        (
            "To explain what drove VaR, first find the scope's 95th-percentile historical scenario date from trade_pnl. "
            "Then, for that single date only, join trade_sensitivities to risk_factor_scenarios (sensitivity_value * shock_value) "
            "to attribute the move by risk factor. This linear attribution is an approximation of the actual trade_pnl on that "
            "date, so an 'Unexplained (non-linear residual)' amount reconciles the attributed total back to actual P&L. "
            "Rank drivers by absolute driver_pnl."
        ),
        ("var", "drivers", "risk_factor", "residual"),
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
