# Risk Explain Copilot

Risk Explain Copilot is a Streamlit prototype for market risk analysts who want to understand how trade sensitivities and historical risk-factor shocks create scenario P&L and VaR.

The app uses natural-language questions to generate safe SQL, run a bounded SQLite query, and answer only from the returned rows. It does not expose the full database in the UI.

## Problem Statement

The core market risk chain is:

```text
Risk Factor -> Sensitivity -> Shock / Scenario -> Scenario P&L -> P&L Distribution -> VaR
```

In business terms:

- Risk factor: what market variable can move?
- Sensitivity: how exposed is the trade to that move?
- Shock/scenario: how much does that risk factor move in a historical scenario?
- P&L: what is the financial impact?
- VaR: what percentile loss comes from the aggregated P&L distribution?

This prototype stores only the two source inputs:

- trade-level sensitivities
- historical risk-factor scenarios

Scenario P&L and VaR are computed from those inputs.

## Architecture

- `app.py`: Streamlit chat UI. Each answer keeps generated SQL, bounded result rows, trace, and retrieved context under collapsed `Explanation details`.
- `src/data_generator.py`: Generates deterministic trade sensitivities and historical RF scenarios.
- `src/db.py`: Validates CSV columns, loads SQLite tables, and resets/reloads data.
- `src/analytics.py`: Computes trade scenario P&L, aggregated P&L distributions, VaR, and drivers.
- `src/knowledge_base.py`: Business definitions and schema guidance used for retrieval.
- `src/vector_store.py`: Local SQLite-backed vector store with deterministic embeddings.
- `src/query_engine.py`: Natural-language-to-SQL planner, SQL safety checks, execution, and answer generation.
- `tests/`: Tests for SQL safety, analytics formulas, query behavior, follow-up context, and retrieval.

## Setup Steps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Optional Gemini mode:

```bash
GEMINI_API_KEY="..."
GEMINI_MODEL="gemini-2.5-flash"
GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
```

Optional OpenAI-compatible mode:

```bash
OPENAI_API_KEY="..."
OPENAI_MODEL="gpt-4o-mini"
OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

If no key is configured, the app uses deterministic fallback SQL and response generation.

## How To Run

```bash
streamlit run app.py
```

or:

```bash
npm run dev
```

Dummy data loads automatically. Use the sidebar to upload replacement CSVs, reload SQLite, regenerate sample data, or clear chat.

## Data Model

The database stores two source tables.

### `trade_sensitivities`

Trade-level exposure data.

Columns:

- `trade_id`
- `risk_factor`
- `desk`
- `sensitivity_type`
- `sensitivity_value`
- `product`

This table answers: how exposed is each trade to each risk factor?

### `risk_factor_scenarios`

Historical risk-factor shocks.

Columns:

- `historical_date`
- `scenario_name`
- `risk_factor`
- `shock_value`
- `shock_unit`

Each historical date is treated as one scenario containing shocks across risk factors.

## Business Logic

Trade scenario P&L:

```text
scenario_pnl = sensitivity_value x shock_value
```

Portfolio scenario P&L:

```text
portfolio_scenario_pnl = SUM(scenario_pnl) by historical_date
```

Loss amount:

```text
loss_amount = max(-portfolio_scenario_pnl, 0)
```

95% VaR:

```text
95% VaR = 95th percentile of loss_amount across historical dates
```

VaR drivers:

1. Find the historical date selected by the 95% VaR percentile.
2. Recompute trade/risk-factor P&L for that date.
3. Group by risk factor.
4. Rank by absolute driver P&L.

## Sample Questions

- What desks are we covering?
- Show trade level P&L for D1.
- Calculate 95% VaR for D1.
- Explain VaR risk factor drivers for D1.
- Show 10-day P&L trend for D2.
- Show risk factor scenarios for SOFR.
- What trades are in Equity Option?

## Query Flow

1. User asks a question in chat.
2. The vector store retrieves relevant schema and business context.
3. If an API key is present, the LLM generates one read-only SQLite query.
4. If no API key is present, a deterministic fallback planner handles common coverage, P&L, VaR, scenario, driver, and trend questions.
5. SQL safety validation blocks writes, multiple statements, `SELECT *`, SQLite internals, and unbounded raw-table exposure.
6. The app executes only the bounded query.
7. The answer layer receives the question plus SQL result rows and produces a concise answer.
8. Previous chat turns are retained so follow-up questions can reuse scope and metric context.

## SQL Safety

- Only `SELECT` or `WITH ... SELECT` statements are allowed.
- `SELECT *` is rejected.
- Multiple statements are rejected.
- Write/admin commands are rejected.
- Results are capped at 50 rows unless the generated query has a lower limit.
- The UI shows only answer-specific bounded result rows inside collapsed details.

## Future Enhancements

- Add explicit Postgres support for Prisma/Vercel storage.
- Add trade economics and notional fields.
- Add product-level and desk-level VaR decomposition.
- Add expected shortfall.
- Add confidence-level selection.
- Add richer scenario labels and real historical market data adapters.
