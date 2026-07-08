# Risk Explain Copilot

Risk Explain Copilot is a Streamlit prototype for market risk analysts who want to understand how trade sensitivities and historical risk-factor shocks create scenario P&L and VaR.

The app uses natural-language questions to generate safe SQL, run a bounded SQLite query, and answer only from the returned rows. It does not expose the full database in the UI.

## Problem Statement

The core market risk chain is:

```text
Risk Factor -> Sensitivity -> Shock / Scenario -> Trade P&L -> P&L Distribution -> VaR
```

In business terms:

- Risk factor: what market variable can move?
- Sensitivity: how exposed is a trade to that move?
- Shock/scenario: how much does that risk factor move on a historical date?
- Trade P&L: what was the actual financial impact on that trade for that scenario?
- VaR: what percentile loss comes from a scope's aggregated P&L distribution?

This prototype stores three source inputs:

- trade-level sensitivities (exposure)
- historical risk-factor scenarios (shocks)
- trade-level P&L (the actual, stored P&L per trade per historical scenario date, as if from full revaluation)

Aggregated P&L and VaR are computed directly from `trade_pnl`. Sensitivities and shocks are used
only to *explain* what drove VaR on its worst-case date, via a linear risk-factor attribution that
may leave a small unexplained residual against the actual P&L — the same gap real desks see between
Greeks-based P&L explain and full revaluation.

**VaR is not additive.** Entity-level VaR is not the sum of desk-level VaRs, because each scope's
95th-percentile scenario date can be a different historical date. Only P&L is additive across scopes.

## Architecture

- `app.py`: Streamlit chat UI, single column, no sidebar. Each answer keeps generated SQL and calculation trace under a collapsed `Explanation details`.
- `src/data_generator.py`: Generates deterministic trade sensitivities, historical RF scenarios, and trade-level P&L.
- `src/db.py`: Validates CSV columns, loads SQLite tables, and resets/reloads data.
- `src/llm_config.py`: Resolves the active LLM provider (OpenAI-compatible or Gemini) from environment variables; shared by chat and embeddings.
- `src/knowledge_base.py`: Hardcoded business-rule/schema chunks, ingested into the vector store alongside real documents.
- `src/document_store.py`: RAG layer — LangChain loaders (`docs/*.md|.txt|.pdf`) → text splitter → embeddings (Gemini native or OpenAI) → a persisted Chroma vector store. Re-embeds only when source content changes.
- `src/query_engine.py`: Natural-language-to-SQL planner, SQL safety checks, execution, and answer generation.
- `docs/`: Source documents for retrieval (methodology, policy). Drop in your own `.md`/`.txt`/`.pdf` files.
- `tests/`: Tests for SQL safety, query behavior, follow-up context, and retrieval.

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
GEMINI_EMBEDDING_MODEL="gemini-embedding-001"
```

Optional OpenAI-compatible mode:

```bash
OPENAI_API_KEY="..."
OPENAI_MODEL="gpt-4o-mini"
OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```

If no key is configured, the app uses deterministic fallback SQL and response generation,
and the RAG knowledge base is not indexed (retrieval returns nothing — the LLM path needs a
key for both chat and embeddings). Gemini embeddings use the native Google GenAI API, since
Gemini's OpenAI-compatible endpoint only proxies chat completions, not `/embeddings`.

## How To Run

```bash
streamlit run app.py
```

or:

```bash
npm run dev
```

Sample data generates and loads automatically on first run. To regenerate it, delete `data/*.csv` and `data/*.db` and restart the app.

Drop `.md`/`.txt`/`.pdf` files into `docs/` and restart the app to index them — ingestion
hashes source content and only re-embeds when something actually changed, so restarts with
unchanged docs are instant.

## Data Model

The database stores three source tables.

### `trade_sensitivities`

Trade-level exposure data.

Columns:

- `trade_id`
- `risk_factor`
- `desk`
- `sensitivity_type`
- `sensitivity_value`
- `product`

This table answers: how exposed is each trade to each risk factor? Used for explain, not aggregation.

### `risk_factor_scenarios`

Historical risk-factor shocks.

Columns:

- `historical_date`
- `scenario_name`
- `risk_factor`
- `shock_value`
- `shock_unit`

Each historical date is treated as one scenario containing shocks across risk factors. Used for explain, not aggregation.

### `trade_pnl`

Actual trade-level P&L per historical scenario date (as if from full revaluation).

Columns:

- `trade_id`
- `desk`
- `product`
- `historical_date`
- `scenario_name`
- `pnl`

This is the source of truth for aggregation and VaR at any scope (entity, desk, product, or trade).

## Business Logic

Portfolio/desk/entity scenario P&L (aggregation, always from `trade_pnl`):

```text
scenario_pnl = SUM(trade_pnl.pnl) by historical_date, for the scope's trades
```

Loss amount:

```text
loss_amount = max(-scenario_pnl, 0)
```

95% VaR (nearest-rank method):

```text
95% VaR = ceil(N x 0.95)-th smallest loss_amount across the scope's N historical dates
        = the 3rd-worst day out of 50 historical scenarios
```

VaR is not additive: entity VaR != SUM(desk VaRs), because each scope's 95th-percentile date can differ.

VaR drivers (explain, not aggregation):

1. Find the historical date selected by the scope's 95% VaR percentile.
2. Attribute P&L on that date by risk factor: `driver_pnl = sensitivity_value x shock_value`, from `trade_sensitivities` joined to `risk_factor_scenarios`.
3. Group by risk factor and rank by absolute driver P&L.
4. Reconcile: `residual = actual scenario_pnl on that date - SUM(driver_pnl)`, shown as an "Unexplained (non-linear residual)" row.

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
- The UI never renders raw database tables; `Explanation details` shows only the generated SQL and the calculation trace.

## Future Enhancements

- Add explicit Postgres support for Prisma/Vercel storage.
- Add trade economics and notional fields.
- Add expected shortfall.
- Add confidence-level selection.
- Add richer scenario labels and real historical market data adapters.
