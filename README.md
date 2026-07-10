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

Aggregated P&L and VaR are computed directly from `trade_pnl`, which stores real P&L per risk
factor per historical scenario date per COB. Explaining VaR by risk factor or by desk means
grouping that same real data on the VaR's own historical date — exact, not an approximation.

**VaR is not additive.** Entity-level VaR is not the sum of desk-level VaRs, because each scope's
worst-case historical date can differ. Only P&L is additive across scopes — which is exactly why a
*desk breakdown of entity VaR* (same date, split by desk) reconciles perfectly, while *desk-level
VaR figures* (each on their own date) do not sum back to the entity.

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

The database stores three source tables, all scoped by `cob_date` (close-of-business snapshot).
The book changes across COBs — trades book and mature, and sensitivities are re-marked every COB
even for surviving trades — so the same `trade_id` can look different on different `cob_date` rows.

### `trade_sensitivities`

Trade-level exposure data, per COB.

Columns: `cob_date`, `trade_id`, `risk_factor`, `desk`, `sensitivity_type`, `sensitivity_value`, `product`

Used for explain (risk-factor context), not aggregation.

### `risk_factor_scenarios`

Historical risk-factor shocks. Not COB-scoped — it's raw market data, the same shock on the same
historical date regardless of which COB's rolling window is looking at it.

Columns: `historical_date`, `scenario_name`, `risk_factor`, `shock_value`, `shock_unit`

### `trade_pnl`

Actual P&L per trade, per risk factor, per historical scenario date, per COB (as if from full
revaluation, attributed by risk factor). This is the source of truth for aggregation, VaR, and
risk-factor/desk breakdowns at any scope.

Columns: `cob_date`, `trade_id`, `desk`, `product`, `risk_factor`, `historical_date`, `scenario_name`, `pnl`

Each `cob_date` uses its own 100-day rolling window of historical dates, ending the day *before*
that COB — COB N's window ends one day later than COB N-1's, so consecutive COBs' windows overlap
by 99 days. `risk_factor_scenarios` is generated once over the union of every COB's window, so the
same historical date always has the same shock regardless of which COB is looking at it.

## Business Logic

Portfolio/desk/entity scenario P&L (aggregation, always from `trade_pnl`, always scoped to one `cob_date`):

```text
scenario_pnl = SUM(trade_pnl.pnl) by historical_date, for the scope's trades on that cob_date
```

Loss amount:

```text
loss_amount = max(-scenario_pnl, 0)
```

95% VaR — literally the k-th worst day, not nearest-rank or linear interpolation:

```text
k = ROUND(N x (1 - 0.95))              e.g. ROUND(100 x 0.05) = 5
95% VaR = the k-th worst day's loss_amount, across the scope's own N-day rolling window
        = the 5th-worst day out of 100 historical scenarios
```

VaR is not additive: entity VaR != SUM(desk VaRs), because each scope computes its own 95% VaR
independently and each one's worst-case historical date can differ.

**Explaining VaR by risk factor** (drill into *why*, on the scope's own VaR date):

1. Find the historical date selected by the scope's 95% VaR.
2. `GROUP BY risk_factor` directly on `trade_pnl` for that single `(cob_date, historical_date, scope)`.
   This is exact — `trade_pnl` already stores real P&L per risk factor — so it reconciles perfectly
   to the scope's total P&L on that date with no approximation or residual. `sensitivity_value`/
   `shock_value` are joined in from `trade_sensitivities`/`risk_factor_scenarios` purely as
   supporting color.

**Explaining VaR by desk** (e.g. break entity VaR down by desk):

1. Find the historical date selected by the scope's 95% VaR (e.g. the entity's).
2. `GROUP BY desk` directly on `trade_pnl` for that same single historical date, with no desk filter.
3. Because every desk's number comes from the exact same date, these figures **are** additive and
   sum exactly to the scope's total — unlike desk-level VaR figures, which are each computed
   independently on their own worst-case date and never sum back to the entity's VaR.

## Sample Questions

- What desks are we covering? (as of the latest COB)
- Show trade level P&L for D1.
- Calculate 95% VaR for D1.
- Calculate 95% VaR for D1 as of COB 2026-07-08.
- Explain VaR risk factor drivers for D1.
- Explain entity VaR by desk.
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

## Deployment

Streamlit needs a persistent process (not serverless functions), so it doesn't run on
Vercel. Deploy to **[Streamlit Community Cloud](https://share.streamlit.io)** instead — free,
built for this, and needs zero code changes:

1. Push this repo to GitHub.
2. On share.streamlit.io: **New app** → pick the repo/branch → main file path `app.py` → **Deploy**.
3. In the app's **Settings → Secrets**, add your key(s):

   ```toml
   GEMINI_API_KEY = "..."
   GEMINI_MODEL = "gemini-2.5-flash"
   GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
   ```

   Streamlit Cloud exposes secrets as real environment variables, so `os.getenv(...)` picks
   them up with no code change.
4. First load regenerates `data/*.csv`/`*.db` and re-indexes `docs/` into Chroma automatically
   (see `bootstrap_database()` and `ingest_documents()`) — no database service or persistent
   volume needed, since both are self-healing on a fresh container.

For a host with a persistent disk (chat history/uploaded docs surviving restarts), Render or
Railway both run `streamlit run app.py` directly with no config changes.

## Future Enhancements

- Add trade economics and notional fields.
- Add expected shortfall.
- Add confidence-level selection.
- Add richer scenario labels and real historical market data adapters.
