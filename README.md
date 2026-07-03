# Risk Explain Copilot

Risk Explain Copilot is a Streamlit prototype for market risk analysts who ask:

> Why did VaR or PNL move between two dates?

The app uses natural-language questions to generate safe SQL, run a bounded SQLite query, and answer only from the returned rows. It does not expose the full database in the UI.

## Problem Statement

Daily risk review is not just raw reporting. Analysts need to separate actual PNL movement from risk estimate movement, then connect the movement to risk factors, scenarios, market moves, and desk/book hierarchy.

This prototype makes that distinction explicit:

- PNL explanation uses `sensitivity_value x market_move`.
- VaR explanation uses changes in `var_contribution` by scenario and risk factor.
- The answer layer is grounded in SQL output, so it should not invent numbers.

## Architecture

- `app.py`: Streamlit chat UI. Each answer keeps generated SQL, bounded result rows, trace, and retrieved context under collapsed `Explanation details`.
- `src/data_generator.py`: Generates deterministic sample market risk data.
- `src/db.py`: Validates CSV columns, loads SQLite tables, and resets/reloads data.
- `src/analytics.py`: Testable business calculations for PNL attribution, VaR attribution, trends, and drilldown.
- `src/knowledge_base.py`: Business definitions and schema guidance used by retrieval.
- `src/vector_store.py`: Local SQLite-backed vector store with deterministic embeddings.
- `src/query_engine.py`: Natural-language-to-SQL planner, SQL safety checks, execution, and answer generation.
- `tests/`: Tests for SQL safety, query behavior, analytics formulas, follow-up context, and retrieval.

## Setup Steps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Optional OpenAI-compatible mode:

```bash
OPENAI_API_KEY="..."
OPENAI_MODEL="gpt-4o-mini"
OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

Optional Gemini mode:

```bash
GEMINI_API_KEY="..."
GEMINI_MODEL="gemini-2.5-flash"
GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
```

If no key is configured, the app uses deterministic fallback SQL and response generation.

## How To Run

```bash
streamlit run app.py
```

Dummy data loads automatically. Use the sidebar to upload replacement CSVs, reload SQLite, regenerate sample data, or clear chat.

If your environment has a generic dev runner, this also works:

```bash
npm run dev
```

The npm script is only a thin wrapper around Streamlit; there are no Node dependencies.

## Business Logic

PNL:

- Actual PNL change = `SUM(pnl_value on date2) - SUM(pnl_value on date1)`.
- Estimated PNL impact = `sensitivity_value from date1 x market_move from date1 exclusive to date2 inclusive`.
- Explained PNL = sum of estimated PNL impacts.
- Residual PNL = actual PNL change - explained PNL.
- PNL drivers are ranked by absolute estimated PNL impact.

VaR:

- VaR is a risk estimate, not actual loss.
- VaR change = `SUM(var_contribution on date2) - SUM(var_contribution on date1)`.
- Scenario drivers compare scenario-level VaR contribution between two dates.
- Risk-factor drivers compare risk-factor VaR contribution between two dates.
- VaR drivers are ranked by absolute contribution movement.

The app should not explain VaR as PNL, and should not explain PNL as VaR.

## Query Flow

1. User asks a question in chat.
2. The vector store retrieves relevant schema and business context.
3. If an API key is present, the LLM generates one read-only SQLite query.
4. If no API key is present, a deterministic fallback planner handles common PNL, VaR, trend, coverage, driver, market move, and drilldown questions.
5. SQL safety validation blocks writes, multiple statements, `SELECT *`, SQLite internals, and unbounded raw-table exposure.
6. The app executes only the bounded query.
7. The answer layer receives the question plus SQL result rows and produces a concise answer.
8. Previous chat turns are retained so follow-up questions can reuse scope and metric context.

## Sample Questions

- Why did VAR move for London Rates?
- Explain PNL movement for London Rates.
- What market moves explain the PNL change?
- What are the top 5 scenario drivers for VAR?
- Show 10-day VAR trend for FX Options.
- Drill down London Rates VAR by risk factor.
- What desks are we covering?

## Data Model

The app loads six CSVs into SQLite:

- `hierarchy.csv`: `date`, `desk`, `book`, `portfolio`, `product`, `currency`
- `pnl_results.csv`: `date`, `desk`, `book`, `portfolio`, `product`, `pnl_value`
- `var_results.csv`: `date`, `desk`, `book`, `portfolio`, `scenario`, `risk_factor`, `product`, `var_contribution`
- `sensitivities.csv`: `date`, `desk`, `book`, `portfolio`, `product`, `risk_factor`, `sensitivity_type`, `sensitivity_value`
- `market_data.csv`: `date`, `risk_factor`, `market_level`, `market_move`, `move_unit`
- `scenario_data.csv`: `date`, `scenario`, `risk_factor`, `shock_value`, `shock_unit`

Table relationships:

- `desk`, `book`, `portfolio`, and `product` link hierarchy, PNL, VaR, and sensitivities.
- `risk_factor` links sensitivities, market data, scenario data, and VaR contribution rows.
- `scenario` links VaR contribution rows to scenario shocks.
- `date` defines the comparison window and trend grain.

Generated sample coverage:

- 5 desks
- 3 books per desk
- 15 portfolios
- 20+ risk factors
- 10 scenarios
- 15 business dates
- Multiple products and currencies

## SQL Safety

- Only `SELECT` or `WITH ... SELECT` statements are allowed.
- `SELECT *` is rejected.
- Multiple statements are rejected.
- Write/admin commands are rejected.
- Results are capped at 50 rows unless the generated query has a lower limit.
- The UI shows only answer-specific bounded result rows inside collapsed details.

## Future Enhancements

- Richer date parsing for explicit ranges, MTD, and quarter-to-date.
- More realistic residual PNL decomposition for carry, fees, trades, and model effects.
- Scenario shock interpretation in the final answer.
- Role-based access controls and desk-level entitlements.
- Production adapters for risk runs, market data, and trade lifecycle events.
- Confidence scoring for stale, missing, or weakly explained movements.
