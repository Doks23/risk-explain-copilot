from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _no_llm_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests exercise the deterministic rule-based fallback, not a live LLM API.

    A real key in the developer's .env would otherwise make generate_query_plan()
    call out to OpenAI/Gemini during every test that doesn't explicitly need it.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
