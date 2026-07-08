from __future__ import annotations

from src.vector_store import initialize_vector_store, retrieve_context


def test_retrieve_context_returns_relevant_coverage_chunk(tmp_path) -> None:
    db_path = tmp_path / "vectors.db"
    initialize_vector_store(db_path=db_path, force=True)

    context = retrieve_context("What desks and trades are we covering?", db_path=db_path)
    titles = [item["title"] for item in context]

    assert "Coverage questions" in titles
    assert context[0]["score"] > 0


def test_retrieve_context_returns_relevant_scenario_pnl_chunk(tmp_path) -> None:
    db_path = tmp_path / "vectors.db"
    initialize_vector_store(db_path=db_path, force=True)

    context = retrieve_context("How does sensitivity times shock create scenario P&L?", db_path=db_path)
    titles = [item["title"] for item in context]

    assert "Scenario P&L logic" in titles


def test_retrieve_context_returns_relevant_var_chunk(tmp_path) -> None:
    db_path = tmp_path / "vectors.db"
    initialize_vector_store(db_path=db_path, force=True)

    context = retrieve_context("How is 95 percent VaR calculated from scenario P&L?", db_path=db_path)
    titles = [item["title"] for item in context]

    assert "VaR percentile logic" in titles
