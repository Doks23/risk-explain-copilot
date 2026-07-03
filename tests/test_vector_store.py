from __future__ import annotations

from src.vector_store import initialize_vector_store, retrieve_context


def test_retrieve_context_returns_relevant_coverage_chunk(tmp_path) -> None:
    db_path = tmp_path / "vectors.db"
    initialize_vector_store(db_path=db_path, force=True)

    context = retrieve_context("What desks are we covering?", db_path=db_path)
    titles = [item["title"] for item in context]

    assert "Hierarchy table" in titles
    assert context[0]["score"] > 0


def test_retrieve_context_returns_relevant_market_move_chunk(tmp_path) -> None:
    db_path = tmp_path / "vectors.db"
    initialize_vector_store(db_path=db_path, force=True)

    context = retrieve_context("What market moves explain the PNL change?", db_path=db_path)
    titles = [item["title"] for item in context]

    assert "Market move explanation" in titles
