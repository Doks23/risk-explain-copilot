from __future__ import annotations

from src.document_store import DOCS_DIR, _source_fingerprint, ingest_documents, retrieve_context


def test_retrieve_context_returns_empty_without_embeddings_configured() -> None:
    assert retrieve_context("What is 95% VaR?") == ()


def test_ingest_documents_is_a_noop_without_embeddings_configured() -> None:
    assert ingest_documents() is False


def test_source_fingerprint_includes_business_rule_chunks_and_doc_files() -> None:
    fingerprint = _source_fingerprint()

    assert any(key.startswith("chunk::") for key in fingerprint)
    if DOCS_DIR.exists() and any(DOCS_DIR.glob("*.md")):
        assert any(key.startswith("file::") for key in fingerprint)
