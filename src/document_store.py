from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .data_generator import DATA_DIR
from .knowledge_base import KNOWLEDGE_CHUNKS
from .llm_config import get_llm_config


ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma_db"
MANIFEST_PATH = CHROMA_DIR / "manifest.json"
COLLECTION_NAME = "risk_explain_knowledge"
DOC_SUFFIXES = {".md", ".txt", ".pdf"}

EMBEDDING_MODEL_ENV = {
    "OpenAI-compatible": ("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    "Gemini": ("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
}


def get_embeddings():
    config = get_llm_config()
    if not config:
        return None

    env_var, default_model = EMBEDDING_MODEL_ENV[config.provider]
    model = os.getenv(env_var, default_model)

    if config.provider == "Gemini":
        # Gemini's OpenAI-compatible endpoint only proxies chat completions, not
        # /embeddings (returns 501). Embeddings need the native Google GenAI API.
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model_name = model if model.startswith("models/") else f"models/{model}"
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=config.api_key)

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(api_key=config.api_key, base_url=config.base_url, model=model)


def _open_store(embeddings):
    from langchain_chroma import Chroma

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))


def _source_fingerprint() -> dict[str, str]:
    fingerprint = {f"chunk::{chunk.chunk_id}": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest() for chunk in KNOWLEDGE_CHUNKS}
    if DOCS_DIR.exists():
        for path in sorted(DOCS_DIR.rglob("*")):
            if path.is_file() and path.suffix.lower() in DOC_SUFFIXES:
                fingerprint[f"file::{path.relative_to(DOCS_DIR)}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprint


def _load_documents() -> list[Any]:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = [
        Document(page_content=chunk.text, metadata={"title": chunk.title, "source": "business_rules", "chunk_id": chunk.chunk_id})
        for chunk in KNOWLEDGE_CHUNKS
    ]

    if not DOCS_DIR.exists():
        return documents

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    for path in sorted(DOCS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in DOC_SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loaded = PyPDFLoader(str(path)).load()
        else:
            from langchain_community.document_loaders import TextLoader

            loaded = TextLoader(str(path), encoding="utf-8").load()
        for doc in loaded:
            doc.metadata["title"] = path.stem.replace("_", " ").replace("-", " ").title()
            doc.metadata["source"] = str(path.relative_to(DOCS_DIR))
        documents.extend(splitter.split_documents(loaded))

    return documents


def ingest_documents(force: bool = False) -> bool:
    """(Re)build the Chroma index only when source content changed. Returns True if it (re)indexed."""
    embeddings = get_embeddings()
    if embeddings is None:
        return False

    fingerprint = _source_fingerprint()
    if not force and MANIFEST_PATH.exists():
        try:
            previous = json.loads(MANIFEST_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            previous = None
        if previous == fingerprint:
            return False

    import chromadb
    from langchain_chroma import Chroma

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    documents = _load_documents()
    if documents:
        Chroma.from_documents(documents=documents, embedding=embeddings, collection_name=COLLECTION_NAME, persist_directory=str(CHROMA_DIR))
    MANIFEST_PATH.write_text(json.dumps(fingerprint))
    return True


def retrieve_context(query: str, top_k: int = 5) -> tuple[dict[str, Any], ...]:
    embeddings = get_embeddings()
    if embeddings is None:
        return ()

    store = _open_store(embeddings)
    results = store.similarity_search_with_score(query, k=top_k)
    context = []
    for doc, distance in results:
        relevance = 1.0 / (1.0 + float(distance))
        context.append(
            {
                "title": doc.metadata.get("title", doc.metadata.get("source", "Document")),
                "text": doc.page_content,
                "score": round(relevance, 4),
                "source": doc.metadata.get("source", ""),
            }
        )
    return tuple(context)
