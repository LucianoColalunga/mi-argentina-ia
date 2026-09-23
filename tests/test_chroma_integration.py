"""Integración contra un ChromaDB real (sin GPU ni Ollama: embeddings deterministas).

  docker compose up -d chroma
  RUN_INTEGRATION=1 CHROMA_HOST=localhost pytest tests/test_chroma_integration.py
"""

import asyncio
import os
import time

import pytest

import ingest
from app.cache import SemanticCache
from app.config import Settings
from app.knowledge import KnowledgeBase
from tests.conftest import fake_embed

pytestmark = pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="requiere ChromaDB (RUN_INTEGRATION=1)")


class DeterministicEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [fake_embed(text) for text in texts]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        kb_collection_prefix="it_faq",
        kb_meta_collection="it_meta",
        semantic_cache_collection="it_semantic_cache",
        ollama_embed_model="deterministic-test",
        kb_refresh_seconds=0,
    )


@pytest.fixture
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ingest, "build_embeddings", lambda _settings: DeterministicEmbeddings())
    chroma = ingest.wait_for_chroma(settings, timeout_seconds=60)
    yield chroma
    for collection in chroma.list_collections():
        if collection.name.startswith("it_"):
            chroma.delete_collection(name=collection.name)


def test_blue_green_ingest_and_search(client, settings: Settings) -> None:
    entries = ingest.clean_entries(ingest.load_local_faqs(ingest.BASE_DIR / "data" / "faqs"))
    documents = ingest.build_chunks(entries, chunk_size=1000, chunk_overlap=150)

    first = ingest.write_index(client, settings, documents, batch_size=16, keep=2)
    assert ingest.active_index_is_usable(client, settings)

    versions = [first]
    for _ in range(2):
        time.sleep(1.1)  # la versión tiene resolución de segundos
        versions.append(ingest.write_index(client, settings, documents[:5], batch_size=16, keep=1))
    second = versions[-1]
    assert len(set(versions)) == 3
    remaining = [c.name for c in client.list_collections() if c.name.startswith("it_faq_")]
    assert len(remaining) == 2  # el activo + el anterior (mínimo 2 aunque se pida keep=1)

    kb = KnowledgeBase(settings)
    info = asyncio.run(kb.active())
    assert info is not None and info.version == second and info.embed_model == "deterministic-test"

    query = documents[0].page_content
    chunks = asyncio.run(kb.search(info, fake_embed(query), 3))
    assert chunks[0].text == query and chunks[0].score > 0.99
    assert chunks[0].url.startswith("https://")
    assert asyncio.run(kb.ping())


def test_semantic_cache_roundtrip(client, settings: Settings) -> None:
    cache = SemanticCache(lambda: client, settings.semantic_cache_collection, threshold=0.95)
    vector = fake_embed("como activo el dni en el celular")

    # Llamadas directas (sin fail-open) para que cualquier error de la API de Chroma falle el test.
    assert cache._lookup_sync(vector, "scope-a") is None  # colección vacía
    cache._store_sync(vector, "scope-a", "answer-key-1", "como activo el dni en el celular")
    assert cache._lookup_sync(vector, "scope-a") == "answer-key-1"
    assert cache._lookup_sync(vector, "scope-b") is None  # otra versión del índice/prompt
    assert cache._lookup_sync(fake_embed("turnos para pasaporte urgente"), "scope-a") is None

    # La ingesta borra la colección: el handle local queda apuntando a un id inexistente.
    client.delete_collection(name=settings.semantic_cache_collection)
    assert cache._lookup_sync(vector, "scope-a") is None  # se recupera sin error
    cache._store_sync(vector, "scope-a", "answer-key-2", "como activo el dni en el celular")
    assert cache._lookup_sync(vector, "scope-a") == "answer-key-2"
