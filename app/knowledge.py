"""Acceso a la base de conocimiento oficial en ChromaDB.

Despliegue blue/green del índice: cada ingesta crea una colección nueva
(`<prefijo>_<versión>`) y, recién cuando está completa, actualiza un puntero en la
colección de metadatos. La API relee ese puntero cada `KB_REFRESH_SECONDS`, así que
se re-indexa sin downtime y la versión del índice forma parte de la clave de caché.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings

logger = logging.getLogger(__name__)

# Claves del puntero guardado en la metadata de la colección de metadatos.
META_ACTIVE_COLLECTION = "active_collection"
META_KB_VERSION = "kb_version"
META_EMBED_MODEL = "embed_model"
META_CHUNKS = "chunks"
META_UPDATED_AT = "updated_at"

COLLECTION_CONFIGURATION = {"hnsw": {"space": "cosine"}}


def create_chroma_client(settings: Settings) -> Any:
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


@dataclass(frozen=True)
class KBInfo:
    collection: str
    version: str
    embed_model: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    title: str
    url: str
    score: float  # similitud coseno (1 = idéntico)


class KnowledgeBase:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any = None
        self._client_lock = threading.Lock()
        self._active: KBInfo | None = None
        self._active_expires_at = 0.0
        self._collections: dict[str, Any] = {}

    # -- cliente ---------------------------------------------------------------

    def client(self) -> Any:
        """Cliente HTTP de Chroma creado de forma perezosa (la API arranca aunque Chroma demore)."""
        with self._client_lock:
            if self._client is None:
                self._client = create_chroma_client(self._settings)
            return self._client

    def _reset(self) -> None:
        with self._client_lock:
            self._client = None
        self._collections.clear()
        self._active_expires_at = 0.0

    def _collection(self, name: str) -> Any:
        if name not in self._collections:
            self._collections[name] = self.client().get_collection(name=name, embedding_function=None)
        return self._collections[name]

    # -- versión activa --------------------------------------------------------

    def _read_active_sync(self) -> KBInfo | None:
        client = self.client()
        try:
            meta = client.get_collection(name=self._settings.kb_meta_collection, embedding_function=None)
            metadata = meta.metadata or {}
            name = metadata.get(META_ACTIVE_COLLECTION)
            if name:
                return KBInfo(
                    collection=str(name),
                    version=str(metadata.get(META_KB_VERSION, name)),
                    embed_model=metadata.get(META_EMBED_MODEL),
                )
        except Exception:  # noqa: BLE001 - la colección de metadatos todavía no existe
            logger.debug("kb_meta_missing")

        # Plan B: la colección más reciente con el prefijo configurado.
        prefix = f"{self._settings.kb_collection_prefix}_"
        names = sorted(c.name for c in client.list_collections() if c.name.startswith(prefix))
        if not names:
            return None
        return KBInfo(collection=names[-1], version=names[-1].removeprefix(prefix))

    async def active(self) -> KBInfo | None:
        now = time.monotonic()
        if self._active is not None and now < self._active_expires_at:
            return self._active
        try:
            info = await asyncio.to_thread(self._read_active_sync)
        except Exception as exc:  # noqa: BLE001
            logger.error("kb_unavailable", extra={"ctx": {"error": str(exc)}})
            self._reset()
            return self._active  # se sigue sirviendo con la última versión conocida
        if info and info.embed_model and info.embed_model != self._settings.embed_model_name:
            logger.error(
                "embedding_model_mismatch",
                extra={"ctx": {"index": info.embed_model, "api": self._settings.embed_model_name}},
            )
        if info != self._active:
            logger.info("kb_active_version", extra={"ctx": {"collection": getattr(info, "collection", None)}})
        self._active = info
        self._active_expires_at = now + self._settings.kb_refresh_seconds
        return info

    # -- búsqueda --------------------------------------------------------------

    def _search_sync(self, collection_name: str, embedding: list[float], k: int) -> list[RetrievedChunk]:
        result = self._collection(collection_name).query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[RetrievedChunk] = []
        if not result["ids"] or not result["ids"][0]:
            return chunks
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            metadata = metadata or {}
            chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=document or "",
                    title=str(metadata.get("title", "")),
                    url=str(metadata.get("url", "")),
                    score=1.0 - float(distance),
                )
            )
        return chunks

    async def search(self, info: KBInfo, embedding: list[float], k: int) -> list[RetrievedChunk]:
        try:
            return await asyncio.to_thread(self._search_sync, info.collection, embedding, k)
        except Exception:
            self._reset()  # p. ej. la colección fue reemplazada por una ingesta nueva
            raise

    async def ping(self) -> bool:
        try:
            await asyncio.to_thread(self.client().heartbeat)
            return True
        except Exception:  # noqa: BLE001
            self._reset()
            return False
