"""Caché de respuestas en Redis, coalescencia de pedidos (single-flight) y caché semántica.

Principio de diseño: la caché nunca es un punto único de falla. Si Redis o Chroma
no responden, se registra el error y el pedido sigue su curso (fail-open).
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections.abc import Callable
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# Libera el lock solo si sigue siendo nuestro (evita borrar el lock de otra réplica).
_RELEASE_LOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


class AnswerCache:
    def __init__(self, redis: Redis, prefix: str = "mia") -> None:
        self._redis = redis
        self._prefix = prefix
        self._release_script = redis.register_script(_RELEASE_LOCK_LUA)

    def _answer_key(self, key: str) -> str:
        return f"{self._prefix}:ans:{key}"

    def _lock_key(self, key: str) -> str:
        return f"{self._prefix}:lock:{key}"

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            raw = await self._redis.get(self._answer_key(key))
        except RedisError as exc:
            logger.warning("cache_get_failed", extra={"ctx": {"error": str(exc)}})
            return None
        return json.loads(raw) if raw else None

    async def set(self, key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        try:
            await self._redis.set(
                self._answer_key(key), json.dumps(payload, ensure_ascii=False), ex=ttl_seconds
            )
        except RedisError as exc:
            logger.warning("cache_set_failed", extra={"ctx": {"error": str(exc)}})

    async def acquire_lock(self, key: str, ttl_seconds: int) -> str | None:
        """Single-flight: solo un worker de todo el clúster genera cada respuesta."""
        token = secrets.token_hex(16)
        try:
            acquired = await self._redis.set(self._lock_key(key), token, nx=True, ex=ttl_seconds)
        except RedisError as exc:
            logger.warning("lock_acquire_failed", extra={"ctx": {"error": str(exc)}})
            return token  # sin Redis, cada worker genera su propia respuesta
        return token if acquired else None

    async def release_lock(self, key: str, token: str) -> None:
        try:
            await self._release_script(keys=[self._lock_key(key)], args=[token])
        except RedisError as exc:
            logger.warning("lock_release_failed", extra={"ctx": {"error": str(exc)}})

    async def wait_for(self, key: str, timeout_seconds: float, poll_seconds: float = 0.15) -> dict[str, Any] | None:
        """Espera a que el worker que tiene el lock publique la respuesta."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            hit = await self.get(key)
            if hit is not None:
                return hit
            try:
                if not await self._redis.exists(self._lock_key(key)):
                    return await self.get(key)  # el líder terminó (o falló) sin dejar respuesta
            except RedisError:
                return None
            await asyncio.sleep(poll_seconds)
        return None


class SemanticCache:
    """Caché semántica sobre Chroma: preguntas distintas con el mismo significado
    ("¿cómo saco el DNI digital?" / "quiero tener el dni en el celular") reutilizan
    la misma respuesta. Solo guarda el embedding y un puntero a la clave de Redis:
    la respuesta vive en Redis con su TTL, así que una entrada vencida es un miss.
    """

    def __init__(self, client_getter: Callable[[], Any], collection_name: str, threshold: float) -> None:
        self._client_getter = client_getter
        self._collection_name = collection_name
        self._threshold = threshold
        self._collection: Any = None

    def _get_collection(self) -> Any:
        if self._collection is None:
            self._collection = self._client_getter().get_or_create_collection(
                name=self._collection_name,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"purpose": "semantic-cache"},
                embedding_function=None,
            )
        return self._collection

    def _with_collection(self, operation: Callable[[Any], Any]) -> Any:
        """Ejecuta la operación; si la colección fue recreada por una ingesta (el handle
        local apunta a un id que ya no existe), la vuelve a obtener y reintenta una vez."""
        try:
            return operation(self._get_collection())
        except Exception:  # noqa: BLE001
            self._collection = None
            return operation(self._get_collection())

    def _lookup_sync(self, embedding: list[float], scope: str) -> str | None:
        return self._with_collection(lambda collection: self._query(collection, embedding, scope))

    def _query(self, collection: Any, embedding: list[float], scope: str) -> str | None:
        result = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"scope": scope},
            include=["metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return None
        similarity = 1.0 - float(result["distances"][0][0])
        if similarity < self._threshold:
            return None
        return str(result["metadatas"][0][0]["answer_key"])

    def _store_sync(self, embedding: list[float], scope: str, answer_key: str, normalized_question: str) -> None:
        self._with_collection(
            lambda collection: collection.upsert(
                ids=[answer_key],
                embeddings=[embedding],
                documents=[normalized_question],
                metadatas=[{"scope": scope, "answer_key": answer_key, "created_at": int(time.time())}],
            )
        )

    async def lookup(self, embedding: list[float], scope: str) -> str | None:
        try:
            return await asyncio.to_thread(self._lookup_sync, embedding, scope)
        except Exception as exc:  # noqa: BLE001 - fail-open
            self._collection = None  # la colección pudo haber sido recreada por la ingesta
            logger.warning("semantic_cache_lookup_failed", extra={"ctx": {"error": str(exc)}})
            return None

    async def store(self, embedding: list[float], scope: str, answer_key: str, normalized_question: str) -> None:
        try:
            await asyncio.to_thread(self._store_sync, embedding, scope, answer_key, normalized_question)
        except Exception as exc:  # noqa: BLE001 - fail-open
            self._collection = None
            logger.warning("semantic_cache_store_failed", extra={"ctx": {"error": str(exc)}})
