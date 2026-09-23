"""Protección de capacidad: rate limiting, presupuesto de tokens y backpressure hacia el LLM."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RateLimiter:
    """Ventana fija por minuto, compartida por todas las réplicas vía Redis."""

    def __init__(self, redis: Redis, limit_per_minute: int, prefix: str = "mia") -> None:
        self._redis = redis
        self._limit = limit_per_minute
        self._prefix = prefix

    async def hit(self, client_id: str) -> tuple[bool, int]:
        """Registra un pedido. Devuelve (permitido, segundos hasta la próxima ventana)."""
        if self._limit <= 0:
            return True, 0
        now = time.time()
        window = int(now // 60)
        key = f"{self._prefix}:rl:{client_id}:{window}"
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, 65)
                count, _ = await pipe.execute()
        except RedisError as exc:
            logger.warning("rate_limit_unavailable", extra={"ctx": {"error": str(exc)}})
            return True, 0  # fail-open: el edge y nginx también limitan
        retry_after = 60 - int(now % 60)
        return int(count) <= self._limit, retry_after


class TokenBudget:
    """Cupo diario de tokens de LLM por cliente (las respuestas cacheadas no consumen)."""

    def __init__(self, redis: Redis, tokens_per_day: int, prefix: str = "mia") -> None:
        self._redis = redis
        self._tokens_per_day = tokens_per_day
        self._prefix = prefix

    def _key(self, client_id: str) -> str:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{self._prefix}:tok:{client_id}:{day}"

    async def remaining(self, client_id: str) -> int:
        if self._tokens_per_day <= 0:
            return 1
        try:
            used = await self._redis.get(self._key(client_id))
        except RedisError as exc:
            logger.warning("token_budget_unavailable", extra={"ctx": {"error": str(exc)}})
            return self._tokens_per_day
        return self._tokens_per_day - int(used or 0)

    async def consume(self, client_id: str, tokens: int) -> None:
        if self._tokens_per_day <= 0 or tokens <= 0:
            return
        key = self._key(client_id)
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.incrby(key, tokens)
                pipe.expire(key, 26 * 3600)
                await pipe.execute()
        except RedisError as exc:
            logger.warning("token_budget_unavailable", extra={"ctx": {"error": str(exc)}})


class LLMBusyError(Exception):
    """No hubo lugar en la cola del LLM dentro del tiempo de espera."""


class LLMGate:
    """Limita las llamadas concurrentes al modelo por proceso.

    Con N réplicas x W workers, la concurrencia máxima hacia vLLM/Ollama es
    N * W * max_concurrent. Si el modelo está saturado, es preferible responder
    503 rápido (el cliente reintenta con backoff) que acumular pedidos colgados.
    """

    def __init__(self, max_concurrent: int, queue_timeout_seconds: float) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_timeout = queue_timeout_seconds

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._queue_timeout)
        except asyncio.TimeoutError as exc:
            raise LLMBusyError from exc
        try:
            yield
        finally:
            self._semaphore.release()
