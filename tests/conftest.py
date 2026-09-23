"""Dobles de prueba: Redis en memoria (fakeredis), índice vectorial y LLM falsos.

Permiten ejecutar el pipeline completo de /v1/chat sin GPU, Chroma ni Ollama.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.cache import AnswerCache
from app.config import Settings
from app.knowledge import KBInfo, RetrievedChunk
from app.limits import LLMGate, RateLimiter, TokenBudget
from app.text import fold
from app.tokens import TokenCounter
from main import Services, create_app

DIMENSIONS = 256


def fake_embed(text: str) -> list[float]:
    """Bolsa de palabras hasheada y normalizada: similitud coseno con sentido léxico."""
    vector = [0.0] * DIMENSIONS
    for word in re.findall(r"[a-z0-9]{3,}", fold(text)):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class FakeEmbeddings:
    async def aembed_query(self, text: str) -> list[float]:
        return fake_embed(text)


@dataclass
class FakeKnowledgeBase:
    chunks: list[tuple[str, str, str]]  # (texto, título, url)
    version: str = "test-v1"

    async def active(self) -> KBInfo:
        return KBInfo(collection="faq_test", version=self.version)

    async def search(self, info: KBInfo, embedding: list[float], k: int) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(id=str(i), text=text, title=title, url=url, score=cosine(embedding, fake_embed(text)))
            for i, (text, title, url) in enumerate(self.chunks)
        ]
        return sorted(scored, key=lambda chunk: chunk.score, reverse=True)[:k]


@dataclass
class FakeSemanticCache:
    threshold: float = 0.95
    entries: list[tuple[list[float], str, str]] = field(default_factory=list)

    async def lookup(self, embedding: list[float], scope: str) -> str | None:
        best = max(
            ((cosine(embedding, vec), key) for vec, entry_scope, key in self.entries if entry_scope == scope),
            default=(0.0, None),
        )
        return best[1] if best[0] >= self.threshold else None

    async def store(self, embedding: list[float], scope: str, answer_key: str, normalized_question: str) -> None:
        self.entries.append((embedding, scope, answer_key))


@dataclass
class FakeChatModel:
    """Devuelve `reply` (o reply(mensajes)) y registra cada invocación."""

    reply: str | Callable[[list[Any]], str] = "Respuesta de prueba.\nFuente: https://www.argentina.gob.ar/miargentina"
    calls: list[list[Any]] = field(default_factory=list)

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        text = self.reply(messages) if callable(self.reply) else self.reply
        return AIMessage(
            content=text,
            usage_metadata={"input_tokens": 900, "output_tokens": 60, "total_tokens": 960},
        )


KB_CHUNKS = [
    (
        "Trámite: DNI en tu celular\nPregunta: ¿Cómo activo el DNI digital en el celular?\n"
        "Respuesta: Ingresá a Mi Argentina, Mis trámites, DNI digital activación, ingresá el código y creá un PIN.",
        "DNI en tu celular",
        "https://www.argentina.gob.ar/interior/dni-en-tu-celular/preguntas-frecuentes-dni-en-tu-celular",
    ),
    (
        "Trámite: Cuenta Mi Argentina\nPregunta: ¿Cómo recupero mi contraseña de Mi Argentina?\n"
        "Respuesta: Usá tu CUIL y el correo o teléfono verificado para recuperar la contraseña.",
        "Cuenta Mi Argentina",
        "https://www.argentina.gob.ar/miargentina/preguntasfrecuentes",
    ),
]


BASE_SETTINGS: dict[str, Any] = {
    "app_env": "test",
    "rate_limit_per_minute": 5,
    "min_relevance_score": 0.3,
    "semantic_cache_threshold": 0.95,
    "max_question_chars": 300,
    "llm_queue_timeout_seconds": 0.2,
}


@pytest.fixture
def chat_model() -> FakeChatModel:
    return FakeChatModel()


@pytest.fixture
def make_client(chat_model: FakeChatModel) -> Callable[..., TestClient]:
    def factory(max_concurrent_llm_calls: int = 4, **overrides: Any) -> TestClient:
        test_settings = Settings(_env_file=None, **{**BASE_SETTINGS, **overrides})

        async def build(s: Settings) -> Services:
            redis = fakeredis.FakeAsyncRedis(decode_responses=True)
            return Services(
                settings=s,
                redis=redis,
                kb=FakeKnowledgeBase(KB_CHUNKS),
                embeddings=FakeEmbeddings(),
                chat_model=chat_model,
                answer_cache=AnswerCache(redis),
                semantic_cache=FakeSemanticCache(threshold=s.semantic_cache_threshold),
                rate_limiter=RateLimiter(redis, s.rate_limit_per_minute),
                token_budget=TokenBudget(redis, s.token_budget_per_client_per_day),
                llm_gate=LLMGate(max_concurrent_llm_calls, s.llm_queue_timeout_seconds),
                tokens=TokenCounter(),
                health_checks={},
                closers=[redis.aclose],
            )

        return TestClient(create_app(settings=test_settings, services_factory=build))

    return factory
