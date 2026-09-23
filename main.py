"""API del asistente de Mi Argentina.

POST /v1/chat corta apenas puede para no gastar GPU: rate limit, limpieza de la pregunta,
prompt injection, datos personales, caché exacta, caché semántica, búsqueda en las FAQ,
single-flight, generación y control de la salida. Más detalle en el README.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.cache import AnswerCache, SemanticCache
from app.config import Settings, get_settings
from app.knowledge import KnowledgeBase, RetrievedChunk
from app.limits import LLMBusyError, LLMGate, RateLimiter, TokenBudget
from app.llm import build_chat_model, build_embeddings, check_llm, message_text
from app.log import configure_logging
from app.metrics import (
    BLOCKED,
    CACHE_HITS,
    LLM_LATENCY,
    LLM_TOKENS,
    REQUEST_LATENCY,
    REQUESTS,
    render_metrics,
)
from app.prompts import (
    BUSY_MESSAGE,
    CONTEXT_BLOCK_TEMPLATE,
    CONTEXT_SEPARATOR,
    NO_INFO_MESSAGE,
    PROMPT_FINGERPRINT,
    REFUSAL_MESSAGE,
    UNAVAILABLE_MESSAGE,
    build_system_prompt,
    build_user_prompt,
)
from app.security import (
    detect_prompt_injection,
    escape_for_prompt,
    guard_output,
    mask_pii,
    sanitize_context,
    sanitize_user_input,
)
from app.text import normalize_for_cache, stable_hash
from app.tokens import TokenCounter

logger = logging.getLogger("mi_argentina_ia")

STATIC_DIR = Path(__file__).parent / "static"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
# Si Redis cae, la API sigue respondiendo (sin caché); si cae Chroma, no hay RAG posible.
CRITICAL_CHECKS = {"chroma"}


# ---------------------------------------------------------------------------
# Esquemas de la API
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=4000, description="Consulta del ciudadano")


class Source(BaseModel):
    title: str
    url: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    cached: bool = False
    cache_layer: Literal["redis", "semantic"] | None = None
    blocked: bool = False
    block_reason: str | None = None
    kb_version: str | None = None
    usage: Usage = Field(default_factory=Usage)
    request_id: str
    latency_ms: int


@dataclass
class ChatResult:
    body: ChatResponse
    edge_cacheable: bool


# ---------------------------------------------------------------------------
# Dependencias (se construyen una vez por proceso)
# ---------------------------------------------------------------------------


@dataclass
class Services:
    settings: Settings
    redis: Any
    kb: Any  # KnowledgeBase
    embeddings: Any  # langchain_core.embeddings.Embeddings
    chat_model: Any  # langchain_core BaseChatModel
    answer_cache: AnswerCache
    semantic_cache: Any  # SemanticCache | None
    rate_limiter: RateLimiter
    token_budget: TokenBudget
    llm_gate: LLMGate
    tokens: TokenCounter
    health_checks: dict[str, Callable[[], Awaitable[bool]]] = field(default_factory=dict)
    closers: list[Callable[[], Awaitable[Any]]] = field(default_factory=list)

    async def aclose(self) -> None:
        for closer in self.closers:
            try:
                await closer()
            except Exception:  # noqa: BLE001
                logger.exception("shutdown_error")


async def build_services(settings: Settings) -> Services:
    redis = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=2,
        socket_connect_timeout=2,
        health_check_interval=30,
    )
    kb = KnowledgeBase(settings)
    http = httpx.AsyncClient()
    semantic_cache = (
        SemanticCache(kb.client, settings.semantic_cache_collection, settings.semantic_cache_threshold)
        if settings.semantic_cache_enabled
        else None
    )

    async def redis_ok() -> bool:
        try:
            return bool(await redis.ping())
        except Exception:  # noqa: BLE001
            return False

    async def chroma_ok() -> bool:
        return await kb.ping() and (await kb.active()) is not None

    async def llm_ok() -> bool:
        return await check_llm(settings, http)

    return Services(
        settings=settings,
        redis=redis,
        kb=kb,
        embeddings=build_embeddings(settings),
        chat_model=build_chat_model(settings),
        answer_cache=AnswerCache(redis),
        semantic_cache=semantic_cache,
        rate_limiter=RateLimiter(redis, settings.rate_limit_per_minute),
        token_budget=TokenBudget(redis, settings.token_budget_per_client_per_day),
        llm_gate=LLMGate(settings.max_concurrent_llm_calls, settings.llm_queue_timeout_seconds),
        tokens=TokenCounter(),
        health_checks={"redis": redis_ok, "chroma": chroma_ok, "llm": llm_ok},
        closers=[redis.aclose, http.aclose],
    )


# ---------------------------------------------------------------------------
# Utilidades del pipeline
# ---------------------------------------------------------------------------


def resolve_client_id(request: Request, settings: Settings) -> str:
    """Identificador seudónimo del cliente para rate limit y cupos (nunca se guarda la IP en claro).

    Cualquier cliente puede mandar X-Client-Id o CF-Connecting-IP inventados (y rotarlos para
    esquivar el rate limit), así que solo se usan cuando el pedido viene autenticado por el
    edge (EDGE_SHARED_SECRET, validado en verify_edge). Si no, se usa X-Real-IP, que nginx
    sobrescribe siempre con la IP real de la conexión.
    """
    identity: str | None = None
    if settings.trust_proxy_headers:
        if settings.edge_shared_secret.get_secret_value():
            identity = request.headers.get("x-client-id") or request.headers.get("cf-connecting-ip")
        identity = identity or request.headers.get("x-real-ip")
    if not identity:
        identity = request.client.host if request.client else "anonimo"
    return stable_hash("client", identity[:128])[:24]


def build_context(
    chunks: list[RetrievedChunk], counter: TokenCounter, max_tokens: int
) -> tuple[str, list[Source]]:
    """Arma el contexto oficial respetando un presupuesto máximo de tokens."""
    blocks: list[str] = []
    sources: list[Source] = []
    seen_urls: set[str] = set()
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        block = CONTEXT_BLOCK_TEMPLATE.format(
            index=index,
            title=chunk.title or "Sin título",
            url=chunk.url or "sin URL",
            content=sanitize_context(chunk.text),
        )
        cost = counter.count(block)
        if used + cost > max_tokens:
            remaining = max_tokens - used
            if remaining < 80:
                break
            block, cost = counter.truncate(block, remaining), remaining
        blocks.append(block)
        used += cost
        if chunk.url and chunk.url not in seen_urls:
            seen_urls.add(chunk.url)
            sources.append(Source(title=chunk.title or chunk.url, url=chunk.url))
    return CONTEXT_SEPARATOR.join(blocks), sources


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


async def answer_question(svc: Services, raw_question: str, client_id: str, request_id: str) -> ChatResult:
    s = svc.settings
    started = time.perf_counter()

    def finish(
        payload: dict[str, Any],
        *,
        outcome: str,
        edge_cacheable: bool,
        cache_layer: Literal["redis", "semantic"] | None = None,
        usage: Usage | None = None,
    ) -> ChatResult:
        latency_ms = int((time.perf_counter() - started) * 1000)
        REQUESTS.labels(outcome).inc()
        body = ChatResponse(
            answer=payload["answer"],
            sources=[Source(**source) for source in payload.get("sources", [])],
            cached=cache_layer is not None,
            cache_layer=cache_layer,
            blocked=bool(payload.get("blocked", False)),
            block_reason=payload.get("block_reason"),
            kb_version=payload.get("kb_version"),
            usage=usage or Usage(),
            request_id=request_id,
            latency_ms=latency_ms,
        )
        logger.info(
            "chat",
            extra={
                "ctx": {
                    "request_id": request_id,
                    "client": client_id,
                    "outcome": outcome,
                    "cache_layer": cache_layer,
                    "block_reason": body.block_reason,
                    "latency_ms": latency_ms,
                    "question_chars": len(raw_question),
                    "input_tokens": body.usage.input_tokens,
                    "output_tokens": body.usage.output_tokens,
                }
            },
        )
        return ChatResult(body=body, edge_cacheable=edge_cacheable)

    # 2) Sanitización y límites de tamaño -------------------------------------
    question = sanitize_user_input(raw_question)
    if not question:
        raise HTTPException(422, "La consulta está vacía.")
    if len(question) > s.max_question_chars:
        raise HTTPException(422, f"La consulta supera el máximo de {s.max_question_chars} caracteres.")
    if svc.tokens.count(question) > s.max_question_tokens:
        raise HTTPException(422, "La consulta es demasiado extensa. Resumila en una o dos oraciones.")

    # 3) Prompt injection y 4) datos personales ----------------------------------
    attack_rule = detect_prompt_injection(question)
    masked_question, pii_types = mask_pii(question)
    # Una consulta con datos personales nunca se guarda en el edge (CDN compartida).
    edge_cacheable = not pii_types
    if attack_rule:
        BLOCKED.labels("prompt_injection").inc()
        logger.warning(
            "prompt_injection_blocked",
            extra={"ctx": {"request_id": request_id, "client": client_id, "rule": attack_rule}},
        )
        return finish(
            {"answer": REFUSAL_MESSAGE, "blocked": True, "block_reason": "prompt_injection"},
            outcome="blocked_injection",
            edge_cacheable=edge_cacheable,
        )

    # 5) Caché exacta ---------------------------------------------------------
    kb = await svc.kb.active()
    if kb is None:
        raise HTTPException(503, UNAVAILABLE_MESSAGE, headers={"Retry-After": "30"})
    normalized = normalize_for_cache(masked_question)
    # La versión del índice y del prompt forman parte de la clave: re-indexar invalida sola la caché.
    scope = stable_hash(s.prompt_version, PROMPT_FINGERPRINT, s.chat_model_name, kb.version)[:16]
    cache_key = stable_hash(scope, normalized)

    cached = await svc.answer_cache.get(cache_key)
    if cached is not None:
        CACHE_HITS.labels("redis").inc()
        return finish(cached, outcome="cache_hit", edge_cacheable=edge_cacheable, cache_layer="redis")

    # 6) Embedding + caché semántica --------------------------------------------
    try:
        embedding = await asyncio.wait_for(
            svc.embeddings.aembed_query(masked_question), timeout=s.embeddings_timeout_seconds
        )
    except Exception as exc:
        logger.error("embedding_failed", extra={"ctx": {"request_id": request_id, "error": repr(exc)}})
        raise HTTPException(503, UNAVAILABLE_MESSAGE, headers={"Retry-After": "10"}) from exc

    if svc.semantic_cache is not None:
        similar_key = await svc.semantic_cache.lookup(embedding, scope)
        if similar_key:
            cached = await svc.answer_cache.get(similar_key)
            if cached is not None:
                CACHE_HITS.labels("semantic").inc()
                await svc.answer_cache.set(cache_key, cached, s.cache_ttl_seconds)  # promoción a caché exacta
                return finish(cached, outcome="cache_hit", edge_cacheable=edge_cacheable, cache_layer="semantic")

    # 7) Recuperación con umbral de relevancia ------------------------------------
    try:
        chunks = await svc.kb.search(kb, embedding, s.retrieval_top_k)
    except Exception as exc:
        logger.error("retrieval_failed", extra={"ctx": {"request_id": request_id, "error": repr(exc)}})
        raise HTTPException(503, UNAVAILABLE_MESSAGE, headers={"Retry-After": "10"}) from exc

    best_score = max((chunk.score for chunk in chunks), default=0.0)
    cutoff = max(s.min_relevance_score, best_score - s.retrieval_score_margin)
    relevant = [chunk for chunk in chunks if chunk.score >= cutoff]
    if not relevant:
        # Nada en la base oficial se parece a la consulta: fuera de alcance, sin tocar el LLM.
        BLOCKED.labels("out_of_scope").inc()
        payload = {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "blocked": True,
            "block_reason": "out_of_scope",
            "kb_version": kb.version,
        }
        await svc.answer_cache.set(cache_key, payload, s.refusal_cache_ttl_seconds)
        return finish(payload, outcome="blocked_out_of_scope", edge_cacheable=edge_cacheable)

    # 8) Single-flight ------------------------------------------------------------
    lock_token = await svc.answer_cache.acquire_lock(cache_key, s.single_flight_lock_seconds)
    if lock_token is None:
        cached = await svc.answer_cache.wait_for(cache_key, s.single_flight_wait_seconds)
        if cached is not None:
            CACHE_HITS.labels("coalesced").inc()
            return finish(cached, outcome="cache_hit", edge_cacheable=edge_cacheable, cache_layer="redis")
        # El líder falló o tardó demasiado: se genera la respuesta en este worker.

    try:
        # 9) Presupuesto de tokens, prompt y generación ---------------------------
        if await svc.token_budget.remaining(client_id) <= 0:
            BLOCKED.labels("token_budget").inc()
            raise HTTPException(
                429,
                "Alcanzaste el límite diario de consultas al asistente. Intentá nuevamente mañana.",
                headers={"Retry-After": "3600"},
            )

        context, sources = build_context(relevant, svc.tokens, s.max_context_tokens)
        canary = secrets.token_hex(6)  # si aparece en la salida, hubo fuga del system prompt
        messages = [
            SystemMessage(content=build_system_prompt(canary)),
            HumanMessage(content=build_user_prompt(context, escape_for_prompt(masked_question))),
        ]

        try:
            async with svc.llm_gate.slot():
                llm_started = time.perf_counter()
                ai_message = await asyncio.wait_for(svc.chat_model.ainvoke(messages), timeout=s.llm_timeout_seconds)
                LLM_LATENCY.observe(time.perf_counter() - llm_started)
        except LLMBusyError as exc:
            BLOCKED.labels("llm_busy").inc()
            raise HTTPException(503, BUSY_MESSAGE, headers={"Retry-After": "5"}) from exc
        except asyncio.TimeoutError as exc:
            logger.error("llm_timeout", extra={"ctx": {"request_id": request_id}})
            raise HTTPException(504, BUSY_MESSAGE, headers={"Retry-After": "5"}) from exc
        except Exception as exc:
            logger.error("llm_failed", extra={"ctx": {"request_id": request_id, "error": repr(exc)}})
            raise HTTPException(503, UNAVAILABLE_MESSAGE, headers={"Retry-After": "10"}) from exc

        raw_answer = message_text(ai_message)
        usage_metadata = getattr(ai_message, "usage_metadata", None) or {}
        usage = Usage(
            input_tokens=int(
                usage_metadata.get("input_tokens")
                or svc.tokens.count(str(messages[0].content) + str(messages[1].content))
            ),
            output_tokens=int(usage_metadata.get("output_tokens") or svc.tokens.count(raw_answer)),
        )
        LLM_TOKENS.labels("input").inc(usage.input_tokens)
        LLM_TOKENS.labels("output").inc(usage.output_tokens)
        await svc.token_budget.consume(client_id, usage.input_tokens + usage.output_tokens)

        # 10) Guard de salida y caché ---------------------------------------------
        guarded = guard_output(raw_answer, canary, s.allowed_link_domains_list)
        if guarded.violation:
            BLOCKED.labels(guarded.violation).inc()
            logger.warning(
                "output_guard_triggered",
                extra={"ctx": {"request_id": request_id, "violation": guarded.violation}},
            )
        is_refusal = guarded.text == REFUSAL_MESSAGE
        is_fallback = is_refusal or guarded.text == NO_INFO_MESSAGE
        payload = {
            "answer": guarded.text,
            "sources": [] if is_fallback else [source.model_dump() for source in sources],
            "blocked": is_refusal,
            "block_reason": guarded.violation or ("out_of_scope" if is_refusal else None),
            "kb_version": kb.version,
        }
        ttl = s.refusal_cache_ttl_seconds if (is_fallback or guarded.violation) else s.cache_ttl_seconds
        await svc.answer_cache.set(cache_key, payload, ttl)
        if svc.semantic_cache is not None and not is_fallback and not guarded.violation:
            await svc.semantic_cache.store(embedding, scope, cache_key, normalized)

        return finish(payload, outcome="generated", edge_cacheable=edge_cacheable, usage=usage)
    finally:
        if lock_token is not None:
            await svc.answer_cache.release_lock(cache_key, lock_token)


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    services_factory: Callable[[Settings], Awaitable[Services]] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    factory = services_factory or build_services
    is_production = settings.app_env == "production"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = await factory(settings)
        logger.info(
            "startup",
            extra={
                "ctx": {
                    "version": __version__,
                    "llm_provider": settings.llm_provider,
                    "chat_model": settings.chat_model_name,
                    "embed_model": settings.embed_model_name,
                }
            },
        )
        try:
            yield
        finally:
            await app.state.services.aclose()

    app = FastAPI(
        title="Asistente IA - Mi Argentina",
        version=__version__,
        lifespan=lifespan,
        docs_url=None if is_production else "/docs",
        redoc_url=None,
        openapi_url=None if is_production else "/openapi.json",
    )

    if settings.cors_origins_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Request-ID", "X-Client-Id"],
            max_age=600,
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        path = request.url.path
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if path.startswith("/demo"):
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            )
        elif not path.startswith(("/docs", "/openapi")):
            response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        headers = dict(exc.headers or {})
        headers["Cache-Control"] = "no-store"  # nunca cachear errores en el edge
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=headers)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"ctx": {"path": request.url.path}})
        return JSONResponse({"detail": UNAVAILABLE_MESSAGE}, status_code=500, headers={"Cache-Control": "no-store"})

    def verify_edge(request: Request) -> None:
        """Si hay secreto configurado, solo se aceptan pedidos que pasaron por el edge (Cloudflare)."""
        secret = settings.edge_shared_secret.get_secret_value()
        if not secret:
            return
        received = request.headers.get("x-edge-auth", "")
        if not secrets.compare_digest(received.encode(), secret.encode()):
            raise HTTPException(401, "No autorizado.")

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(verify_edge)])
    async def chat(body: ChatRequest, request: Request, response: Response) -> ChatResponse:
        svc: Services = request.app.state.services
        incoming_id = request.headers.get("x-request-id", "")
        request_id = incoming_id if _REQUEST_ID.match(incoming_id) else uuid.uuid4().hex
        client_id = resolve_client_id(request, settings)

        # 1) Rate limit por cliente (compartido entre réplicas)
        allowed, retry_after = await svc.rate_limiter.hit(client_id)
        if not allowed:
            BLOCKED.labels("rate_limit").inc()
            raise HTTPException(
                429,
                "Hiciste demasiadas consultas seguidas. Esperá un momento e intentá de nuevo.",
                headers={"Retry-After": str(retry_after)},
            )

        with REQUEST_LATENCY.time():
            result = await answer_question(svc, body.question, client_id, request_id)

        answer = result.body
        if answer.cached:
            x_cache = f"HIT-{(answer.cache_layer or 'redis').upper()}"
        else:
            x_cache = "BLOCKED" if answer.blocked else "MISS"
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Cache"] = x_cache
        response.headers["Cache-Control"] = (
            f"public, max-age=0, s-maxage={settings.edge_cache_max_age}"
            if result.edge_cacheable
            else "private, no-store"
        )
        return answer

    @app.get("/health/live", include_in_schema=False)
    async def live() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/health/ready", include_in_schema=False)
    async def ready(request: Request) -> JSONResponse:
        svc: Services = request.app.state.services
        names = list(svc.health_checks)
        results = await asyncio.gather(*(check() for check in svc.health_checks.values()), return_exceptions=True)
        checks = {name: result is True for name, result in zip(names, results)}
        is_ready = all(checks.get(name, False) for name in CRITICAL_CHECKS if name in checks)
        return JSONResponse(
            {"status": "ready" if is_ready else "not_ready", "checks": checks},
            status_code=200 if is_ready else 503,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        data, content_type = render_metrics()
        return Response(content=data, media_type=content_type)

    if settings.demo_ui_enabled and STATIC_DIR.is_dir():
        app.mount("/demo", StaticFiles(directory=STATIC_DIR, html=True), name="demo")

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse("/demo/")

    return app


app = create_app()
