"""Métricas Prometheus (compatibles con gunicorn multi-proceso)."""

from __future__ import annotations

import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)

# En modo multi-proceso los valores se escriben en archivos: el directorio debe existir
# también cuando la app se carga fuera de gunicorn (uvicorn directo, scripts, tests).
if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
    os.makedirs(os.environ["PROMETHEUS_MULTIPROC_DIR"], exist_ok=True)

REQUESTS = Counter("mia_chat_requests_total", "Consultas procesadas por resultado", ["outcome"])
CACHE_HITS = Counter("mia_cache_hits_total", "Aciertos de caché por capa", ["layer"])
BLOCKED = Counter("mia_blocked_total", "Consultas bloqueadas por motivo", ["reason"])
LLM_TOKENS = Counter("mia_llm_tokens_total", "Tokens procesados por el LLM", ["kind"])
LLM_LATENCY = Histogram(
    "mia_llm_latency_seconds",
    "Latencia de la generación del LLM",
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60),
)
REQUEST_LATENCY = Histogram(
    "mia_chat_latency_seconds",
    "Latencia total de /v1/chat",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)


def render_metrics() -> tuple[bytes, str]:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
