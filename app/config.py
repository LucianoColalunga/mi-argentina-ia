"""Configuración centralizada (12-factor): todo se controla por variables de entorno."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- General ---
    app_name: str = "mi-argentina-ia"
    app_env: Literal["development", "production", "test"] = "production"
    log_level: str = "INFO"
    demo_ui_enabled: bool = True

    # --- Modelo de lenguaje ---
    llm_provider: Literal["ollama", "vllm"] = "ollama"
    ollama_base_url: str = "http://ollama:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_keep_alive: str = "24h"
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    vllm_api_key: SecretStr = SecretStr("cambiar-esta-clave")
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192
    llm_timeout_seconds: float = 60.0
    max_concurrent_llm_calls: int = 32
    llm_queue_timeout_seconds: float = 10.0

    # --- Embeddings ---
    embeddings_provider: Literal["ollama", "openai"] = "ollama"
    ollama_embed_model: str = "bge-m3"
    embeddings_base_url: str = "http://tei:80/v1"
    embeddings_model: str = "BAAI/bge-m3"
    embeddings_api_key: SecretStr = SecretStr("no-key")
    embeddings_timeout_seconds: float = 15.0

    # --- Base vectorial / RAG ---
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    kb_collection_prefix: str = "mi_argentina_faq"
    kb_meta_collection: str = "mi_argentina_meta"
    kb_refresh_seconds: int = 30
    retrieval_top_k: int = 4
    # Calibrado con bge-m3 (tests/e2e/calibrate.py): trámites >= 0,62, temas ajenos <= 0,45.
    min_relevance_score: float = 0.52
    # Solo se envían al modelo los documentos cercanos al mejor: menos ruido, menos
    # alucinaciones y menos tokens de entrada.
    retrieval_score_margin: float = 0.08

    # --- Control de tokens ---
    max_question_chars: int = 1000
    max_question_tokens: int = 256
    max_context_tokens: int = 2500
    max_output_tokens: int = 512
    token_budget_per_client_per_day: int = 100_000

    # --- Caché ---
    redis_url: str = "redis://redis:6379/0"
    cache_ttl_seconds: int = 86_400
    refusal_cache_ttl_seconds: int = 3_600
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95
    semantic_cache_collection: str = "mi_argentina_semantic_cache"
    single_flight_lock_seconds: int = 90
    single_flight_wait_seconds: float = 20.0
    edge_cache_max_age: int = 3_600

    # --- Seguridad ---
    rate_limit_per_minute: int = 20
    trust_proxy_headers: bool = True
    edge_shared_secret: SecretStr = SecretStr("")
    cors_allow_origins: str = ""
    allowed_link_domains: str = "argentina.gob.ar,gob.ar"
    prompt_version: str = "v1"

    @property
    def cors_origins_list(self) -> list[str]:
        return _csv(self.cors_allow_origins)

    @property
    def allowed_link_domains_list(self) -> list[str]:
        return [d.lower() for d in _csv(self.allowed_link_domains)]

    @property
    def chat_model_name(self) -> str:
        return self.ollama_chat_model if self.llm_provider == "ollama" else self.vllm_model

    @property
    def embed_model_name(self) -> str:
        return self.ollama_embed_model if self.embeddings_provider == "ollama" else self.embeddings_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
