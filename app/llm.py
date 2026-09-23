"""Fábricas de LangChain para el modelo de chat y los embeddings (Ollama o vLLM)."""

from __future__ import annotations

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_predict=settings.max_output_tokens,  # tope duro de tokens de salida
            num_ctx=settings.llm_num_ctx,
            keep_alive=settings.ollama_keep_alive,
            client_kwargs={"timeout": settings.llm_timeout_seconds},
        )
    # vLLM expone una API compatible con OpenAI (continuous batching + prefix caching).
    return ChatOpenAI(
        model=settings.vllm_model,
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.max_output_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embeddings_provider == "ollama":
        # OllamaEmbeddings solo acepta keep_alive en segundos (int); el modelo queda
        # residente igual por OLLAMA_KEEP_ALIVE del servidor (docker-compose.yml).
        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
            client_kwargs={"timeout": settings.embeddings_timeout_seconds},
        )
    # Servidor de embeddings OpenAI-compatible (Text Embeddings Inference, vLLM, etc.).
    return OpenAIEmbeddings(
        model=settings.embeddings_model,
        base_url=settings.embeddings_base_url,
        api_key=settings.embeddings_api_key,
        check_embedding_ctx_length=False,  # no tokenizar con tiktoken: el servidor no es OpenAI
        timeout=settings.embeddings_timeout_seconds,
        max_retries=1,
    )


async def check_llm(settings: Settings, http: httpx.AsyncClient) -> bool:
    """Chequeo liviano de disponibilidad del servidor de modelos (no genera tokens)."""
    try:
        if settings.llm_provider == "ollama":
            response = await http.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        else:
            response = await http.get(
                f"{settings.vllm_base_url}/models",
                headers={"Authorization": f"Bearer {settings.vllm_api_key.get_secret_value()}"},
                timeout=3,
            )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def message_text(message: BaseMessage) -> str:
    """Texto de un AIMessage, sea `content` un string o una lista de bloques."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
