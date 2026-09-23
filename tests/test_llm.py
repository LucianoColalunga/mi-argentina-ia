"""Construye los modelos reales de LangChain (sin red) para detectar parámetros inválidos."""

import pytest
from langchain_core.messages import AIMessage

from app.config import Settings
from app.llm import build_chat_model, build_embeddings, message_text


@pytest.mark.parametrize(
    ("llm_provider", "embeddings_provider"),
    [("ollama", "ollama"), ("vllm", "ollama"), ("vllm", "openai")],
)
def test_factories_build_real_models(llm_provider: str, embeddings_provider: str) -> None:
    settings = Settings(_env_file=None, llm_provider=llm_provider, embeddings_provider=embeddings_provider)
    chat_model = build_chat_model(settings)
    embeddings = build_embeddings(settings)
    assert chat_model is not None and embeddings is not None
    if llm_provider == "ollama":
        assert chat_model.num_predict == settings.max_output_tokens
    else:
        assert chat_model.max_tokens == settings.max_output_tokens


def test_message_text_handles_content_blocks() -> None:
    assert message_text(AIMessage(content="hola")) == "hola"
    blocks = AIMessage(content=[{"type": "text", "text": "ho"}, {"type": "text", "text": "la"}])
    assert message_text(blocks) == "hola"


def test_system_prompt_only_differs_in_last_line() -> None:
    """El canario va al final para que el prefijo se reutilice (prefix caching)."""
    from app.prompts import build_system_prompt

    first, second = build_system_prompt("aaaaaa").splitlines(), build_system_prompt("bbbbbb").splitlines()
    assert first[:-1] == second[:-1]
    assert first[-1] != second[-1] and first[-1].endswith("aaaaaa")
