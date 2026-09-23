"""Diagnóstico: arma el mismo prompt que la API y muestra la salida CRUDA del modelo,
antes del guard de salida, junto con el veredicto del guard.

  docker compose -f docker-compose.yml -f docker-compose.local.yml run --rm --no-deps \
      -e PYTHONPATH=/app -v ./tests/e2e:/e2e:ro api python /e2e/debug_llm.py "pregunta"
"""

import asyncio
import secrets
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.knowledge import KnowledgeBase
from app.llm import build_chat_model, build_embeddings, message_text
from app.prompts import build_system_prompt, build_user_prompt
from app.security import _LEAK_MARKERS, escape_for_prompt, guard_output
from app.text import fold
from app.tokens import TokenCounter
from main import build_context


async def run(question: str) -> None:
    settings = get_settings()
    kb = KnowledgeBase(settings)
    info = await kb.active()
    vector = await build_embeddings(settings).aembed_query(question)
    found = await kb.search(info, vector, settings.retrieval_top_k)
    cutoff = max(settings.min_relevance_score, max((c.score for c in found), default=0) - settings.retrieval_score_margin)
    chunks = [c for c in found if c.score >= cutoff]
    print(f"P: {question}\nchunks relevantes: {[round(c.score, 3) for c in chunks]}")
    if not chunks:
        print("-> fuera de alcance (no llama al modelo)")
        return
    context, _ = build_context(chunks, TokenCounter(), settings.max_context_tokens)
    canary = secrets.token_hex(6)
    messages = [
        SystemMessage(content=build_system_prompt(canary)),
        HumanMessage(content=build_user_prompt(context, escape_for_prompt(question))),
    ]
    started = time.perf_counter()
    raw = message_text(await build_chat_model(settings).ainvoke(messages))
    print(f"--- salida cruda ({time.perf_counter() - started:.1f} s) ---\n{raw}\n---")
    print(f"marcadores de fuga presentes: {[m for m in _LEAK_MARKERS if m in fold(raw)]}")
    result = guard_output(raw, canary, settings.allowed_link_domains_list)
    print(f"guard: violation={result.violation}\n{'=' * 100}")



async def main_all() -> None:
    for q in sys.argv[1:]:
        await run(q)


asyncio.run(main_all())
