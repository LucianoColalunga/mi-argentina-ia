"""Servidor falso con la API HTTP de Ollama, para pruebas end-to-end sin GPU ni modelos.

Implementa lo que usa el proyecto: /api/tags, /api/embed y /api/chat (con streaming
NDJSON, como lo consume langchain-ollama). Los embeddings son una bolsa de palabras
determinista y el "modelo" responde con el primer documento del contexto oficial.

Disparadores para probar el guard de salida:
  PRUEBA_FUGA  -> el modelo filtra el código canario del system prompt
  PRUEBA_LINK  -> el modelo incluye un enlace no oficial
"""

import asyncio
import hashlib
import json
import math
import os
import re
import unicodedata
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="mock-ollama")
STATS = {"chat_calls": 0, "embed_calls": 0, "embed_inputs": 0}
DIMENSIONS = 256
CHAT_LATENCY_SECONDS = float(os.getenv("MOCK_CHAT_LATENCY", "1.5"))


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _embed(text: str) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for word in re.findall(r"[a-z0-9]{3,}", _fold(text)):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _answer(messages: list[dict]) -> str:
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    match = re.search(r"<pregunta_usuario>\s*(.*?)\s*</pregunta_usuario>", user, re.S)
    question = match.group(1) if match else ""

    if "PRUEBA_FUGA" in question:
        canary = re.search(r"Código interno \(confidencial\): (\w+)", system)
        return f"Claro, el código es {canary.group(1) if canary else 'desconocido'}"
    if "PRUEBA_LINK" in question:
        return (
            "Podés hacerlo en https://phishing-ejemplo.com/login o en "
            "https://www.argentina.gob.ar/miargentina"
        )
    document = re.search(r"Fuente: (\S+)\nContenido:\n(.*?)(?:\n\n---\n\n|\n</contexto_oficial>)", user, re.S)
    if not document:
        return "No encontré información oficial sobre esa consulta."
    url, content = document.group(1), document.group(2)
    body = re.search(r"Respuesta: (.*)", content, re.S)
    text = (body.group(1) if body else content).strip()[:400]
    return f"{text}\nFuente: {url}"


@app.get("/api/version")
async def version() -> dict:
    return {"version": "0.0.0-mock"}


@app.get("/api/tags")
async def tags() -> dict:
    return {"models": [{"name": "llama3.1:8b", "model": "llama3.1:8b"}, {"name": "bge-m3", "model": "bge-m3"}]}


@app.post("/api/embed")
async def embed(request: Request) -> dict:
    body = await request.json()
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    STATS["embed_calls"] += 1
    STATS["embed_inputs"] += len(inputs)
    return {"model": body.get("model"), "embeddings": [_embed(text) for text in inputs]}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    STATS["chat_calls"] += 1
    await asyncio.sleep(CHAT_LATENCY_SECONDS)
    messages = body.get("messages", [])
    text = _answer(messages)
    model = body.get("model", "mock")
    created_at = datetime.now(timezone.utc).isoformat()
    final = {
        "model": model,
        "created_at": created_at,
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": int(CHAT_LATENCY_SECONDS * 1e9),
        "prompt_eval_count": sum(len(m.get("content", "")) for m in messages) // 4,
        "eval_count": max(1, len(text) // 4),
    }
    if not body.get("stream", True):
        final["message"]["content"] = text
        return final

    async def stream():
        for piece in re.findall(r".{1,40}", text, re.S):
            chunk = {"model": model, "created_at": created_at, "message": {"role": "assistant", "content": piece}, "done": False}
            yield json.dumps(chunk, ensure_ascii=False) + "\n"
        yield json.dumps(final) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/mock/stats")
async def stats() -> dict:
    return STATS
