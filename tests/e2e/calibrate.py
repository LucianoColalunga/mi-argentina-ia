"""Calibra MIN_RELEVANCE_SCORE con los embeddings reales configurados.

Imprime la similitud del mejor documento para preguntas dentro y fuera de alcance.
El umbral debe quedar por encima de las "fuera" y por debajo de las "dentro".

  docker compose -f docker-compose.yml -f docker-compose.local.yml run --rm --no-deps \
      -v ./tests/e2e:/e2e:ro api python /e2e/calibrate.py
"""

import asyncio

from app.config import get_settings
from app.knowledge import KnowledgeBase
from app.llm import build_embeddings

IN_SCOPE = [
    "¿Cómo activo el DNI en el celular?",
    "quiero tener el documento en el teléfono",
    "Olvidé mi contraseña de Mi Argentina",
    "no puedo entrar a mi cuenta, perdí la clave",
    "¿Cómo valido mi identidad?",
    "la selfie para validar no me funciona",
    "¿Puedo viajar a Europa con el DNI digital?",
    "me robaron el celular",
    "¿Cuánto cuesta el DNI digital?",
    "¿Cómo agrego a mis hijos?",
    "¿Cómo borro mi cuenta?",
    "¿Puedo prestarle el auto a mi hermano con la cédula digital?",
]
OUT_OF_SCOPE = [
    "¿Quién ganó el mundial de 1986?",
    "Pasame una receta de empanadas",
    "¿Cuál es la capital de Francia?",
    "Escribime un poema sobre el mar",
    "¿Qué opinás del presidente?",
    "¿Cómo configuro un router wifi?",
    "Recomendame una película",
    "¿Cuánto es 2 + 2?",
]


async def main() -> None:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    kb = KnowledgeBase(settings)
    info = await kb.active()
    print(f"índice={info.collection} embeddings={settings.embed_model_name} umbral={settings.min_relevance_score}\n")

    async def best(question: str) -> tuple[float, str]:
        vector = await embeddings.aembed_query(question)
        chunks = await kb.search(info, vector, 1)
        return chunks[0].score, chunks[0].text.split("\n")[1][:70] if chunks else ""

    groups = {}
    for label, questions in (("DENTRO", IN_SCOPE), ("FUERA", OUT_OF_SCOPE)):
        scores = []
        for question in questions:
            score, match = await best(question)
            scores.append(score)
            print(f"{label:6} {score:.3f}  {question[:55]:55} -> {match}")
        groups[label] = scores
        print()
    print(f"mínimo DENTRO = {min(groups['DENTRO']):.3f}   máximo FUERA = {max(groups['FUERA']):.3f}")


asyncio.run(main())
