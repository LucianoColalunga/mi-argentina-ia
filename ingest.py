#!/usr/bin/env python
"""Ingesta de FAQs oficiales de Argentina.gob.ar en ChromaDB (índice blue/green).

Pipeline: carga (JSON/Markdown locales y/o páginas oficiales) -> limpieza ->
deduplicación -> chunking -> embeddings -> colección nueva -> cambio atómico del
puntero activo -> invalidación de la caché semántica -> limpieza de índices viejos.

Uso:
  python ingest.py                  # FAQs locales de data/faqs -> índice nuevo
  python ingest.py --from-web       # además, páginas oficiales listadas en data/sources.txt
  python ingest.py --if-empty       # solo si todavía no hay un índice activo (arranque de compose)
  python ingest.py --dry-run        # procesa y muestra los chunks sin escribir nada
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.knowledge import (
    COLLECTION_CONFIGURATION,
    META_ACTIVE_COLLECTION,
    META_CHUNKS,
    META_EMBED_MODEL,
    META_KB_VERSION,
    META_UPDATED_AT,
    create_chroma_client,
)
from app.llm import build_embeddings
from app.text import clean_text, fold, stable_hash

logger = logging.getLogger("ingest")

BASE_DIR = Path(__file__).parent
USER_AGENT = "MiArgentinaIA-Ingest/1.0 (indexador interno de preguntas frecuentes oficiales)"
ALLOWED_SCRAPE_DOMAINS = ("gob.ar",)
MIN_ANSWER_CHARS = 30

# Líneas de navegación / pie de página que no aportan al conocimiento (texto sin tildes).
_BOILERPLATE = [
    re.compile(pattern)
    for pattern in (
        r"^¿?te sirvio esta informacion\??$",
        r"^compartir( en .*)?$",
        r"^(facebook|twitter|x|whatsapp|linkedin|telegram|imprimir|email)$",
        r"^volver( arriba| al inicio)?$",
        r"^ir al contenido( principal)?$",
        r"^(inicio|menu|buscar)$",
        r"^(si|no)$",
        r"este sitio (web )?utiliza cookies",
        r"^los contenidos de argentina\.gob\.ar",
        r"creative commons",
    )
]


@dataclass
class FaqEntry:
    question: str
    answer: str
    url: str
    category: str = ""
    source: str = "local"


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def load_local_faqs(faq_dir: Path) -> list[FaqEntry]:
    """Lee FAQs en JSON (lista o {"faqs": [...]}) y documentos Markdown/TXT."""
    entries: list[FaqEntry] = []
    if not faq_dir.is_dir():
        logger.warning("No existe el directorio de FAQs: %s", faq_dir)
        return entries
    for path in sorted(faq_dir.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data.get("faqs", []) if isinstance(data, dict) else data
            for item in items:
                entries.append(
                    FaqEntry(
                        question=str(item.get("pregunta") or item.get("question") or ""),
                        answer=str(item.get("respuesta") or item.get("answer") or ""),
                        url=str(item.get("url") or item.get("fuente") or ""),
                        category=str(item.get("tramite") or item.get("categoria") or item.get("category") or ""),
                        source=path.name,
                    )
                )
        elif suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
            url_match = re.search(r"^(?:url|fuente)\s*:\s*(https://\S+)", text, flags=re.MULTILINE | re.IGNORECASE)
            entries.append(
                FaqEntry(
                    question=title_match.group(1) if title_match else path.stem.replace("_", " "),
                    answer=text,
                    url=url_match.group(1) if url_match else "",
                    source=path.name,
                )
            )
    logger.info("FAQs locales cargadas: %d", len(entries))
    return entries


class PoliteFetcher:
    """Cliente HTTP que respeta robots.txt, limita a dominios oficiales y espacia pedidos."""

    def __init__(self, delay_seconds: float = 1.0, timeout_seconds: float = 20.0) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept-Language": "es-AR,es;q=0.9"},
            timeout=timeout_seconds,
            follow_redirects=True,
        )
        self._robots: dict[str, RobotFileParser] = {}
        self._delay = delay_seconds
        self._last_request = 0.0

    def close(self) -> None:
        self._client.close()

    def _robots_for(self, base: str) -> RobotFileParser:
        if base not in self._robots:
            parser = RobotFileParser()
            try:
                response = self._client.get(f"{base}/robots.txt")
                if response.status_code in (401, 403):
                    parser.disallow_all = True
                parser.parse(response.text.splitlines() if response.status_code == 200 else [])
            except httpx.HTTPError:
                parser.parse([])
            self._robots[base] = parser
        return self._robots[base]

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.scheme != "https":
            return False
        if not any(host == domain or host.endswith(f".{domain}") for domain in ALLOWED_SCRAPE_DOMAINS):
            return False
        return self._robots_for(f"{parts.scheme}://{parts.netloc}").can_fetch(USER_AGENT, url)

    def get(self, url: str) -> str | None:
        if not self.allowed(url):
            logger.warning("URL omitida (fuera de dominio oficial o bloqueada por robots.txt): %s", url)
            return None
        wait = self._delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            response = self._client.get(url)
            self._last_request = time.monotonic()
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            logger.warning("No se pudo descargar %s: %s", url, exc)
            return None


def is_boilerplate(line: str) -> bool:
    folded = fold(line)
    return any(pattern.search(folded) for pattern in _BOILERPLATE)


_HEADINGS = {"h2", "h3", "h4", "h5", "h6"}
_BLOCKS = ["h2", "h3", "h4", "h5", "h6", "p", "li", "dt", "dd", "td", "th", "blockquote"]


def is_table_of_contents(element: Any) -> bool:
    """Ítems de índice ("1. ¿Qué es?" -> #01): enlaces internos que no aportan contenido."""
    text = element.get_text(" ", strip=True)
    anchors = element.find_all("a", href=lambda href: bool(href) and href.startswith("#"))
    anchor_chars = sum(len(a.get_text(" ", strip=True)) for a in anchors)
    return bool(text) and anchor_chars >= 0.8 * len(text)


def extract_sections(html: str, url: str) -> list[FaqEntry]:
    """Convierte una página oficial en secciones (título h2-h6 -> contenido)."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe", "svg", "button", "template"]):
        tag.decompose()
    for selector in (".breadcrumb", "[role=navigation]", ".sr-only", ".visually-hidden", ".social-share"):
        for element in soup.select(selector):
            element.decompose()

    h1 = soup.find("h1")
    if h1:
        page_title = clean_text(h1.get_text(" ", strip=True))
    elif soup.title:
        page_title = clean_text(soup.title.get_text(" ", strip=True))
    else:
        page_title = url
    root = soup.find("main") or soup.find("article") or soup.body or soup

    sections: list[FaqEntry] = []
    heading = page_title
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(FaqEntry(question=heading, answer=body, url=url, category=page_title, source="web"))
        buffer.clear()

    for element in root.find_all(_BLOCKS):
        if element.name in _HEADINGS:
            flush()
            heading = clean_text(element.get_text(" ", strip=True)) or heading
            continue
        if element.find_parent(["p", "li", "dd", "td", "th", "blockquote"]) is not None:
            continue  # el texto ya se incluyó con el elemento contenedor
        if is_table_of_contents(element):
            continue
        text = clean_text(element.get_text(" ", strip=True))
        if not text or is_boilerplate(text):
            continue
        buffer.append(f"- {text}" if element.name == "li" else text)
    flush()
    return sections


def read_urls(path: Path) -> list[str]:
    if not path.is_file():
        logger.warning("No existe el archivo de URLs: %s", path)
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def scrape(urls: list[str], delay_seconds: float) -> list[FaqEntry]:
    fetcher = PoliteFetcher(delay_seconds=delay_seconds)
    entries: list[FaqEntry] = []
    try:
        for url in urls:
            html = fetcher.get(url)
            if html is None:
                continue
            sections = extract_sections(html, url)
            logger.info("%s -> %d secciones", url, len(sections))
            entries.extend(sections)
    finally:
        fetcher.close()
    return entries


# ---------------------------------------------------------------------------
# Limpieza y chunking
# ---------------------------------------------------------------------------


def clean_entries(entries: list[FaqEntry]) -> list[FaqEntry]:
    cleaned: list[FaqEntry] = []
    seen: set[str] = set()
    for entry in entries:
        question = clean_text(entry.question)
        answer_lines = [line for line in clean_text(entry.answer).split("\n") if not is_boilerplate(line)]
        answer = "\n".join(answer_lines).strip()
        if len(answer) < MIN_ANSWER_CHARS:
            continue
        url = entry.url.strip() if entry.url.strip().startswith("https://") else ""
        fingerprint = stable_hash(fold(question), fold(answer))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        cleaned.append(FaqEntry(question, answer, url, clean_text(entry.category), entry.source))
    logger.info("Entradas limpias y deduplicadas: %d (de %d)", len(cleaned), len(entries))
    return cleaned


def build_chunks(entries: list[FaqEntry], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Cada chunk repite trámite y pregunta para que sea autocontenido al recuperarlo."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    documents: list[Document] = []
    for entry in entries:
        header = (f"Trámite: {entry.category}\n" if entry.category else "") + f"Pregunta: {entry.question}\n"
        pieces = splitter.split_text(entry.answer)
        entry_id = stable_hash(entry.url, entry.question, entry.answer)[:20]
        for index, piece in enumerate(pieces):
            documents.append(
                Document(
                    id=f"{entry_id}-{index}",
                    page_content=f"{header}Respuesta: {piece}",
                    metadata={
                        "title": (entry.category or entry.question)[:300],
                        "question": entry.question[:500],
                        "url": entry.url,
                        "category": entry.category[:200],
                        "source": entry.source,
                        "chunk": index,
                        "chunks_in_entry": len(pieces),
                    },
                )
            )
    logger.info("Chunks generados: %d", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Escritura en Chroma
# ---------------------------------------------------------------------------


def wait_for_chroma(settings: Settings, timeout_seconds: float = 180) -> Any:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            client = create_chroma_client(settings)
            client.heartbeat()
            return client
        except Exception as exc:  # noqa: BLE001
            if time.monotonic() > deadline:
                raise RuntimeError(f"Chroma no respondió en {settings.chroma_host}:{settings.chroma_port}") from exc
            logger.info("Esperando a Chroma...")
            time.sleep(3)


def embed_with_retry(embeddings: Any, texts: list[str], attempts: int = 6) -> list[list[float]]:
    for attempt in range(1, attempts + 1):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts:
                raise
            wait = min(30, 2**attempt)
            logger.warning("Fallo al generar embeddings (%s). Reintento %d en %ds", exc, attempt, wait)
            time.sleep(wait)
    raise RuntimeError("inalcanzable")


def read_active_pointer(client: Any, settings: Settings) -> dict[str, Any]:
    try:
        return dict(client.get_collection(name=settings.kb_meta_collection, embedding_function=None).metadata or {})
    except Exception:  # noqa: BLE001 - todavía no hay índice
        return {}


def active_index_is_usable(client: Any, settings: Settings) -> bool:
    pointer = read_active_pointer(client, settings)
    name = pointer.get(META_ACTIVE_COLLECTION)
    if not name:
        return False
    if pointer.get(META_EMBED_MODEL) != settings.embed_model_name:
        logger.warning(
            "El índice activo usa otro modelo de embeddings (%s != %s): se re-indexa",
            pointer.get(META_EMBED_MODEL),
            settings.embed_model_name,
        )
        return False
    try:
        return client.get_collection(name=name, embedding_function=None).count() > 0
    except Exception:  # noqa: BLE001
        return False


def prune_old_collections(client: Any, settings: Settings, keep: int, active: str) -> None:
    """Borra índices viejos. Siempre conserva al menos 2: las réplicas de la API pueden
    seguir usando el índice anterior hasta KB_REFRESH_SECONDS después del cambio."""
    prefix = f"{settings.kb_collection_prefix}_"
    names = sorted((c.name for c in client.list_collections() if c.name.startswith(prefix)), reverse=True)
    for name in names[max(keep, 2):]:
        if name != active:
            client.delete_collection(name=name)
            logger.info("Índice anterior eliminado: %s", name)


def write_index(client: Any, settings: Settings, documents: list[Document], batch_size: int, keep: int) -> str:
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%d%H%M%S")
    collection_name = f"{settings.kb_collection_prefix}_{version}"
    content_hash = stable_hash(*(doc.page_content for doc in documents))[:8]
    kb_version = f"{version}-{content_hash}"

    embeddings = build_embeddings(settings)
    collection = client.create_collection(
        name=collection_name,
        configuration=COLLECTION_CONFIGURATION,
        metadata={"embed_model": settings.embed_model_name, "created_at": now.isoformat()},
        embedding_function=None,
    )
    logger.info("Colección nueva: %s (embeddings: %s)", collection_name, settings.embed_model_name)

    try:
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            vectors = embed_with_retry(embeddings, [doc.page_content for doc in batch])
            collection.upsert(
                ids=[str(doc.id) for doc in batch],
                embeddings=vectors,
                documents=[doc.page_content for doc in batch],
                metadatas=[doc.metadata for doc in batch],
            )
            logger.info("Indexados %d/%d chunks", min(start + batch_size, len(documents)), len(documents))
        count = collection.count()
        if count != len(documents):
            raise RuntimeError(f"Se esperaban {len(documents)} chunks y la colección tiene {count}")
    except Exception:
        logger.exception("Falló la indexación: se descarta %s y el índice activo no cambia", collection_name)
        client.delete_collection(name=collection_name)
        raise

    # Cambio atómico del índice activo: la API lo detecta en <= KB_REFRESH_SECONDS.
    pointer = {
        "purpose": "kb-pointer",
        META_ACTIVE_COLLECTION: collection_name,
        META_KB_VERSION: kb_version,
        META_EMBED_MODEL: settings.embed_model_name,
        META_CHUNKS: count,
        META_UPDATED_AT: now.isoformat(),
    }
    meta = client.get_or_create_collection(
        name=settings.kb_meta_collection, metadata={"purpose": "kb-pointer"}, embedding_function=None
    )
    meta.modify(metadata=pointer)
    logger.info("Índice activo -> %s (kb_version=%s)", collection_name, kb_version)

    # La caché semántica apunta a respuestas del índice anterior: se descarta.
    try:
        client.delete_collection(name=settings.semantic_cache_collection)
        logger.info("Caché semántica invalidada")
    except Exception:  # noqa: BLE001 - no existía
        pass

    prune_old_collections(client, settings, keep=keep, active=collection_name)
    return kb_version


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indexa FAQs oficiales en ChromaDB")
    parser.add_argument("--faq-dir", type=Path, default=BASE_DIR / "data" / "faqs")
    parser.add_argument("--urls-file", type=Path, default=BASE_DIR / "data" / "sources.txt")
    parser.add_argument("--from-web", action="store_true", help="descargar también las URLs oficiales")
    parser.add_argument("--skip-local", action="store_true", help="no usar las FAQs locales")
    parser.add_argument("--if-empty", action="store_true", help="salir sin cambios si ya hay un índice activo")
    parser.add_argument("--dry-run", action="store_true", help="procesar y mostrar chunks sin escribir")
    parser.add_argument("--preview", type=int, default=3, help="chunks a mostrar en --dry-run")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--keep", type=int, default=2, help="índices a conservar, incluido el activo (mínimo 2)")
    parser.add_argument("--delay", type=float, default=1.0, help="segundos entre descargas")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args(argv)
    settings = get_settings()

    client = None
    if not args.dry_run:
        client = wait_for_chroma(settings)
        if args.if_empty and active_index_is_usable(client, settings):
            logger.info("Ya existe un índice activo y utilizable: no se re-indexa (--if-empty)")
            return 0

    entries: list[FaqEntry] = []
    if not args.skip_local:
        entries.extend(load_local_faqs(args.faq_dir))
    if args.from_web:
        entries.extend(scrape(read_urls(args.urls_file), args.delay))
    entries = clean_entries(entries)
    if not entries:
        logger.error("No hay contenido para indexar")
        return 1
    documents = build_chunks(entries, args.chunk_size, args.chunk_overlap)

    if args.dry_run:
        for doc in documents[: args.preview]:
            print("-" * 80)
            print(f"id={doc.id} url={doc.metadata['url']}")
            print(doc.page_content)
        print("-" * 80)
        print(f"Total: {len(entries)} entradas, {len(documents)} chunks (dry-run, sin escribir)")
        return 0

    kb_version = write_index(client, settings, documents, args.batch_size, args.keep)
    logger.info("Ingesta completa: %d chunks, kb_version=%s", len(documents), kb_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
