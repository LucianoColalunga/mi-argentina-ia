"""Limpieza y normalización de texto compartida por la ingesta y la API."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_ZERO_WIDTH = re.compile("[­​-‏‪-‮⁠-⁤﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_WS = re.compile(r"[ \t\r\f\v ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")

# Saludos y fórmulas de cortesía que no cambian el sentido de la consulta.
# Quitarlos de la clave de caché sube mucho la tasa de aciertos.
_FILLER = re.compile(
    r"\b("
    r"hola+|buenas( tardes| noches)?|buenos dias|buen dia|"
    r"por ?favor|porfa|much[ao]s gracias|gracias|disculp[ae]|che"
    r")\b"
)


def clean_text(text: str) -> str:
    """NFKC, sin caracteres invisibles ni de control, espacios colapsados por línea."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub(" ", text)
    lines = [_INLINE_WS.sub(" ", line).strip() for line in text.split("\n")]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def fold(text: str) -> str:
    """Minúsculas y sin tildes: base común para comparar texto."""
    return strip_accents(clean_text(text).lower())


def normalize_for_cache(text: str) -> str:
    """Forma canónica de una consulta para usarla como clave de caché.

    "¡Hola! ¿Cómo saco el DNI digital, por favor?" -> "como saco el dni digital"
    """
    folded = _NON_ALNUM.sub(" ", fold(text))
    folded = _FILLER.sub(" ", folded)
    return _SPACES.sub(" ", folded).strip()


def stable_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()
