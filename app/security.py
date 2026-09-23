"""Capas de seguridad del asistente.

1. Sanitización de la entrada (caracteres invisibles, control, longitud).
2. Detección heurística de prompt injection / jailbreak (ES + EN, ofuscaciones).
3. Enmascarado de datos personales antes de que lleguen al modelo, la caché o los logs.
4. Validación de la salida del modelo (fuga del prompt, enlaces no oficiales, HTML).

Las heurísticas no son infalibles: son la primera línea de una defensa en
profundidad que se completa con el prompt delimitado, el umbral de relevancia
del RAG y el guard de salida.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from app.prompts import NO_INFO_MESSAGE, REFUSAL_MESSAGE
from app.text import clean_text, fold, normalize_for_cache

# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------


def sanitize_user_input(text: str) -> str:
    return clean_text(text)


def escape_for_prompt(text: str) -> str:
    """Impide que el usuario abra o cierre las etiquetas que delimitan el prompt."""
    return text.replace("<", "‹").replace(">", "›")


_SPECIAL_TOKENS = re.compile(r"<\|[^|>]{1,40}\|>|\[/?INST\]|<</?SYS>>", re.IGNORECASE)
_PROMPT_TAGS = re.compile(
    r"</?\s*(contexto_oficial|pregunta_usuario|system|assistant|user)\s*>", re.IGNORECASE
)


def sanitize_context(text: str) -> str:
    """Limpia documentos recuperados: aunque vengan de la ingesta, no se confía en ellos."""
    text = _SPECIAL_TOKENS.sub(" ", text)
    text = _PROMPT_TAGS.sub(" ", text)
    return clean_text(text)


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------

_OVERRIDE_VERBS = (
    r"(ignora|ignore|ignoren|ignorar|olvida|olvidate|olviden|olvidar|omiti|omite|omitir|"
    r"descarta|descartar|saltea|saltear|anula|anular|desactiva|desactivar|deja de lado|"
    r"dejar de lado|pasa por alto|pasar por alto|no sigas|no respetes|no cumplas|"
    r"desobedece|desobedecer|saltate|salta)"
)
_ASSISTANT_QUALIFIER = (
    r"(tus|tu|sus|todas|anteriores|previas|de arriba|del sistema|iniciales|originales|"
    r"que te (dieron|dio|pusieron|programaron)|que (tenes|tienes))"
)
_RULE_NOUNS = (
    r"(instrucciones|instruccion|reglas|regla|indicaciones|directivas|restricciones|"
    r"ordenes|politicas|limitaciones|filtros|prompt|programacion|configuracion|consignas)"
)
_REVEAL_VERBS = (
    r"(revela|revelame|revelar|mostra|mostrame|muestra|muestrame|mostrar|deci|decime|dime|"
    r"decir|repeti|repetime|repite|repetir|imprimi|imprime|imprimir|escribi|escribime|"
    r"escribe|copia|copiame|traduci|traduce|resumi|resume|lista|enumera|cuales son|"
    r"que dice|que dicen)"
)

# Se evalúan sobre el texto "plegado": minúsculas y sin tildes.
_FOLDED_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "override_rules_es",
        re.compile(
            rf"\b{_OVERRIDE_VERBS}\b[^.\n]{{0,40}}"
            rf"(\b{_ASSISTANT_QUALIFIER}\b[^.\n]{{0,25}}\b{_RULE_NOUNS}\b"
            rf"|\b{_RULE_NOUNS}\b[^.\n]{{0,15}}\b{_ASSISTANT_QUALIFIER}\b)"
        ),
    ),
    (
        "forget_everything_es",
        re.compile(r"\b(olvida|olvidate|ignora|ignore|borra)\b[^.\n]{0,10}\btodo (lo )?(anterior|que te (dije|dijeron))"),
    ),
    (
        "role_change_es",
        re.compile(
            r"\b(a partir de (ahora|este momento)|desde ahora|de ahora en mas)\b[^.\n]{0,40}"
            r"\b(sos|eres|seras|vas a ser|actua|actuas|actuaras|responde|responderas|respondes|"
            r"hablas|hablaras|tu nombre es|te llamas|ya no)\b"
        ),
    ),
    (
        "roleplay_es",
        re.compile(
            r"(^|[.!?,;:¡¿]\s*|\b(quiero|necesito|te pido) que\s+)"
            r"(actua|actues|comportate|te comportes|finge|fingi|finjas|simula|simules|"
            r"hace de cuenta|hagas de cuenta|pretende|interpreta|interpretes)\b"
        ),
    ),
    (
        "jailbreak_keywords",
        re.compile(
            r"\b(jailbreak|jail break|do anything now|developer mode|dan mode|roleplay|role play|"
            r"juego de rol|tu nuevo (rol|nombre|personaje)|cambia(r)? (de|tu) (rol|personalidad)|"
            r"modo (desarrollador|developer|dios|sin restricciones|sin filtros|sin censura|libre|"
            r"debug|depuracion|administrador|admin|root|jailbreak))\b"
        ),
    ),
    (
        "reveal_prompt_es",
        re.compile(rf"\b{_REVEAL_VERBS}\b[^.\n]{{0,30}}\b(tus|tu)\s+{_RULE_NOUNS}\b"),
    ),
    (
        "system_prompt_reference",
        re.compile(
            r"\b(system prompt|prompt (de|del) sistema|mensaje (de|del) sistema|"
            r"instrucciones (de|del) sistema|instrucciones (iniciales|originales|ocultas|internas)|"
            r"prompt (inicial|original|oculto|interno))\b"
        ),
    ),
    (
        "override_rules_en",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass|skip)\b[^.\n]{0,30}"
            r"\b(previous|prior|above|earlier|all|your|the|any|system)\b[^.\n]{0,20}"
            r"\b(instructions?|rules?|prompts?|guidelines|directives|constraints)\b"
        ),
    ),
    (
        "role_change_en",
        re.compile(
            r"\b(you are now|act as|pretend (to be|you are)|roleplay as|from now on|"
            r"new instructions|reveal your|show me your (prompt|instructions|rules)|"
            r"print your (prompt|instructions)|what are your (instructions|rules))\b"
        ),
    ),
    (
        "chat_template_tokens",
        re.compile(
            r"<\|[a-z_ ]{2,30}\|>|\[/?inst\]|<</?sys>>"
            r"|<\s*/?\s*(system|assistant|contexto_oficial|pregunta_usuario|instrucciones)\s*>"
            r"|(^|\n)\s*(#{1,6}\s*)?(system|assistant|sistema|asistente|instrucciones)\s*:"
        ),
    ),
]

# Payloads codificados (base64 largo) usados para esconder instrucciones.
_ENCODED_PAYLOAD = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")

# Frases de ataque buscadas en el texto "aplastado" (sin espacios ni símbolos) para
# detectar ofuscaciones como "i g n o r a   t u s   r e g l a s" o "1gn0r4".
_SQUASHED_KEYWORDS = (
    "ignoratusinstrucciones",
    "ignoratodaslasinstrucciones",
    "ignoralasinstruccionesanteriores",
    "ignoratusreglas",
    "olvidatusinstrucciones",
    "olvidatusreglas",
    "ignorepreviousinstructions",
    "ignoreallpreviousinstructions",
    "ignoreallinstructions",
    "disregardpreviousinstructions",
    "systemprompt",
    "jailbreak",
    "developermode",
    "mododesarrollador",
    "modosinrestricciones",
    "doanythingnow",
)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_NON_LETTERS = re.compile(r"[^a-z]")


def detect_prompt_injection(text: str) -> str | None:
    """Devuelve el nombre de la regla que detectó el ataque, o None si parece legítimo."""
    if _ENCODED_PAYLOAD.search(text):
        return "encoded_payload"
    folded = fold(text)
    for name, pattern in _FOLDED_RULES:
        if pattern.search(folded):
            return name
    squashed = _NON_LETTERS.sub("", folded.translate(_LEET))
    for keyword in _SQUASHED_KEYWORDS:
        if keyword in squashed:
            return "obfuscated_injection"
    return None


# ---------------------------------------------------------------------------
# Datos personales (PII)
# ---------------------------------------------------------------------------

# El orden importa: primero los patrones más largos/específicos.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("CBU", re.compile(r"(?<!\d)\d{22}(?!\d)")),
    ("TELEFONO", re.compile(r"\+54[ -]?9?[ -]?(?:\d[ -]?){9,10}\d(?!\d)")),
    ("TARJETA", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")),
    ("CUIL", re.compile(r"(?<!\d)(?:20|23|24|25|26|27|30|33|34)[-. ]?\d{8}[-. ]?\d(?!\d)")),
    ("TELEFONO", re.compile(r"(?<!\d)(?:11|15)[ -]\d{4}[ -]\d{4}(?!\d)")),
    ("DNI", re.compile(r"(?<!\d)\d{1,2}\.?\d{3}\.?\d{3}(?!\d)")),
    ("PASAPORTE", re.compile(r"\b[A-Z]{3}\d{6}\b", re.IGNORECASE)),
]


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Reemplaza datos personales por marcadores ([DNI], [EMAIL], ...)."""
    found: list[str] = []
    for label, pattern in _PII_PATTERNS:
        text, count = pattern.subn(f"[{label}]", text)
        if count and label not in found:
            found.append(label)
    return text, found


# ---------------------------------------------------------------------------
# Salida del modelo
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]{0,200}>")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"'()\[\]]+", re.IGNORECASE)
# Solo el system prompt es confidencial. Que el modelo repita la pregunta del usuario o el
# contexto (bloques del mensaje del usuario) no es una fuga: se limpia en vez de bloquear.
_LEAK_MARKERS = (
    "reglas obligatorias",
    "codigo interno",
    "responde unicamente basandote en el contexto oficial",
)
_ECHOED_BLOCK = re.compile(
    r"<\s*(pregunta_usuario|contexto_oficial)\s*>.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL
)
_STRAY_PROMPT_TAG = re.compile(r"</?\s*(pregunta_usuario|contexto_oficial)\s*>", re.IGNORECASE)
_ECHOED_INSTRUCTION = re.compile(r"^\s*respond[oeé]\s+la\s+consulta\s+del\s+ciudadano\b.*$", re.IGNORECASE | re.MULTILINE)
_REFUSAL_KEY = normalize_for_cache(REFUSAL_MESSAGE)
_NO_INFO_KEY = "no encontre informacion oficial"
REMOVED_LINK = "[enlace no oficial removido]"


@dataclass(frozen=True)
class GuardResult:
    text: str
    violation: str | None = None


def _is_allowed_url(url: str, allowed_domains: list[str]) -> bool:
    candidate = url if "://" in url else f"https://{url}"
    host = (urlsplit(candidate).hostname or "").lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def _filter_url(url: str, allowed_domains: list[str]) -> str:
    trailing = ""
    while url and url[-1] in ".,;:!?":
        trailing = url[-1] + trailing
        url = url[:-1]
    return (url if _is_allowed_url(url, allowed_domains) else REMOVED_LINK) + trailing


def guard_output(
    text: str, canary: str, allowed_domains: list[str], max_chars: int = 2500
) -> GuardResult:
    """Valida la respuesta del modelo antes de devolverla o cachearla."""
    if canary and canary in text:
        return GuardResult(REFUSAL_MESSAGE, "canary_leak")
    text = _ECHOED_BLOCK.sub("", text)
    text = _STRAY_PROMPT_TAG.sub("", text)
    text = _ECHOED_INSTRUCTION.sub("", text)
    text = clean_text(text)
    if not text:
        return GuardResult(NO_INFO_MESSAGE, "empty_output")
    folded = fold(text)
    if any(marker in folded for marker in _LEAK_MARKERS):
        return GuardResult(REFUSAL_MESSAGE, "prompt_leak")

    # Si el modelo se negó o no encontró información, se devuelve el texto exacto.
    normalized = normalize_for_cache(text)
    if _REFUSAL_KEY in normalized:
        return GuardResult(REFUSAL_MESSAGE)
    # "No encontré información" seguido de una derivación con fuente oficial es una respuesta
    # útil (p. ej. "consultá el valor vigente en RENAPER"): solo se reemplaza si no cita fuente.
    if _NO_INFO_KEY in normalized and "fuente" not in normalized:
        return GuardResult(NO_INFO_MESSAGE)

    text = _HTML_TAG.sub("", text)
    text = _URL.sub(lambda match: _filter_url(match.group(0), allowed_domains), text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return GuardResult(text.strip())
