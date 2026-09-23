# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    # El vocabulario del contador de tokens queda dentro de la imagen: sin descargas en runtime
    # (necesario en servidores on-premise sin salida a internet).
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY app ./app
COPY static ./static
COPY data ./data
COPY main.py ingest.py gunicorn.conf.py ./

RUN useradd --system --uid 10001 --home-dir /app appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]
