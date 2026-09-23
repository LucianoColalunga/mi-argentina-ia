"""Gunicorn como gestor de procesos + workers Uvicorn (ASGI).

Cada contenedor de la API corre WEB_CONCURRENCY procesos; la escala horizontal
se logra sumando contenedores detrás del balanceador.
"""

import os
import shutil

bind = os.getenv("BIND", "0.0.0.0:8000")
workers = int(os.getenv("WEB_CONCURRENCY", "2"))
worker_class = "uvicorn_worker.UvicornWorker"
worker_tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
# El contenedor es de solo lectura: el socket de control de gunicorn va al tmpfs de /tmp.
control_socket = os.getenv("GUNICORN_CONTROL_SOCKET", "/tmp/gunicorn.ctl")
backlog = 2048
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5
# Reciclado periódico de workers para acotar fragmentación de memoria.
max_requests = 20_000
max_requests_jitter = 2_000
accesslog = None  # la API ya emite un log JSON por consulta
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()


def on_starting(server):
    """Limpia las métricas multi-proceso de ejecuciones anteriores."""
    directory = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if directory:
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)


def child_exit(server, worker):
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(worker.pid)
