"""Consulta puntual para pruebas manuales: python ask.py "pregunta" """

import os
import sys
import uuid

import httpx

headers = {"X-Client-Id": uuid.uuid4().hex}
if os.getenv("E2E_EDGE_SECRET"):
    headers["X-Edge-Auth"] = os.environ["E2E_EDGE_SECRET"]
r = httpx.post("http://nginx/v1/chat", json={"question": sys.argv[1]}, headers=headers, timeout=120)
b = r.json()
print(f"HTTP {r.status_code} X-Cache={r.headers.get('x-cache')} kb_version={b.get('kb_version')} blocked={b.get('blocked')} reason={b.get('block_reason')}")
