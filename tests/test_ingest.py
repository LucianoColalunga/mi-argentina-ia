from pathlib import Path

from ingest import build_chunks, clean_entries, extract_sections, load_local_faqs

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "faqs"

HTML = """
<html><head><title>Preguntas frecuentes</title></head>
<body>
  <header><nav>Inicio | Menú</nav></header>
  <main>
    <h1>Preguntas frecuentes sobre Mi Argentina</h1>
    <ol>
      <li><a href="#01">¿Cómo creo mi cuenta?</a></li>
      <li><a href="#02">¿Puedo borrar mi cuenta?</a></li>
    </ol>
    <h2>¿Cómo creo mi cuenta?</h2>
    <p>Ingresá a Mi Argentina y seguí los pasos de registro con tu CUIL y un correo electrónico.</p>
    <ul><li>Necesitás una contraseña segura <p>de al menos 8 caracteres</p></li></ul>
    <h5 id="02">¿Puedo borrar mi cuenta?</h5>
    <p>Sí, desde la sección Configurar mi cuenta, eligiendo el motivo y confirmando.</p>
    <p>¿Te sirvió esta información?</p>
    <script>alert('x')</script>
  </main>
  <footer>Los contenidos de Argentina.gob.ar están licenciados bajo Creative Commons</footer>
</body></html>
"""


def test_extract_sections_from_official_html() -> None:
    sections = extract_sections(HTML, "https://www.argentina.gob.ar/miargentina/preguntasfrecuentes")
    assert [s.question for s in sections] == ["¿Cómo creo mi cuenta?", "¿Puedo borrar mi cuenta?"]
    first = sections[0].answer
    assert "CUIL" in first and "- Necesitás una contraseña segura de al menos 8 caracteres" in first
    assert first.count("de al menos 8 caracteres") == 1  # sin duplicar texto anidado
    assert all("alert" not in s.answer and "sirvió" not in s.answer for s in sections)
    assert sections[0].category == "Preguntas frecuentes sobre Mi Argentina"


def test_demo_dataset_loads_and_chunks() -> None:
    entries = clean_entries(load_local_faqs(DATA_DIR))
    assert len(entries) >= 20
    assert all(e.url.startswith("https://www.argentina.gob.ar/") for e in entries)
    documents = build_chunks(entries, chunk_size=1000, chunk_overlap=150)
    ids = [doc.id for doc in documents]
    assert len(ids) == len(set(ids))
    assert all(doc.page_content.startswith(("Trámite:", "Pregunta:")) for doc in documents)


def test_clean_entries_deduplicates() -> None:
    entries = extract_sections(HTML, "https://www.argentina.gob.ar/x") * 2
    assert len(clean_entries(entries)) == 2
