// Cliente de demostración de /v1/chat. Todo el texto del modelo se inserta con
// textContent (nunca innerHTML) y solo se enlazan URLs https.
(() => {
  const chat = document.getElementById("chat");
  const form = document.getElementById("form");
  const input = document.getElementById("q");
  const send = document.getElementById("send");
  const status = document.getElementById("status");

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function addMessage(role, text, extraClass) {
    const article = el("article", `msg ${role}${extraClass ? " " + extraClass : ""}`);
    article.appendChild(el("p", "", text));
    chat.appendChild(article);
    article.scrollIntoView({ behavior: "smooth", block: "end" });
    return article;
  }

  function safeHttpsUrl(value) {
    try {
      const url = new URL(value);
      return url.protocol === "https:" ? url.href : null;
    } catch {
      return null;
    }
  }

  function renderAnswer(data, cacheHeader) {
    const article = addMessage("bot", data.answer);

    const links = (data.sources || [])
      .map((s) => ({ title: s.title, href: safeHttpsUrl(s.url) }))
      .filter((s) => s.href);
    if (links.length) {
      const list = el("ul", "sources");
      for (const link of links) {
        const item = el("li");
        const anchor = el("a", "", link.title || link.href);
        anchor.href = link.href;
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
        item.appendChild(anchor);
        list.appendChild(item);
      }
      article.appendChild(list);
    }

    const meta = el("div", "meta");
    let label;
    let cls;
    if (data.cached) {
      label = `caché ${data.cache_layer}`;
      cls = "hit";
    } else if (data.blocked) {
      label = `bloqueado: ${data.block_reason}`;
      cls = "blocked";
    } else {
      label = "generado por el modelo local";
      cls = "";
    }
    meta.appendChild(el("span", `tag ${cls}`, label));
    const tokens = data.usage && data.usage.output_tokens
      ? ` · ${data.usage.input_tokens} tokens entrada / ${data.usage.output_tokens} salida`
      : "";
    meta.appendChild(document.createTextNode(` · ${data.latency_ms} ms${tokens} · X-Cache: ${cacheHeader || "-"}`));
    article.appendChild(meta);
  }

  async function ask(question) {
    addMessage("user", question);
    send.disabled = true;
    const pending = addMessage("bot", "Buscando en la información oficial…");
    try {
      const response = await fetch("/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await response.json().catch(() => ({}));
      pending.remove();
      if (!response.ok) {
        const detail = typeof data.detail === "string" ? data.detail : "No se pudo procesar la consulta.";
        addMessage("bot", `${detail} (HTTP ${response.status})`, "error");
        return;
      }
      renderAnswer(data, response.headers.get("X-Cache"));
    } catch {
      pending.remove();
      addMessage("bot", "No hay conexión con el asistente.", "error");
    } finally {
      send.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    ask(question);
  });

  document.querySelectorAll(".suggestions button").forEach((button) => {
    button.addEventListener("click", () => ask(button.dataset.q));
  });

  fetch("/health/ready")
    .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
    .then(({ ok, d }) => {
      const failing = Object.entries(d.checks || {}).filter(([, v]) => !v).map(([k]) => k);
      status.textContent = ok ? (failing.length ? `listo (sin ${failing.join(", ")})` : "listo") : "no disponible";
      status.className = `pill ${ok ? "ok" : "bad"}`;
    })
    .catch(() => {
      status.textContent = "sin conexión";
      status.className = "pill bad";
    });
})();
