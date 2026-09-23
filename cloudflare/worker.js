/**
 * Cloudflare Worker: caché en el edge para POST /v1/chat.
 *
 * Las CDN no cachean POST. Este Worker calcula una clave a partir de la pregunta
 * normalizada (misma normalización que la API), busca la respuesta en la caché
 * del datacenter de Cloudflare más cercano al usuario y solo va al origen en un
 * miss. Guarda únicamente respuestas 200 que el origen marcó como `public` con
 * `s-maxage` (la API marca `private, no-store` las consultas con datos personales).
 *
 * Variables:
 *   CACHE_NAMESPACE     (var)    cambiarla invalida toda la caché del edge (p. ej. tras re-indexar)
 *   EDGE_SHARED_SECRET  (secret) se envía al origen en X-Edge-Auth
 */

const CHAT_PATH = "/v1/chat";
const MAX_BODY_BYTES = 8 * 1024;

// Saludos y cortesías que no cambian el sentido (igual que app/text.py).
const FILLER =
  /\b(hola+|buenas( tardes| noches)?|buenos dias|buen dia|por ?favor|porfa|much[ao]s gracias|gracias|disculp[ae]|che)\b/g;

export function normalizeQuestion(text) {
  return text
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9 ]+/g, " ")
    .replace(FILLER, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function sha256Hex(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function jsonError(status, detail) {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

function withEdgeAuth(headers, env) {
  const out = new Headers(headers);
  if (env.EDGE_SHARED_SECRET) out.set("X-Edge-Auth", env.EDGE_SHARED_SECRET);
  return out;
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method !== "POST" || url.pathname !== CHAT_PATH) {
      return fetch(new Request(request, { headers: withEdgeAuth(request.headers, env) }));
    }

    const body = await request.text();
    if (body.length > MAX_BODY_BYTES) return jsonError(413, "La consulta es demasiado larga.");

    let question = null;
    try {
      question = JSON.parse(body).question;
    } catch {
      question = null;
    }

    const originRequest = new Request(request.url, {
      method: "POST",
      headers: withEdgeAuth(request.headers, env),
      body,
    });
    if (typeof question !== "string" || !question.trim()) {
      return fetch(originRequest); // el origen devuelve el error de validación
    }

    const namespace = env.CACHE_NAMESPACE || "v1";
    const key = await sha256Hex(`${namespace}|${normalizeQuestion(question)}`);
    const cacheKey = new Request(`${url.origin}/__edge-cache/chat/${key}`, { method: "GET" });
    const cache = caches.default;

    const hit = await cache.match(cacheKey);
    if (hit) {
      const response = new Response(hit.body, hit);
      response.headers.set("X-Edge-Cache", "HIT");
      return response;
    }

    const originResponse = await fetch(originRequest);
    const cacheControl = originResponse.headers.get("Cache-Control") || "";
    const sMaxAge = /(?:^|,)\s*s-maxage=(\d+)/.exec(cacheControl);
    const cacheable = originResponse.status === 200 && /\bpublic\b/.test(cacheControl) && sMaxAge;

    const response = new Response(originResponse.body, originResponse);
    response.headers.set("X-Edge-Cache", cacheable ? "MISS" : "BYPASS");

    if (cacheable) {
      const stored = response.clone();
      const headers = new Headers(stored.headers);
      headers.set("Cache-Control", `public, max-age=${sMaxAge[1]}`);
      headers.delete("Set-Cookie");
      ctx.waitUntil(cache.put(cacheKey, new Response(stored.body, { status: 200, headers })));
    }
    return response;
  },
};
