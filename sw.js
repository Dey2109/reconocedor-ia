// =====================================================================
//  sw.js  -  Service Worker: hace que la página funcione SIN INTERNET
// =====================================================================
// Un service worker es un programa que el navegador ejecuta aparte de la página. Se coloca
// "en medio" de cada petición (imagen, script, modelo...) y decide si responder desde la
// red o desde una copia guardada (caché). Solo funciona en localhost o HTTPS.

const VERSION = "v1";                          // súbelo (v2, v3...) cuando cambies archivos del proyecto
const CACHE_APP = `inframen-app-${VERSION}`;   // archivos propios del proyecto
const CACHE_EXT = "inframen-externo-v1";       // librería, modelos de IA y motor (MISMO nombre que en app.js)
const ARCHIVOS = ["./", "index.html", "style.css", "app.js", "logo.png", "alacran.png",
                  "manifest.webmanifest", "icon-192.png", "icon-512.png", "fonts/OXYGENE1.TTF"];
const HOSTS_EXTERNOS = ["cdn.jsdelivr.net", "storage.googleapis.com", "fonts.googleapis.com", "fonts.gstatic.com"];

// 1) INSTALAR: guarda los archivos propios. allSettled = si uno falta (ej. la fuente), los demás se guardan igual.
self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const cache = await caches.open(CACHE_APP);
    await Promise.allSettled(ARCHIVOS.map(a => cache.add(a)));
    await self.skipWaiting();                  // la versión nueva se activa sin esperar
  })());
});

// 2) ACTIVAR: borra cachés de versiones viejas y toma el control de la página abierta.
self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith("inframen-app-") && k !== CACHE_APP) await caches.delete(k);
    await self.clients.claim();
  })());
});

// 3) CADA PETICIÓN: según de dónde venga se usa una estrategia distinta.
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin === location.origin) e.respondWith(redPrimero(req));
  else if (HOSTS_EXTERNOS.includes(url.hostname)) e.respondWith(cachePrimero(req));
});

// Archivos propios: RED PRIMERO. Así, mientras programas, ves tus cambios al recargar;
// si no hay internet, se usa la última copia guardada.
async function redPrimero(req) {
  try {
    const r = await fetch(req);
    if (r.ok) (await caches.open(CACHE_APP)).put(req, r.clone());
    return r;
  } catch {
    return (await caches.match(req, { ignoreSearch: true })) ||
           (req.mode === "navigate" ? await caches.match("index.html") : Response.error());
  }
}

// Modelos, librerías y motor: CACHÉ PRIMERO. Pesan mucho y no cambian, así que si ya están guardados no se vuelven a descargar.
async function cachePrimero(req) {
  const cache = await caches.open(CACHE_EXT);
  const guardado = await cache.match(req);
  if (guardado) return guardado;
  const r = await fetch(req);
  if (r.ok || r.type === "opaque") cache.put(req, r.clone());   // "opaque" = respuesta de otro sitio sin permisos de lectura
  return r;
}