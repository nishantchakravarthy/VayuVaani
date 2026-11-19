// VayuVaani service worker: lets the app install to a phone's home screen and open offline.
// The page and icons come from the cache first; forecast data always tries the network first,
// so you see the newest forecast when online and the last one you saw when offline.
const CACHE = "vayuvaani-v1";
const SHELL = ["./", "./index.html", "./manifest.webmanifest", "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;   // maps, fonts, POSTs: straight to the network
  const isData = url.pathname.includes("/data/") || url.pathname.includes("/api/");
  if (url.pathname.endsWith("/api/tts") || url.pathname.endsWith("/api/ping")) return;
  if (isData || e.request.mode === "navigate") {
    e.respondWith(fetch(e.request).then(r => {
      if (r.ok) { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); }
      return r;
    }).catch(() => caches.match(e.request).then(r => r || caches.match("./index.html"))));
  } else {
    e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
  }
});
