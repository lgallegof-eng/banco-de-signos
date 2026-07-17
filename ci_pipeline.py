# ci_pipeline.py
# Banco de Signos — ingesta robusta (RSS + autodiscovery + sitemaps + fallback) y objetivo 40–50 señales/día.
import os, time, math, datetime as dt, re
from pathlib import Path
from email.utils import parsedate_to_datetime
from collections import Counter, deque

import feedparser
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------ Volumen/ventanas/umbrales ------------
TARGET_MIN = 40         # objetivo mínimo de piezas publicadas
TARGET_MAX = 50         # objetivo máximo de piezas publicadas

HISTORY_DAYS = 21       # histórico p/ novedad
MAX_AGE_DAYS = 14       # analizamos últimos X días (más cobertura)
RECENCY_TAU_HOURS = 144 # semivida recencia (penaliza menos el tiempo)
SIM_THRESHOLD = 0.66    # similitud mínima para clúster (más laxo)
CLUSTER_MIN = 2         # tamaño mínimo de micro‑clúster
CLUSTER_MAX = 12        # tamaño máximo de micro‑clúster
MAX_PER_SOURCE = 8      # límite global por fuente (clústeres + piezas)

# ------------ Publicación/paths ------------
PUBLISH_DIR = Path(os.getenv("PUBLISH_DIR", "public")).resolve()
HIST_DIR = Path(os.getenv("HIST_DIR", "public/data")).resolve()
PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

HIST_FILE = HIST_DIR / "history.npz"
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# ------------ Fuentes por dominio ------------
# Puedes añadir o editar sitemaps/RSS por marca. Se intentará: RSS → autodiscovery → sitemaps → fallback HTML (opcional).
SOURCES = [
    {"name":"El Periódico","home":"https://www.elperiodico.com/es/","rss":["https://www.elperiodico.com/es/rss/rss_portada.xml"],"sitemaps":["https://www.elperiodico.com/es/sitemap.xml"]},
    {"name":"La Voz de Galicia","home":"https://www.lavozdegalicia.es/","rss":["https://www.lavozdegalicia.es/rss/index.xml"],"sitemaps":["https://www.lavozdegalicia.es/sitemap.xml"]},
    {"name":"SModa","home":"https://smoda.elpais.com/","rss":["https://smoda.elpais.com/feed/"],"sitemaps":["https://smoda.elpais.com/sitemap.xml"]},
    {"name":"Código Nuevo","home":"https://www.codigonuevo.com/","rss":["https://www.codigonuevo.com/feed"],"sitemaps":["https://www.codigonuevo.com/sitemap.xml"]},
    {"name":"Reason Why","home":"https://www.reasonwhy.es/","rss":["https://www.reasonwhy.es/rss","https://www.reasonwhy.es/rss.xml"],"sitemaps":["https://www.reasonwhy.es/sitemap.xml"]},
    {"name":"Control Publicidad","home":"https://controlpublicidad.com/","rss":["https://controlpublicidad.com/feed/"],"sitemaps":["https://controlpublicidad.com/sitemap.xml"]},
    {"name":"El Confidencial","home":"https://www.elconfidencial.com/","rss":["https://www.elconfidencial.com/rss/ultimas_noticias/"],"sitemaps":["https://www.elconfidencial.com/sitemap.xml"]},

    {"name":"The Guardian","home":"https://www.theguardian.com/international","rss":[
        "https://www.theguardian.com/world/rss",
        "https://www.theguardian.com/technology/rss",
        "https://www.theguardian.com/business/rss",
        "https://www.theguardian.com/science/rss"
    ],"sitemaps":["https://www.theguardian.com/sitemaps/news.xml"]},
    {"name":"The Atlantic","home":"https://www.theatlantic.com/","rss":["https://www.theatlantic.com/feed/all/"],"sitemaps":["https://www.theatlantic.com/sitemap.xml"]},
    {"name":"The New York Times","home":"https://www.nytimes.com/","rss":[
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"
    ],"sitemaps":["https://www.nytimes.com/sitemaps/news.xml"]},
    {"name":"Fast Company","home":"https://www.fastcompany.com/","rss":["https://www.fastcompany.com/rss"],"sitemaps":["https://www.fastcompany.com/sitemap.xml"]},
    {"name":"Highsnobiety","home":"https://www.highsnobiety.com/","rss":["https://www.highsnobiety.com/feed/"],"sitemaps":["https://www.highsnobiety.com/sitemap.xml"]},
    {"name":"Financial Times","home":"https://www.ft.com/","rss":["https://www.ft.com/rss/home"],"sitemaps":["https://www.ft.com/sitemap_index.xml"]},
    {"name":"Business Insider","home":"https://www.businessinsider.com/","rss":["https://www.businessinsider.com/rss"],"sitemaps":["https://www.businessinsider.com/sitemap.xml"]},
    {"name":"Business Insider ES","home":"https://www.businessinsider.es/","rss":["https://www.businessinsider.es/rss"],"sitemaps":["https://www.businessinsider.es/sitemap.xml"]},
    {"name":"Bloomberg","home":"https://www.bloomberg.com/","rss":[
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://feeds.bloomberg.com/technology/news.rss"
    ],"sitemaps":["https://www.bloomberg.com/sitemaps/sitemap_news.xml"]},

    # Extra para volumen (opcional; puedes desactivar si no quieres)
    {"name":"Reuters","home":"https://www.reuters.com/","rss":["https://www.reuters.com/world/rss"],"sitemaps":["https://www.reuters.com/sitemap_index.xml"]},
    {"name":"The Verge","home":"https://www.theverge.com/","rss":["https://www.theverge.com/rss/index.xml"],"sitemaps":["https://www.theverge.com/sitemap.xml"]},
    {"name":"Wired","home":"https://www.wired.com/","rss":["https://www.wired.com/feed/rss"],"sitemaps":["https://www.wired.com/sitemap-index.xml"]},
]

# ------------ Marcos/opinión (enriquecido pero ligero) ------------
FRAMES = {
    "economía": ["precio","precios","inflación","coste","costes","empleo","desempleo","recuperación","salario","pib","mercado","inversión",
                 "profits","growth","inflation","cost","jobs","recession","recovery","gdp","market","investment","funding","valuation","earnings"],
    "salud": ["salud","sanitario","sanitaria","epidemia","pandemia","hospital","oms","mental","bienestar","contagio","vacuna",
              "vaccine","health","public health","wellbeing","mental health"],
    "sostenibilidad": ["clima","climático","emisiones","co2","energía","renovable","circular","sostenible","biodiversidad",
                       "green","sustainable","renewable","emissions","net zero","climate","carbon"],
    "privacidad/tech": ["privacidad","datos","algoritmo","ia","inteligencia artificial","modelo","plataforma","red social","app",
                        "tracking","cookies","privacy","data","algorithm","ai","platform","surveillance","biometrics"],
    "justicia_social": ["género","igualdad","diversidad","inclusión","derechos","discriminación","racismo","lgtbi","feminismo",
                        "equity","diversity","inclusion","rights","discrimination","racism","gender"],
    "geopolítica": ["guerra","conflicto","frontera","sanciones","otan","ue","china","rusia","eeuu","diplomacia",
                    "war","conflict","border","sanctions","nato","eu","geopolitics","us"],
    "consumo/estilo": ["consumo","tendencia","moda","lifestyle","estilo","consumidor","compra","retail","cultura","estética",
                       "trend","style","fashion","consumer","shopping","culture"],
}
OPINION_HINTS = ["opinión","opinion","análisis","analysis","editorial","tribuna","columna","column","op-ed","view","viewpoint","perspective"]

STOPWORDS = set("""
a al algo algunas algunos ante antes así aún aunque cada como con contra cual cuales cuando
de del desde donde dos el la los las en entre era eran es esa esas ese eso esos esta estaban
estamos estar esta esta estas este esto estos estuvo ha había habían haber hacia hace hacen
hacer hacen haciendo han hasta hay he hemos hizo más mejor menos mi mis muy nada ni no nos
nuestra nuestras nuestro nuestros nunca o otras otros para pero poco por porque qué que quien
se será ser si sido sin sobre son su sus tal también tampoco tan tanto te tener tiene tienen
toda todas todo todos tras tú un una unas uno unos ya the a an and or of for to from in on by
is are was were be been being have has had do does did with as at into over after before about
between more most other some such no nor not only own same so than too very can will just should
""".split())
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")

# ------------ Ingesta robusta (RSS → autodiscovery → sitemaps → fallback HTML) ------------
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (compatible; BancoDeSignos/1.0; +https://github.com/lgallegof-eng/banco-de-signos)"
REQ_TIMEOUT = 18
MAX_SITEMAP_LINKS = 200  # tope global desde sitemaps
MAX_HTML_FETCH = 60      # tope global de HTML fallback (título/description)
ALLOWED_SCRAPE = set([d.strip().lower() for d in os.getenv("ALLOWED_SCRAPE_DOMAINS","").split(",") if d.strip()])

def http_get(url):
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT, allow_redirects=True)
        if r.status_code == 200 and r.content:
            return r
    except Exception:
        return None
    return None

def autodiscover_feeds(home_url):
    r = http_get(home_url)
    urls = []
    if not r: return urls
    try:
        soup = BeautifulSoup(r.text, "lxml")
    except Exception:
        soup = BeautifulSoup(r.text, "html.parser")
    for link in soup.find_all("link", rel=lambda x: x and "alternate" in x):
        t = (link.get("type") or "").lower()
        if "rss" in t or "atom" in t or "xml" in t:
            href = link.get("href")
            if href:
                urls.append(urljoin(home_url, href))
    return list(dict.fromkeys(urls))

def parse_entry_dt_feed(e):
    # Intenta varios campos de fecha del feed
    for k in ("published_parsed","updated_parsed"):
        t = e.get(k)
        if t:
            try:
                return dt.datetime.fromtimestamp(time.mktime(t), tz=dt.timezone.utc)
            except Exception:
                pass
    for k in ("published","updated"):
        s = e.get(k)
        if s:
            try:
                d = parsedate_to_datetime(s)
                return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
            except Exception:
                pass
    return None

def parse_feed(url):
    d = feedparser.parse(url, request_headers={"User-Agent": UA})
    items = []
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=MAX_AGE_DAYS)
    for e in d.entries:
        link = (e.get("link") or "").split("?")[0]
        if not link: continue
        dpub = parse_entry_dt_feed(e) or dt.datetime.now(dt.timezone.utc)  # fallback
        if dpub < cutoff: continue
        items.append({
            "url": link,
            "title": (e.get("title","") or "").strip() or "(Sin título)",
            "summary": (e.get("summary","") or "").strip(),
            "published_dt": dpub,
        })
    return items

def parse_sitemap(url):
    r = http_get(url)
    if not r: return []
    try:
        soup = BeautifulSoup(r.content, "xml")
    except Exception:
        return []
    out = []
    now_utc = dt.datetime.now(dt.timezone.utc)
    cutoff = now_utc - dt.timedelta(days=MAX_AGE_DAYS)

    idx = soup.find_all("sitemap")
    if idx:
        # prioriza sitemaps con "news"/"latest" y lastmod más reciente
        def score(sm):
            loc = (sm.find("loc").get_text() if sm.find("loc") else "").lower()
            lastmod = sm.find("lastmod").get_text() if sm.find("lastmod") else ""
            ts = 0
            try:
                ts = parsedate_to_datetime(lastmod).timestamp()
            except Exception:
                pass
            return (("news" in loc) or ("latest" in loc), ts)
        subs = sorted(idx, key=score, reverse=True)
        urls = [sm.find("loc").get_text() for sm in subs if sm.find("loc")]
        collected = []
        for su in urls[:10]:
            collected += parse_sitemap(su)
            if len(collected) >= MAX_SITEMAP_LINKS:
                break
        return collected[:MAX_SITEMAP_LINKS]

    for u in soup.find_all("url"):
        loc = u.find("loc").get_text() if u.find("loc") else None
        if not loc: continue
        last = u.find("lastmod").get_text() if u.find("lastmod") else None
        dpub = None
        if last:
            try:
                dd = parsedate_to_datetime(last)
                dpub = dd if dd.tzinfo else dd.replace(tzinfo=dt.timezone.utc)
            except Exception:
                pass
        if not dpub: dpub = now_utc
        if dpub < cutoff: continue
        out.append({"url": loc, "published_dt": dpub})
    return out

def fetch_title_desc(url):
    # Solo si el dominio está permitido
    from urllib.parse import urlparse
    dom = urlparse(url).netloc.lower()
    dom = dom[4:] if dom.startswith("www.") else dom
    if dom not in ALLOWED_SCRAPE:
        return None, None
    r = http_get(url)
    if not r: return None, None
    try:
        soup = BeautifulSoup(r.text, "lxml")
    except Exception:
        soup = BeautifulSoup(r.text, "html.parser")
    title = (soup.title.string.strip() if soup.title and soup.title.string else None)
    desc = None
    m = soup.find("meta", attrs={"name":"description"}) or soup.find("meta", attrs={"property":"og:description"})
    if m and m.get("content"): desc = m.get("content").strip()
    return title, desc

def ingest_sources(sources, now_utc):
    seen = set()
    items = []
    html_fetches = 0
    for s in sources:
        # 1) RSS explícitos
        for ru in (s.get("rss") or []):
            try:
                for it in parse_feed(ru):
                    u = it["url"]
                    if u in seen: continue
                    seen.add(u); it["source"] = s["name"]; items.append(it)
            except Exception:
                continue
        # 2) Autodiscovery desde home
        try:
            for fu in autodiscover_feeds(s["home"]):
                for it in parse_feed(fu):
                    u = it["url"]
                    if u in seen: continue
                    seen.add(u); it["source"] = s["name"]; items.append(it)
        except Exception:
            pass
        # 3) Sitemaps (respaldo)
        for su in (s.get("sitemaps") or []):
            try:
                sm_items = parse_sitemap(su)
            except Exception:
                sm_items = []
            for si in sm_items:
                u = si["url"]
                if u in seen: continue
                title, desc = (None, None)
                if html_fetches < MAX_HTML_FETCH:
                    t, d = fetch_title_desc(u)
                    if t: title = t
                    if d: desc = d
                    if t or d: html_fetches += 1
                items.append({
                    "source": s["name"],
                    "url": u,
                    "title": title or "(Sin título)",
                    "summary": desc or "",
                    "published_dt": si["published_dt"],
                })
                seen.add(u)
    items.sort(key=lambda r: r["published_dt"], reverse=True)
    return items

# ------------ Utilidades de análisis ------------
def is_opinion(title, summary):
    t = (title or "").lower(); s = (summary or "").lower()
    return any(h in t or h in s for h in OPINION_HINTS)

def tokenize(text):
    return [w.lower() for w in TOKEN_RE.findall(text or "") if w and w.lower() not in STOPWORDS and len(w) >= 4]

def top_terms(texts, k=5):
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    return [w for w,_ in c.most_common(k)]

def load_history():
    if HIST_FILE.exists():
        data = np.load(HIST_FILE, allow_pickle=False)
        return data["E"], data["d"]
    return None, None

def save_history(E_hist, d_hist):
    np.savez_compressed(HIST_FILE, E=E_hist, d=d_hist)

def build_clusters(E, items, novelty):
    n = E.shape[0]
    S = E @ E.T; np.fill_diagonal(S, 0.0)
    adj = [[] for _ in range(n)]
    for i in range(n):
        js = np.where(S[i] >= SIM_THRESHOLD)[0]
        for j in js:
            adj[i].append(j); adj[j].append(i)

    visited = [False]*n; raw_clusters = []
    for i in range(n):
        if visited[i]: continue
        comp = []; q = deque([i]); visited[i] = True
        while q:
            u = q.popleft(); comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True; q.append(v)
        if CLUSTER_MIN <= len(comp) <= CLUSTER_MAX:
            raw_clusters.append(sorted(comp))

    results = []
    for comp in raw_clusters:
        comp_items = [items[i] for i in comp]
        comp_texts = [it["title"] + " " + (it["summary"] or "") for it in comp_items]
        comp_nov = float(np.mean([novelty[i] for i in comp]))
        sources = [it["source"] for it in comp_items]
        diversity = len(set(sources)) / max(1, len(comp))
        size = len(comp); size_pref = 1.0 if size <= 6 else 0.8
        terms = top_terms(comp_texts, k=5)
        opinions = any(is_opinion(it["title"], it["summary"]) for it in comp_items)
        opinion_bonus = 0.05 if opinions else 0.0
        score = max(0.0, min(1.0, 0.55*comp_nov + 0.30*diversity + 0.10*size_pref + opinion_bonus))
        results.append({
            "idxs": comp,
            "size": size,
            "avg_novelty": round(comp_nov, 3),
            "diversity": round(diversity, 3),
            "opinion": opinions,
            "terms": terms,
            "score": round(score, 3),
            "sources": sorted(set(sources)),
            "items": comp_items,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results

def build_html(today, clusters, singles, total_count):
    fn = PUBLISH_DIR / f"weak_signals_{today}.html"
    parts = []
    parts += [
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Banco de Signos — {today}</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;background:#fafafa;color:#222}",
        ".card{background:#fff;border:1px solid #eee;border-radius:10px;padding:16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,0.03)}",
        ".badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;margin-left:6px;color:#fff}",
        ".hi{background:#0b8457}.med{background:#f1a208}.lo{background:#888}",
        "a{color:#0a58ca;text-decoration:none} a:hover{text-decoration:underline}",
        ".meta{font-size:12px;color:#555;margin-top:6px}",
        "ul{margin:6px 0 0 18px}",
        "</style></head><body>",
        f"<h1>Banco de Signos — Señales (objetivo {TARGET_MIN}-{TARGET_MAX}) · {today}</h1>",
        "<p>Micro‑clústeres recientes (novedad + diversidad) y piezas individuales (novedad × recencia). Sin duplicados. Límite por fuente global.</p>",
        f"<p><strong>Total publicado:</strong> {total_count} piezas</p>",
    ]
    if clusters:
        parts.append("<h2>Señales débiles (micro‑clústeres)</h2>")
        for c in clusters:
            cls = "hi" if c["score"] >= 0.6 else ("med" if c["score"] >= 0.4 else "lo")
            title = " / ".join(c["terms"]) if c["terms"] else "Micro‑tema"
            parts.append("<div class='card'>")
            parts.append(f"<h3>{title} <span class='badge {cls}' title='score'>{c['score']:.2f}</span></h3>")
            bullets = []
            bullets.append(f"Tamaño: {c['size']} · Fuentes: {len(c['sources'])} ({', '.join(c['sources'][:6])})")
            bullets.append(f"Novedad media: {c['avg_novelty']:.2f} · Diversidad: {c['diversity']:.2f}")
            if c["opinion"]:
                bullets.append("Incluye opinión/análisis")
            parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
            for it in c["items"]:
                meta = f"{it['source']} · {it['published_dt'].isoformat(timespec='minutes')}"
                parts.append(f"<p><a href='{it['url']}' target='_blank' rel='noopener'>{it['title']}</a><br><span class='meta'>{meta}</span></p>")
            parts.append("</div>")
    else:
        parts.append("<p>No se han detectado micro‑clústeres hoy.</p>")

    if singles:
        parts.append("<h2>Más señales (piezas individuales: novedad × recencia)</h2>")
        for r in singles:
            score = r["score"]
            cls = "hi" if score >= 0.6 else ("med" if score >= 0.4 else "lo")
            parts.append("<div class='card'>")
            parts.append(f"<h3><a href='{r['url']}' target='_blank' rel='noopener'>{r['title']}</a> "
                         f"<span class='badge {cls}'>{score:.2f}</span></h3>")
            if r.get("summary"):
                short = r["summary"]; short = (short[:320] + "…") if len(short) > 340 else short
                parts.append(f"<p>{short}</p>")
            meta = f"{r['source']} · {r['published_iso']} · novedad {r['novelty']:.2f} · hace {r['age_hours']:.0f}h"
            parts.append(f"<div class='meta'>{meta}</div>")
            parts.append("</div>")

    parts.append("<hr><small>RSS/Autodiscovery/Sitemaps. Clúster = novedad×diversidad×preferencia por micro; piezas = novedad×recencia. Ventana: "
                 f"{MAX_AGE_DAYS} días.</small>")
    parts.append("</body></html>")
    fn.write_text("\n".join(parts), encoding="utf-8")
    return fn.name

def build_index():
    files = sorted([p.name for p in PUBLISH_DIR.glob("weak_signals_*.html")], reverse=True)[:30]
    idx = PUBLISH_DIR / "index.html"
    lines = [
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Banco de Signos — Índice</title>",
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px}</style>",
        "</head><body><h1>Banco de Signos — Informes</h1><ul>"
    ]
    if not files:
        lines.append("<li>Aún no hay informes. Ejecuta el workflow y espera unos minutos.</li>")
    for f in files:
        ymd = f.replace("weak_signals_","").replace(".html","")
        lines.append(f"<li><a href='{f}'>{ymd}</a></li>")
    lines.append("</ul><p>Últimos 30 días.</p></body></html>")
    idx.write_text("\n".join(lines), encoding="utf-8")

# ------------ Pipeline principal ------------
def main():
    now_utc = dt.datetime.now(dt.timezone.utc)
    today_iso = now_utc.date().isoformat()

    # 1) Ingesta robusta
    items = ingest_sources(SOURCES, now_utc)
    if not items:
        build_index()
        print("Sin artículos recientes.")
        return

    # 2) Embeddings (título + resumen)
    texts = [(it["title"] + " || " + (it["summary"] or "")[:1000]).strip() for it in items]
    model = SentenceTransformer(MODEL_NAME)
    E_today = model.encode(texts, batch_size=16, normalize_embeddings=True)
    E_today = np.asarray(E_today).astype(np.float32)

    # 3) Novedad vs histórico
    E_hist, d_hist = load_history()
    cutoff = (now_utc.date() - dt.timedelta(days=HISTORY_DAYS)).toordinal()
    if E_hist is not None and d_hist is not None:
        mask = d_hist >= cutoff; E_hist, d_hist = E_hist[mask], d_hist[mask]
    if E_hist is None or E_hist.shape[0] == 0:
        novelty = np.ones(E_today.shape[0], dtype=np.float32)
    else:
        sims = E_today @ E_hist.T; novelty = 1.0 - np.max(sims, axis=1)

    # 4) Piezas individuales (novedad × recencia)
    picked = []
    for i, it in enumerate(items):
        age_h = max(0.0, (now_utc - it["published_dt"]).total_seconds() / 3600.0)
        recency_w = math.exp(-age_h / float(RECENCY_TAU_HOURS))
        score = float(novelty[i]) * recency_w
        picked.append({
            **it,
            "published_iso": it["published_dt"].isoformat(timespec="minutes"),
            "novelty": float(novelty[i]),
            "age_hours": age_h,
            "score": score,
            "url": it["url"],
        })
    picked.sort(key=lambda r: r["score"], reverse=True)

    # 5) Clústeres
    clusters = build_clusters(E_today, items, novelty)

    # 6) Selección final (objetivo 40–50) sin duplicados y con límite por fuente global
    cluster_urls = set()
    source_counts = {}
    total_count = 0
    for c in clusters:
        for it in c["items"]:
            u = it["url"]
            cluster_urls.add(u)
            source_counts[it["source"]] = source_counts.get(it["source"], 0) + 1
            total_count += 1

    singles = []
    for r in picked:
        if r["url"] in cluster_urls:
            continue
        src = r["source"]
        if source_counts.get(src, 0) >= MAX_PER_SOURCE:
            continue
        if total_count >= TARGET_MAX:
            break
        singles.append(r)
        source_counts[src] = source_counts.get(src, 0) + 1
        total_count += 1

    if total_count < TARGET_MIN:
        for r in picked:
            if r["url"] in cluster_urls or any(s["url"] == r["url"] for s in singles):
                continue
            if total_count >= TARGET_MIN:
                break
            singles.append(r)
            total_count += 1

    # 7) Publicación
    build_html(today_iso, clusters, singles, total_count)
    build_index()

    # 8) Actualiza histórico (recorte a 60 días)
    today_ord = now_utc.date().toordinal()
    d_today = np.full((E_today.shape[0],), today_ord, dtype=np.int32)
    if E_hist is None or d_hist is None:
        E_new, d_new = E_today, d_today
    else:
        E_new = np.vstack([E_hist, E_today]); d_new = np.concatenate([d_hist, d_today])
        keep = d_new >= (now_utc.date() - dt.timedelta(days=60)).toordinal()
        E_new, d_new = E_new[keep], d_new[keep]
    save_history(E_new, d_new)

    print(f"Publicado: {total_count} (clústeres: {sum(len(c['items']) for c in clusters)} · singles: {len(singles)})")

if __name__ == "__main__":
    main()
