# ci_pipeline.py
# Banco de Signos — Señales débiles: recencia + novedad + micro-clústeres + marcos + opinión + aceleración (7 días)
import os, time, math, datetime as dt, re
from pathlib import Path
from email.utils import parsedate_to_datetime
from collections import Counter, deque

import feedparser
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------ Configuración ------------
# --------- FUENTES (por dominio) ---------
SOURCES = [
    # nombre, home, rss (lista, puede estar vacía), sitemaps opcionales
    {"name":"El Periódico","home":"https://www.elperiodico.com/es/","rss":["https://www.elperiodico.com/es/rss/rss_portada.xml"],"sitemaps":["https://www.elperiodico.com/es/sitemap.xml"]},
    {"name":"La Voz de Galicia","home":"https://www.lavozdegalicia.es/","rss":["https://www.lavozdegalicia.es/rss/index.xml"],"sitemaps":["https://www.lavozdegalicia.es/sitemap.xml"]},
    {"name":"SModa","home":"https://smoda.elpais.com/","rss":["https://smoda.elpais.com/feed/"],"sitemaps":["https://smoda.elpais.com/sitemap.xml"]},
    {"name":"Código Nuevo","home":"https://www.codigonuevo.com/","rss":["https://www.codigonuevo.com/feed"],"sitemaps":["https://www.codigonuevo.com/sitemap.xml"]},
    {"name":"Reason Why","home":"https://www.reasonwhy.es/","rss":["https://www.reasonwhy.es/rss","https://www.reasonwhy.es/rss.xml"],"sitemaps":["https://www.reasonwhy.es/sitemap.xml"]},
    {"name":"Control Publicidad","home":"https://controlpublicidad.com/","rss":["https://controlpublicidad.com/feed/"],"sitemaps":["https://controlpublicidad.com/sitemap.xml"]},
    {"name":"El Confidencial","home":"https://www.elconfidencial.com/","rss":["https://www.elconfidencial.com/rss/ultimas_noticias/"],"sitemaps":["https://www.elconfidencial.com/sitemap.xml"]},
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
    {"name":"The Guardian","home":"https://www.theguardian.com/international","rss":[
        "https://www.theguardian.com/world/rss",
        "https://www.theguardian.com/technology/rss",
        "https://www.theguardian.com/business/rss",
        "https://www.theguardian.com/science/rss"
    ],"sitemaps":["https://www.theguardian.com/sitemaps/news.xml"]},
]

# --------- INGESTA ROBUSTA (RSS → autodiscovery → sitemaps → fallback título/desc) ---------
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime

UA = "Mozilla/5.0 (compatible; BancoDeSignos/1.0; +https://github.com/lgallegof-eng/banco-de-signos)"
REQ_TIMEOUT = 18
MAX_AGE_DAYS = 14  # usa el valor que tengas configurado en tu script
MAX_SITEMAP_LINKS = 200     # tope global desde sitemaps
MAX_HTML_FETCH = 60         # tope global de páginas para titulo/description
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
    soup = BeautifulSoup(r.text, "lxml")
    for link in soup.find_all("link", rel=lambda x: x and "alternate" in x):
        t = (link.get("type") or "").lower()
        if "rss" in t or "atom" in t or "xml" in t:
            href = link.get("href")
            if href:
                urls.append(urljoin(home_url, href))
    return list(dict.fromkeys(urls))

def parse_feed(url):
    d = feedparser.parse(url, request_headers={"User-Agent": UA})
    items = []
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=MAX_AGE_DAYS)
    for e in d.entries:
        link = (e.get("link") or "").split("?")[0]
        if not link: continue
        # fecha
        dpub = None
        for k in ("published_parsed","updated_parsed"):
            t = e.get(k)
            if t:
                try:
                    dpub = dt.datetime.fromtimestamp(time.mktime(t), tz=dt.timezone.utc)
                    break
                except Exception:
                    pass
        if not dpub:
            for k in ("published","updated"):
                s = e.get(k)
                if s:
                    try:
                        dd = parsedate_to_datetime(s)
                        dpub = dd if dd.tzinfo else dd.replace(tzinfo=dt.timezone.utc)
                        break
                    except Exception:
                        pass
        if not dpub: dpub = dt.datetime.now(dt.timezone.utc)  # fallback
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

    # sitemapindex → recorrer sub-sitemaps (prioriza news)
    idx = soup.find_all("sitemap")
    if idx:
        # prioriza sitemaps con "news" o los más recientes
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
        for su in urls[:10]:  # limita sub-sitemaps
            collected += parse_sitemap(su)
            if len(collected) >= MAX_SITEMAP_LINKS:
                break
        return collected[:MAX_SITEMAP_LINKS]

    # urlset → URLs
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
        if not dpub:
            dpub = now_utc
        if dpub < cutoff: continue
        out.append({"url": loc, "published_dt": dpub})
    return out

def fetch_title_desc(url):
    # Solo si el dominio está permitido
    dom = urlparse(url).netloc.lower()
    dom = dom[4:] if dom.startswith("www.") else dom
    if dom not in ALLOWED_SCRAPE:
        return None, None
    r = http_get(url)
    if not r: return None, None
    soup = BeautifulSoup(r.text, "lxml")
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
                    seen.add(u)
                    it["source"] = s["name"]
                    items.append(it)
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
        # 3) Sitemaps (como feed de respaldo)
        for su in (s.get("sitemaps") or []):
            try:
                sm_items = parse_sitemap(su)
            except Exception:
                sm_items = []
            for si in sm_items:
                u = si["url"]
                if u in seen: continue
                # Fallback título/description muy limitado
                title, desc = (None, None)
                if html_fetches < MAX_HTML_FETCH:
                    t, d = fetch_title_desc(u)
                    if t: title = t
                    if d: desc = d
                    html_fetches += 1 if (t or d) else 0
                items.append({
                    "source": s["name"],
                    "url": u,
                    "title": title or "(Sin título)",
                    "summary": desc or "",
                    "published_dt": si["published_dt"],
                })
                seen.add(u)
    # Orden por fecha
    items.sort(key=lambda r: r["published_dt"], reverse=True)
    return items

]

# Señales
TOP_K_ITEMS = 25            # respaldo de piezas si no hay clústeres
MAX_PER_SOURCE = 4          # límite por fuente en respaldo
# Ventanas/umbrales
HISTORY_DAYS = 21           # histórico para novedad
MAX_AGE_DAYS = 7            # analizamos últimos 7 días
RECENCY_TAU_HOURS = 96      # semivida recencia
SIM_THRESHOLD = 0.72        # similitud mínima para clusterizar
CLUSTER_MIN = 2             # micro-clúster mínimo
CLUSTER_MAX = 8             # micro-clúster máximo

# Publicación
PUBLISH_DIR = Path(os.getenv("PUBLISH_DIR", "public")).resolve()
HIST_DIR = Path(os.getenv("HIST_DIR", "public/data")).resolve()
PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

HIST_FILE = HIST_DIR / "history.npz"
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# ------------ Marcos y opinión (diccionarios ES/EN) ------------
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

# ------------ Utilidades ------------
def parse_entry_dt(entry):
    for k in ("published_parsed", "updated_parsed"):
        t = entry.get(k)
        if t:
            try:
                return dt.datetime.fromtimestamp(time.mktime(t), tz=dt.timezone.utc)
            except Exception:
                pass
    for k in ("published", "updated"):
        s = entry.get(k)
        if s:
            try:
                d = parsedate_to_datetime(s)
                return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
            except Exception:
                pass
    return None

def is_opinion(title, summary):
    t = (title or "").lower(); s = (summary or "").lower()
    return any(h in t or h in s for h in OPINION_HINTS)

def frame_profile(text):
    txt = (text or "").lower()
    scores = {}
    for f, kws in FRAMES.items():
        c = sum(1 for kw in kws if kw in txt)
        scores[f] = c
    total = sum(scores.values()) or 1
    for k in scores:
        scores[k] = scores[k] / total
    top = max(scores, key=scores.get) if scores else None
    return top, scores

def tokenize(text):
    return [w.lower() for w in TOKEN_RE.findall(text or "") if w and w.lower() not in STOPWORDS and len(w) >= 4]

def top_terms(texts, k=5):
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    return [w for w,_ in c.most_common(k)]

def fetch_items(feeds, now_utc):
    items, seen = [], set()
    cutoff = now_utc - dt.timedelta(days=MAX_AGE_DAYS)
    for url in feeds:
        try:
            d = feedparser.parse(url)
        except Exception:
            continue
        source = d.feed.get("title", url)
        for e in d.entries:
            link = (e.get("link") or "").split("?")[0]
            if not link or link in seen: continue
            seen.add(link)
            dpub = parse_entry_dt(e)
            if not dpub or dpub < cutoff: continue
            title = (e.get("title", "") or "").strip() or "(Sin título)"
            summary = (e.get("summary", "") or "").strip()
            items.append({
                "source": source,
                "title": title,
                "url": link,
                "summary": summary,
                "published_dt": dpub.astimezone(dt.timezone.utc),
            })
    items.sort(key=lambda r: r["published_dt"], reverse=True)
    return items

def load_history():
    if HIST_FILE.exists():
        data = np.load(HIST_FILE, allow_pickle=False)
        return data["E"], data["d"]
    return None, None

def save_history(E_hist, d_hist):
    np.savez_compressed(HIST_FILE, E=E_hist, d=d_hist)

def build_clusters(E, items, novelty, now_utc, frame_global_freq):
    # Matriz de similitud
    n = E.shape[0]
    S = E @ E.T
    np.fill_diagonal(S, 0.0)

    # Grafo por umbral y componentes conexas
    adj = [[] for _ in range(n)]
    for i in range(n):
        js = np.where(S[i] >= SIM_THRESHOLD)[0]
        for j in js:
            adj[i].append(j)
            adj[j].append(i)

    visited = [False]*n
    raw_clusters = []
    for i in range(n):
        if visited[i]: continue
        comp = []
        q = deque([i]); visited[i] = True
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
        opinions = any(is_opinion(it["title"], it["summary"]) for it in comp_items)
        # Marco dominante y rareza global
        frame_counts = Counter()
        for it in comp_items:
            topf, _ = frame_profile(it["title"] + " " + it["summary"])
            if topf: frame_counts[topf] += 1
        top_frame, _cnt = (frame_counts.most_common(1)[0] if frame_counts else (None, 0))
        rarity = 0.0
        if top_frame and frame_global_freq:
            rarity = 1.0 - frame_global_freq.get(top_frame, 0.0)

        # Aceleración: últimos 3 días vs 4 anteriores (dentro de los 7 días)
        ages_d = [max(0, (now_utc.date() - it["published_dt"].date()).days) for it in comp_items]
        last3 = sum(1 for d in ages_d if d <= 2)
        prev4 = sum(1 for d in ages_d if 3 <= d <= 6)
        accel = (last3 + 1) / (prev4 + 1)  # suavizado
        # Normaliza aceleración a 0..1 (>=2.5 satura)
        accel_norm = max(0.0, min(1.0, (accel - 1.0) / 1.5))

        # Preferencia por micro-tamaño (2-6)
        size = len(comp)
        size_pref = 1.0 if size <= 6 else 0.8

        # Términos representativos
        terms = top_terms(comp_texts, k=5)

        # Puntuación compuesta
        opinion_bonus = 0.05 if opinions else 0.0
        score = (0.40*comp_nov + 0.20*diversity + 0.10*size_pref +
                 0.15*accel_norm + 0.15*rarity + opinion_bonus)
        score = max(0.0, min(1.0, score))

        results.append({
            "idxs": comp,
            "size": size,
            "avg_novelty": round(comp_nov, 3),
            "diversity": round(diversity, 3),
            "opinion": opinions,
            "top_frame": top_frame,
            "rarity": round(rarity, 3),
            "accel": round(accel, 2),
            "accel_norm": round(accel_norm, 3),
            "terms": terms,
            "score": round(score, 3),
            "sources": sorted(set(sources)),
            "items": comp_items,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results

def build_html(today, clusters, picked_items):
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
        f"<h1>Banco de Signos — Señales débiles (últimos {MAX_AGE_DAYS} días)</h1>",
        "<p>Micro‑clústeres recientes con alta novedad, diversidad de fuentes, opinión/análisis, marcos poco frecuentes y aceleración (3d vs 4d prev).</p>",
    ]
    # Clústeres
    if clusters:
        for c in clusters:
            cls = "hi" if c["score"] >= 0.6 else ("med" if c["score"] >= 0.4 else "lo")
            title = " / ".join(c["terms"]) if c["terms"] else "Micro‑tema"
            parts.append("<div class='card'>")
            parts.append(f"<h2>{title} <span class='badge {cls}' title='score'>{c['score']:.2f}</span></h2>")
            bullets = []
            bullets.append(f"Tamaño: {c['size']} · Fuentes: {len(c['sources'])} ({', '.join(c['sources'][:5])})")
            bullets.append(f"Novedad media: {c['avg_novelty']:.2f} · Diversidad: {c['diversity']:.2f}")
            if c["top_frame"]:
                bullets.append(f"Marco dominante: {c['top_frame']} · Rareza: {c['rarity']:.2f}")
            bullets.append(f"Aceleración (3d/4d prev): {c['accel']:.2f}")
            if c["opinion"]:
                bullets.append("Incluye opinión/análisis")
            parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
            for it in c["items"]:
                meta = f"{it['source']} · {it['published_dt'].isoformat(timespec='minutes')}"
                parts.append(f"<p><a href='{it['url']}' target='_blank' rel='noopener'>{it['title']}</a><br><span class='meta'>{meta}</span></p>")
            parts.append("</div>")
    else:
        parts.append("<p>No se han detectado micro‑clústeres hoy. Mostramos piezas destacadas por novedad+recencia.</p>")

    # Respaldo: piezas individuales (si no hay clústeres)
    if picked_items and not clusters:
        parts.append("<h2>Piezas destacadas (novedad × recencia)</h2>")
        for r in picked_items:
            score = r["score"]
            cls = "hi" if score >= 0.6 else ("med" if score >= 0.4 else "lo")
            parts.append("<div class='card'>")
            parts.append(f"<h3><a href='{r['url']}' target='_blank' rel='noopener'>{r['title']}</a> "
                         f"<span class='badge {cls}'>{score:.2f}</span></h3>")
            if r.get("summary"):
                short = r["summary"]; short = (short[:320] + "…") if len(short) > 340 else short
                parts.append(f"<p>{short}</p>")
            meta = f"{r['source']} · {r['published_iso']} · novedad {r['novelty']:.2f} · hace {r['age_hours']:.0f}h"
            if r.get("top_frame"): meta += f" · marco: {r['top_frame']}"
            if r.get("opinion"): meta += " · opinión/análisis"
            parts.append(f"<div class='meta'>{meta}</div>")
            parts.append("</div>")

    parts.append("<hr><small>RSS públicos. Puntuación de clúster = novedad×diversidad×preferencia por micro + aceleración (3d/4d) + rareza del marco + bonus de opinión.</small>")
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

# ------------ Pipeline ------------
def main():
    now_utc = dt.datetime.now(dt.timezone.utc)
    today_iso = now_utc.date().isoformat()

    # 1) Ingesta (últimos 7 días)
    items = fetch_items(FEEDS, now_utc)
    if not items:
        build_index()
        print("Sin artículos recientes.")
        return

    # 2) Embeddings de la ventana (título + resumen)
    texts = [(it["title"] + " || " + (it["summary"] or "")[:1000]).strip() for it in items]
    model = SentenceTransformer(MODEL_NAME)
    E_today = model.encode(texts, batch_size=16, normalize_embeddings=True)
    E_today = np.asarray(E_today).astype(np.float32)

    # 3) Novedad vs. histórico
    E_hist, d_hist = load_history()
    cutoff = (now_utc.date() - dt.timedelta(days=HISTORY_DAYS)).toordinal()
    if E_hist is not None and d_hist is not None:
        mask = d_hist >= cutoff
        E_hist, d_hist = E_hist[mask], d_hist[mask]
    if E_hist is None or E_hist.shape[0] == 0:
        novelty = np.ones(E_today.shape[0], dtype=np.float32)
    else:
        sims = E_today @ E_hist.T
        novelty = 1.0 - np.max(sims, axis=1)

    # 4) Perfil global de marcos (rareza)
    frame_global_counts = Counter()
    for it in items:
        topf, _ = frame_profile(it["title"] + " " + it["summary"])
        if topf: frame_global_counts[topf] += 1
    total_frames = sum(frame_global_counts.values()) or 1
    frame_global_freq = {k: v/total_frames for k,v in frame_global_counts.items()}

    # 5) Micro‑clústeres (7 días) con aceleración
    clusters = build_clusters(E_today, items, novelty, now_utc, frame_global_freq)

    # 6) Respaldo de piezas (novedad × recencia) si no hubiera clústeres
    picked = []
    for i, it in enumerate(items):
        age_h = max(0.0, (now_utc - it["published_dt"]).total_seconds() / 3600.0)
        recency_w = math.exp(-age_h / float(RECENCY_TAU_HOURS))
        score = float(novelty[i]) * recency_w
        topf, _ = frame_profile(it["title"] + " " + it["summary"])
        picked.append({
            **it,
            "published_iso": it["published_dt"].isoformat(timespec="minutes"),
            "novelty": float(novelty[i]),
            "age_hours": age_h,
            "score": score,
            "top_frame": topf,
            "opinion": is_opinion(it["title"], it["summary"]),
            "url": it["url"],
        })
    picked.sort(key=lambda r: r["score"], reverse=True)
    out_items, seen_per_source = [], {}
    for r in picked:
        s = r["source"]
        if seen_per_source.get(s, 0) < MAX_PER_SOURCE:
            out_items.append(r)
            seen_per_source[s] = seen_per_source.get(s, 0) + 1
        if len(out_items) >= TOP_K_ITEMS:
            break

    # 7) Publicación
    build_html(today_iso, clusters, out_items if not clusters else [])
    build_index()

    # 8) Actualizar histórico (recorte a 60 días)
    today_ord = now_utc.date().toordinal()
    d_today = np.full((E_today.shape[0],), today_ord, dtype=np.int32)
    if E_hist is None or d_hist is None:
        E_new, d_new = E_today, d_today
    else:
        E_new = np.vstack([E_hist, E_today])
        d_new = np.concatenate([d_hist, d_today])
        keep = d_new >= (now_utc.date() - dt.timedelta(days=60)).toordinal()
        E_new, d_new = E_new[keep], d_new[keep]
    save_history(E_new, d_new)

    print(f"Informe generado. Clústeres: {len(clusters)}")

if __name__ == "__main__":
    main()
