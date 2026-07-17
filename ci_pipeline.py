# ci_pipeline.py
# Banco de Signos — Señales débiles: recencia + novedad + micro-clústeres + marcos + opinión
import os, time, math, datetime as dt, re
from pathlib import Path
from email.utils import parsedate_to_datetime
from collections import Counter, deque, defaultdict

import feedparser
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------ Configuración ------------
FEEDS = [
    # ——— España ———
    "https://www.elperiodico.com/es/rss/rss_portada.xml",
    "https://www.lavozdegalicia.es/rss/index.xml",
    "https://smoda.elpais.com/feed/",
    "https://www.codigonuevo.com/feed",
    "https://www.reasonwhy.es/rss",
    "https://controlpublicidad.com/feed/",
    "https://www.elconfidencial.com/rss/ultimas_noticias/",
    # ——— Internacional ———
    "https://www.theatlantic.com/feed/all/",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.fastcompany.com/rss",
    "https://www.highsnobiety.com/feed/",
    "https://www.ft.com/rss/home",
    "https://www.businessinsider.com/rss",
    "https://www.businessinsider.es/rss",
    "https://feeds.bloomberg.com/markets/news.rss",
    # puedes añadir más
]

TOP_K_ITEMS = 25            # listado de respaldo (si faltan clústeres)
HISTORY_DAYS = 21           # ventana histórico para novedad
MAX_AGE_DAYS = 5            # recencia: últimos 5 días
RECENCY_TAU_HOURS = 96      # semivida recencia
SIM_THRESHOLD = 0.72        # umbral similitud para micro-clústeres
CLUSTER_MIN = 2             # tamaño mínimo de clúster
CLUSTER_MAX = 8             # tamaño máximo de clúster (señales débiles suelen ser micro)
MAX_PER_SOURCE = 4          # límite por cabecera (variedad)

# Directorios controlados por el workflow
PUBLISH_DIR = Path(os.getenv("PUBLISH_DIR", "public")).resolve()
HIST_DIR = Path(os.getenv("HIST_DIR", "public/data")).resolve()
PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

HIST_FILE = HIST_DIR / "history.npz"
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# ------------ Diccionarios de marcos (ES/EN) ------------
FRAMES = {
    "economía": [
        "precio", "precios", "inflación", "coste", "costes", "empleo", "desempleo",
        "recuperación", "salario", "PIB", "mercado", "inversión", "profits", "growth",
        "inflation", "cost", "costs", "jobs", "recession", "recovery", "gdp", "market",
        "investment", "funding", "valuation", "earnings"
    ],
    "salud": [
        "salud", "sanitario", "sanitaria", "epidemia", "pandemia", "hospital", "OMS",
        "mental", "bienestar", "contagio", "vacuna", "vaccine", "health", "public health",
        "hospital", "wellbeing", "mental health"
    ],
    "sostenibilidad": [
        "clima", "climático", "emisiones", "CO2", "energía", "renovable", "circular",
        "sostenible", "biodiversidad", "green", "sustainable", "renewable", "emissions",
        "net zero", "climate", "carbon"
    ],
    "privacidad/tech": [
        "privacidad", "datos", "algoritmo", "IA", "inteligencia artificial", "modelo",
        "plataforma", "red social", "app", "tracking", "cookies", "privacy", "data",
        "algorithm", "ai", "platform", "tracking", "surveillance", "biometrics"
    ],
    "justicia_social": [
        "género", "igualdad", "diversidad", "inclusión", "derechos", "discriminación",
        "racismo", "LGTBI", "feminismo", "equity", "diversity", "inclusion", "rights",
        "discrimination", "racism", "gender"
    ],
    "geopolítica": [
        "guerra", "conflicto", "frontera", "sanciones", "OTAN", "UE", "china", "rusia",
        "eeuu", "diplomacia", "war", "conflict", "border", "sanctions", "nato", "eu",
        "china", "russia", "us", "geopolitics"
    ],
    "consumo/estilo": [
        "consumo", "tendencia", "moda", "lifestyle", "estilo", "consumidor", "compra",
        "retail", "cultura", "estética", "trend", "style", "fashion", "consumer",
        "shopping", "culture"
    ],
}

OPINION_HINTS = [
    "opinión", "opinion", "análisis", "analysis", "editorial", "tribuna",
    "columna", "column", "op-ed", "view", "viewpoint", "perspective"
]

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
    t = (title or "").lower()
    s = (summary or "").lower()
    return any(h in t or h in s for h in OPINION_HINTS)

def frame_profile(text):
    txt = (text or "").lower()
    scores = {}
    for f, kws in FRAMES.items():
        c = sum(1 for kw in kws if kw in txt)
        scores[f] = c
    # normaliza
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
            if not link or link in seen:
                continue
            seen.add(link)
            dpub = parse_entry_dt(e)
            if not dpub or dpub < cutoff:
                continue
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

def build_clusters(E, items, novelty):
    # E: (n,d) normalizado. Construye grafo por umbral y extrae componentes
    n = E.shape[0]
    S = E @ E.T
    np.fill_diagonal(S, 0.0)
    adj = [[] for _ in range(n)]
    for i in range(n):
        # umbral de similitud
        js = np.where(S[i] >= SIM_THRESHOLD)[0]
        for j in js:
            adj[i].append(j)
            adj[j].append(i)
    visited = [False]*n
    clusters = []
    for i in range(n):
        if visited[i]: continue
        comp = []
        q = deque([i]); visited[i] = True
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        if CLUSTER_MIN <= len(comp) <= CLUSTER_MAX:
            clusters.append(sorted(comp))
    # Calcula métricas y etiquetas por clúster
    results = []
    for comp in clusters:
        comp_items = [items[i] for i in comp]
        comp_nov = float(np.mean([novelty[i] for i in comp]))
        sources = [it["source"] for it in comp_items]
        uniq_sources = len(set(sources))
        diversity = uniq_sources / max(1, len(comp))
        opinions = any(is_opinion(it["title"], it["summary"]) for it in comp_items)
        opinion_bonus = 0.15 if opinions else 0.0
        # marcos
        frame_counts = Counter()
        for it in comp_items:
            topf, _ = frame_profile((it["title"] + " " + it["summary"]))
            if topf: frame_counts[topf] += 1
        top_frame, top_frame_cnt = (frame_counts.most_common(1)[0] if frame_counts else (None, 0))
        # términos
        terms = top_terms([it["title"] + " " + it["summary"] for it in comp_items], k=5)
        # puntuación compuesta (0..1)
        # pequeña preferencia por microtamaño (2-6)
        size = len(comp)
        size_pref = 1.0 if size <= 6 else 0.8
        score = 0.5*comp_nov + 0.3*diversity + 0.2*size_pref + opinion_bonus
        results.append({
            "idxs": comp,
            "size": size,
            "avg_novelty": round(comp_nov, 3),
            "diversity": round(diversity, 3),
            "opinion": opinions,
            "top_frame": top_frame,
            "terms": terms,
            "score": round(min(score, 1.0), 3),
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
        "<p>Prioriza micro‑clústeres recientes con alta novedad, diversidad de fuentes, opinión/análisis y marcos poco frecuentes.</p>",
    ]
    # Sección de clústeres
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
                bullets.append(f"Marco dominante: {c['top_frame']}")
            if c["opinion"]:
                bullets.append("Incluye opinión/análisis")
            parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
            # muestras
            for it in c["items"]:
                meta = f"{it['source']} · {it['published_dt'].isoformat(timespec='minutes')}"
                parts.append(f"<p><a href='{it['url']}' target='_blank' rel='noopener'>{it['title']}</a><br><span class='meta'>{meta}</span></p>")
            parts.append("</div>")
    else:
        parts.append("<p>No se han detectado micro‑clústeres hoy. Mostramos piezas destacadas por novedad+recencia.</p>")

    # Respaldo: piezas individuales si faltan clústeres
    if picked_items:
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
            meta = f"{r['source']} · {r.get('published_iso','')} · novedad {r['novelty']:.2f} · hace {r['age_hours']:.0f}h"
            if r.get("top_frame"):
                meta += f" · marco: {r['top_frame']}"
            if r.get("opinion"):
                meta += " · opinión/análisis"
            parts.append(f"<div class='meta'>{meta}</div>")
            parts.append("</div>")

    parts.append("<hr><small>RSS públicos. Score de clúster = novedad×diversidad×tamaño (micro) + bonus opinión. "
                 "Los marcos se estiman con diccionarios ES/EN.</small>")
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

    items = fetch_items(FEEDS, now_utc)
    # Genera siempre índice (evita 404)
    if not items:
        build_index()
        print("Sin artículos recientes.")
        return

    # Texto para embeddings
    texts = [(it["title"] + " || " + (it["summary"] or "")[:1000]).strip() for it in items]

    # Modelo multilingüe
    model = SentenceTransformer(MODEL_NAME)
    E_today = model.encode(texts, batch_size=16, normalize_embeddings=True)
    E_today = np.asarray(E_today).astype(np.float32)

    # Cargar histórico y recortar ventana
    E_hist, d_hist = load_history()
    cutoff = (now_utc.date() - dt.timedelta(days=HISTORY_DAYS)).toordinal()
    if E_hist is not None and d_hist is not None:
        mask = d_hist >= cutoff
        E_hist, d_hist = E_hist[mask], d_hist[mask]

    # Novedad vs. histórico
    if E_hist is None or E_hist.shape[0] == 0:
        novelty = np.ones(E_today.shape[0], dtype=np.float32)
    else:
        sims = E_today @ E_hist.T
        novelty = 1.0 - np.max(sims, axis=1)

    # Recencia y score individual (respaldo)
    picked = []
    for i, it in enumerate(items):
        age_h = max(0.0, (now_utc - it["published_dt"]).total_seconds() / 3600.0)
        recency_w = math.exp(-age_h / float(RECENCY_TAU_HOURS))
        score = float(novelty[i]) * recency_w
        topf, _ = frame_profile((it["title"] + " " + it["summary"]))
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
    # limitar por fuente para variedad
    picked.sort(key=lambda r: r["score"], reverse=True)
    out_items, seen_per_source = [], {}
    for r in picked:
        s = r["source"]
        if seen_per_source.get(s, 0) < MAX_PER_SOURCE:
            out_items.append(r)
            seen_per_source[s] = seen_per_source.get(s, 0) + 1
        if len(out_items) >= TOP_K_ITEMS:
            break

    # Micro‑clústeres
    clusters = build_clusters(E_today, items, novelty)

    # Generar salida
    build_html(today_iso, clusters, out_items if not clusters else [])
    build_index()

    # Actualizar histórico (recorte a 60 días)
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

    print(f"Informe generado. Clústeres: {len(clusters)} · Respaldo de piezas: {len(out_items) if not clusters else 0}")

if __name__ == "__main__":
    main()
