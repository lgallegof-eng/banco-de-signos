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
