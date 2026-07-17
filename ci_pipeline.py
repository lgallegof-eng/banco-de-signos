# ci_pipeline.py
# Banco de Signos: lee RSS, filtra por recencia, calcula novedad y publica informe HTML.
import os, time, math, datetime as dt
from pathlib import Path
from email.utils import parsedate_to_datetime
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
    # Puedes añadir más aquí
]

TOP_K = 25                 # cuántas piezas mostrar
MAX_PER_SOURCE = 4         # tope por cabecera para diversidad
HISTORY_DAYS = 21          # ventana de histórico para calcular novedad
MAX_AGE_DAYS = 3           # filtrar artículos más antiguos que X días
RECENCY_TAU_HOURS = 72     # “semivida” de recencia (a menor, más peso a lo reciente)

# Directorios controlados por el workflow
PUBLISH_DIR = Path(os.getenv("PUBLISH_DIR", "public")).resolve()
HIST_DIR = Path(os.getenv("HIST_DIR", "public/data")).resolve()
PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

HIST_FILE = HIST_DIR / "history.npz"
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# ------------ Utilidades ------------
def parse_entry_dt(entry):
    # Intenta varios campos de fecha del feed
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
            if not dpub:
                # si no hay fecha fiable, salta la pieza
                continue
            # filtra por recencia
            if dpub < cutoff:
                continue
            title = (e.get("title", "") or "").strip() or "(Sin título)"
            summary = (e.get("summary", "") or "").strip()
            items.append({
                "source": source,
                "title": title,
                "url": link,
                "summary": summary,
                "published_iso": dpub.astimezone(dt.timezone.utc).isoformat(timespec="minutes"),
                "published_dt": dpub,
            })
    # orden preliminar por fecha (más reciente primero)
    items.sort(key=lambda r: r["published_dt"], reverse=True)
    return items

def load_history():
    if HIST_FILE.exists():
        data = np.load(HIST_FILE, allow_pickle=False)
        return data["E"], data["d"]
    return None, None

def save_history(E_hist, d_hist):
    np.savez_compressed(HIST_FILE, E=E_hist, d=d_hist)

def build_html(today, ranked):
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
        "</style></head><body>",
        f"<h1>Banco de Signos — Informe de los últimos {MAX_AGE_DAYS} días</h1>",
        "<p>Prioriza piezas recientes y con alta novedad semántica frente al histórico.</p>",
    ]
    for r in ranked:
        n = r["novelty"]
        age_h = r["age_hours"]
        score = r["score"]
        cls = "hi" if score >= 0.6 else ("med" if score >= 0.35 else "lo")
        parts.append("<div class='card'>")
        parts.append(
            f"<h3><a href='{r['url']}' target='_blank' rel='noopener'>{r['title']}</a> "
            f"<span class='badge {cls}' title='score (novedad x recencia)'>{score:.2f}</span></h3>"
        )
        if r.get("summary"):
            short = r["summary"]
            short = (short[:320] + "…") if len(short) > 340 else short
            parts.append(f"<p>{short}</p>")
        meta = f"{r['source']} · {r.get('published_iso','')} · novedad {n:.2f} · hace {age_h:.0f}h"
        parts.append(f"<div class='meta'>{meta}</div>")
        parts.append("</div>")
    parts.append("<hr><small>RSS públicos (titulares y resúmenes). Solo artículos de los últimos "
                 f"{MAX_AGE_DAYS} días. Score = novedad × factor de recencia.</small>")
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
    # Siempre crea índice (evita 404) aunque no haya items
    if not items:
        build_index()
        print("Sin artículos recientes en las fuentes configuradas.")
        return

    # Texto para embeddings
    texts = [(it["title"] + " || " + (it["summary"] or "")[:1000]).strip() for it in items]

    # Modelo multilingüe
    model = SentenceTransformer(MODEL_NAME)
    E_today = model.encode(texts, batch_size=16, normalize_embeddings=True)
    E_today = np.asarray(E_today).astype(np.float32)

    # Cargar histórico (ventana)
    E_hist, d_hist = load_history()
    cutoff = (now_utc.date() - dt.timedelta(days=HISTORY_DAYS)).toordinal()
    if E_hist is not None and d_hist is not None:
        mask = d_hist >= cutoff
        E_hist, d_hist = E_hist[mask], d_hist[mask]

    # Novedad vs. histórico
    if E_hist is None or E_hist.shape[0] == 0:
        novelty = np.ones(E_today.shape[0], dtype=np.float32)
    else:
        sims = E_today @ E_hist.T  # embeddings normalizados
        novelty = 1.0 - np.max(sims, axis=1)

    # Recencia y score combinado
    ranked = []
    for i, it in enumerate(items):
        age_h = max(0.0, (now_utc - it["published_dt"]).total_seconds() / 3600.0)
        recency_w = math.exp(-age_h / float(RECENCY_TAU_HOURS))
        score = float(novelty[i]) * recency_w
        ranked.append({
            **it,
            "novelty": float(novelty[i]),
            "age_hours": age_h,
            "score": score,
        })

    # Diversidad por fuente y TOP_K
    ranked.sort(key=lambda r: r["score"], reverse=True)
    picked, per_source = [], {}
    for r in ranked:
        s = r["source"]
        if per_source.get(s, 0) < MAX_PER_SOURCE:
            picked.append(r)
            per_source[s] = per_source.get(s, 0) + 1
        if len(picked) >= TOP_K:
            break

    # Generar salida
    build_html(today_iso, picked)
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

    print(f"Informe generado con {len(picked)} piezas recientes.")

if __name__ == "__main__":
    main()
