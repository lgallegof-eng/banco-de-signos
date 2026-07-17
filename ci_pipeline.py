# ci_pipeline.py
# Banco de Signos: lee RSS, calcula novedad semántica y publica un informe HTML diario.
import os, datetime as dt
from pathlib import Path
import feedparser
import numpy as np
from sentence_transformers import SentenceTransformer

# -------- Fuentes (puedes ajustar) --------
FEEDS = [
    # ——— España ———
    # El Periódico (portada)
    "https://www.elperiodico.com/es/rss/rss_portada.xml",
    # La Voz de Galicia (portada)
    "https://www.lavozdegalicia.es/rss/index.xml",
    # SModa (El País, WordPress)
    "https://smoda.elpais.com/feed/",
    # Código Nuevo (WordPress)
    "https://www.codigonuevo.com/feed",
    # Reason Why (marketing)
    "https://www.reasonwhy.es/rss",
    # Control Publicidad (WordPress)
    "https://controlpublicidad.com/feed/",
    # El Confidencial (últimas noticias)
    "https://www.elconfidencial.com/rss/ultimas_noticias/",

    # ——— Internacional ———
    # The Atlantic (todos los artículos)
    "https://www.theatlantic.com/feed/all/",
    # The New York Times (Home y Mundo)
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    # Fast Company (general)
    "https://www.fastcompany.com/rss",
    # Highsnobiety (WordPress)
    "https://www.highsnobiety.com/feed/",
    # Financial Times (Home)
    "https://www.ft.com/rss/home",
    # Business Insider (Global y España)
    "https://www.businessinsider.com/rss",
    "https://www.businessinsider.es/rss",
    # Bloomberg (a validar; Bloomberg cambia a menudo sus RSS)
    "https://feeds.bloomberg.com/markets/news.rss",
]

TOP_K = 20
HISTORY_DAYS = 21

# Directorios de publicación e histórico (los controla el workflow)
PUBLISH_DIR = Path(os.getenv("PUBLISH_DIR", "public")).resolve()
HIST_DIR = Path(os.getenv("HIST_DIR", "public/data")).resolve()
PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

HIST_FILE = HIST_DIR / "history.npz"
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"

def fetch_items(feeds):
    items, seen = [], set()
    for url in feeds:
        try:
            d = feedparser.parse(url)
        except Exception:
            continue
        for e in d.entries:
            link = (e.get("link") or "").split("?")[0]
            if not link or link in seen:
                continue
            seen.add(link)
            items.append({
                "source": d.feed.get("title", url),
                "title": (e.get("title", "") or "").strip() or "(Sin título)",
                "url": link,
                "summary": (e.get("summary", "") or "").strip(),
                "published": e.get("published", "") or "",
            })
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
        f"<h1>Banco de Signos — Señales débiles — {today}</h1>",
        "<p>Selección automática de piezas con alta novedad semántica frente al histórico reciente.</p>",
    ]
    for r in ranked:
        n = r["novelty"]
        cls = "hi" if n >= 0.6 else ("med" if n >= 0.35 else "lo")
        parts.append("<div class='card'>")
        parts.append(f"<h3><a href='{r['url']}' target='_blank' rel='noopener'>{r['title']}</a> "
                     f"<span class='badge {cls}'>{n:.2f}</span></h3>")
        if r.get("summary"):
            short = r["summary"]
            short = (short[:320] + "…") if len(short) > 340 else short
            parts.append(f"<p>{short}</p>")
        meta = f"{r['source']} · {r.get('published','')}"
        parts.append(f"<div class='meta'>{meta}</div>")
        parts.append("</div>")
    parts.append("<hr><small>Generado automáticamente con RSS públicos (titulares y resúmenes).</small>")
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
        lines.append("<li>Aún no hay informes. Ejecuta de nuevo el workflow en unos minutos.</li>")
    for f in files:
        ymd = f.replace("weak_signals_","").replace(".html","")
        lines.append(f"<li><a href='{f}'>{ymd}</a></li>")
    lines.append("</ul><p>Últimos 30 días.</p></body></html>")
    idx.write_text("\n".join(lines), encoding="utf-8")

def main():
    today = dt.date.today()
    today_iso = today.isoformat()

    items = fetch_items(FEEDS)
    # Genera siempre index.html para evitar 404 aunque falle la red
    if not items:
        build_index()
        print("No se han obtenido artículos hoy.")
        return

    texts = [(it["title"] + " || " + (it["summary"] or "")[:1000]).strip() for it in items]

    model = SentenceTransformer(MODEL_NAME)
    E_today = model.encode(texts, batch_size=16, normalize_embeddings=True)
    E_today = np.asarray(E_today).astype(np.float32)

    E_hist, d_hist = load_history()
    cutoff = (today - dt.timedelta(days=HISTORY_DAYS)).toordinal()
    if E_hist is not None and d_hist is not None:
        mask = d_hist >= cutoff
        E_hist, d_hist = E_hist[mask], d_hist[mask]

    if E_hist is None or E_hist.shape[0] == 0:
        novelty = np.ones(E_today.shape[0], dtype=np.float32)
    else:
        sims = E_today @ E_hist.T
        novelty = 1.0 - np.max(sims, axis=1)

    ranked = [{**it, "novelty": float(n)} for it, n in zip(items, novelty)]
    ranked.sort(key=lambda r: r["novelty"], reverse=True)
    ranked = ranked[:TOP_K]

    build_html(today_iso, ranked)
    build_index()

    today_ord = today.toordinal()
    d_today = np.full((E_today.shape[0],), today_ord, dtype=np.int32)
    if E_hist is None or d_hist is None:
        E_new, d_new = E_today, d_today
    else:
        E_new = np.vstack([E_hist, E_today])
        d_new = np.concatenate([d_hist, d_today])
        keep = d_new >= (today - dt.timedelta(days=60)).toordinal()
        E_new, d_new = E_new[keep], d_new[keep]
    save_history(E_new, d_new)

    print("Informe generado y índice actualizado.")

if __name__ == "__main__":
    main()
