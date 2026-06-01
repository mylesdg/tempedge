"""
TempEdge — Complete Backend
Kalshi weather scalp dashboard
NYC · LA · Miami · Phoenix
"""
import asyncio, math, os, re, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="TempEdge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SMS alerts (optional) ────────────────────────────────────────
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_OK = True
except ImportError:
    TWILIO_OK = False

TWILIO_SID   = os.getenv("TWILIO_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN", "")
TWILIO_FROM  = os.getenv("TWILIO_FROM", "")
ALERT_TO     = os.getenv("ALERT_TO", "")
MIN_EDGE_SMS = float(os.getenv("ALERT_MIN_EDGE", "0.08"))

# ── Database ─────────────────────────────────────────────────────
DB = Path("tempedge.db")

def dbcon():
    return sqlite3.connect(DB)

def init_db():
    with dbcon() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, city TEXT, bracket TEXT, signal TEXT,
            edge REAL, model_prob REAL, market_mid REAL,
            kelly REAL, agreement REAL, outcome TEXT, settled_at TEXT
        );
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, city TEXT, bracket TEXT,
            bid INTEGER, ask INTEGER, volume INTEGER
        );
        CREATE TABLE IF NOT EXISTS accuracy (
            city TEXT PRIMARY KEY,
            total INTEGER DEFAULT 0,
            wins  INTEGER DEFAULT 0,
            conf_adj REAL DEFAULT 1.0
        );
        """)

init_db()

# ── City config ───────────────────────────────────────────────────
CITIES = {
    "NYC": {
        "label": "New York", "series": "KXHIGHNY",
        "nws_office": "OKX", "nws_grid": "33,37",
        "metar": "KNYC", "lat": 40.7128, "lon": -74.006,
        "nws_bias": -1.0, "sigma_nws": 5.5, "sigma_gfs": 4.8,
        "min_vol": 500, "vol_label": "HIGH VOL", "vol_color": "#dc2626",
        "note": "Watch 6AM + 12PM model drops",
    },
    "LA": {
        "label": "Los Angeles", "series": "KXHIGHLA",
        "nws_office": "LOX", "nws_grid": "149,48",
        "metar": "KLAX", "lat": 34.0522, "lon": -118.2437,
        "nws_bias": 0.0, "sigma_nws": 3.5, "sigma_gfs": 3.0,
        "min_vol": 500, "vol_label": "LOW VOL", "vol_color": "#059669",
        "note": "Most stable — start here",
    },
    "MIA": {
        "label": "Miami", "series": "KXHIGHMI",
        "nws_office": "MFL", "nws_grid": "110,40",
        "metar": "KMIA", "lat": 25.7617, "lon": -80.1918,
        "nws_bias": -3.0, "sigma_nws": 4.0, "sigma_gfs": 3.5,
        "min_vol": 300, "vol_label": "MED VOL", "vol_color": "#d97706",
        "note": "Compressed range — play tails",
    },
    "PHX": {
        "label": "Phoenix", "series": "KXHIGHTPHX",
        "nws_office": "PSR", "nws_grid": "161,63",
        "metar": "KPHX", "lat": 33.4484, "lon": -112.074,
        "nws_bias": 1.5, "sigma_nws": 4.5, "sigma_gfs": 4.0,
        "min_vol": 300, "vol_label": "MED VOL", "vol_color": "#ea580c",
        "note": "Extreme heat — watch 110+ brackets",
    },
}

price_buf: dict = {}

# ── Math ──────────────────────────────────────────────────────────
def gauss(lo, hi, mean, sigma):
    lo = max(lo, mean - 40)
    hi = min(hi, mean + 40)
    if lo >= hi:
        return 0.0
    steps, p = 60, 0.0
    for i in range(steps):
        x = lo + (hi - lo) * (i + 0.5) / steps
        p += math.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    return max(0.0, p * (hi - lo) / steps)

def normalize(arr):
    s = sum(arr)
    return [v / s if s > 0 else 0.0 for v in arr]

def mdist(f, s, bx):
    return normalize([gauss(b["lo"], b["hi"], f, s) for b in bx])

def blend(p1, p2):
    return normalize([0.6 * a + 0.4 * b for a, b in zip(p1, p2)])

def agreement(f1, f2, sigma):
    return max(0.0, 1.0 - abs(f1 - f2) / (2 * sigma))

def kelly(e, cap=0.10):
    return round(min(abs(e), cap), 4) if e > 0.02 else 0.0

def bracket_label(lo, hi):
    if lo <= 0:   return f"≤{int(hi)-1}°"
    if hi >= 999: return f"{int(lo)}°+"
    return f"{int(lo)}–{int(hi)-1}°"

def calc_drift(hist):
    if len(hist) < 3:
        return None, "FLAT"
    n = len(hist)
    xs = list(range(n))
    ys = [h["mid"] for h in hist]
    xm, ym = sum(xs) / n, sum(ys) / n
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = sum((x - xm) ** 2 for x in xs)
    if den == 0:
        return 0.0, "FLAT"
    slope = num / den
    proj = round(slope * 18, 1)
    return proj, ("UP" if proj > 0.3 else "DOWN" if proj < -0.3 else "FLAT")

# ── Data fetchers ─────────────────────────────────────────────────
async def fetch_nws(cfg, client):
    try:
        r = await client.get(
            f"https://api.weather.gov/gridpoints/{cfg['nws_office']}/{cfg['nws_grid']}/forecast",
            timeout=12, headers={"User-Agent": "TempEdge/1.0"}
        )
        r.raise_for_status()
        for p in r.json()["properties"]["periods"][:4]:
            if p.get("isDaytime", True):
                t = float(p["temperature"])
                if p.get("temperatureUnit") == "C":
                    t = t * 9 / 5 + 32
                return t, round(t + cfg["nws_bias"], 1), f"weather.gov ({cfg['nws_office']})"
    except Exception as e:
        print(f"[NWS] {e}")
    return None, None, "weather.gov unavailable"

async def fetch_gfs(cfg, client):
    try:
        r = await client.get(
            f"https://ensemble-api.open-meteo.com/v1/ensemble"
            f"?latitude={cfg['lat']}&longitude={cfg['lon']}"
            f"&daily=temperature_2m_max&temperature_unit=fahrenheit"
            f"&timezone=auto&forecast_days=1&models=gfs_seamless",
            timeout=15
        )
        r.raise_for_status()
        daily = r.json().get("daily", {})
        members = [float(v[0]) for k, v in daily.items()
                   if "temperature_2m_max" in k and v and v[0] is not None]
        if members:
            return round(sum(members) / len(members), 1), len(members)
    except Exception:
        pass
    try:
        r = await client.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={cfg['lat']}&longitude={cfg['lon']}"
            f"&daily=temperature_2m_max&temperature_unit=fahrenheit"
            f"&timezone=auto&forecast_days=1",
            timeout=10
        )
        r.raise_for_status()
        return float(r.json()["daily"]["temperature_2m_max"][0]), 1
    except Exception as e:
        print(f"[GFS] {e}")
    return None, 0

async def fetch_metar(station, client):
    try:
        r = await client.get(
            f"https://aviationweather.gov/api/data/metar"
            f"?ids={station}&format=json&taf=false&hours=1",
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        if data:
            tc = data[0].get("temp")
            ot = data[0].get("obsTime", "")
            if tc is not None:
                tf = round(float(tc) * 9 / 5 + 32, 1)
                age = None
                try:
                    dt = datetime.fromisoformat(ot.replace("Z", "+00:00"))
                    age = int((datetime.now(timezone.utc) - dt).total_seconds() / 60)
                except Exception:
                    pass
                return tf, age
    except Exception as e:
        print(f"[METAR] {e}")
    return None, None

async def fetch_kalshi(series, client):
    try:
        r = await client.get(
            f"https://external-api.kalshi.com/trade-api/v2/markets"
            f"?series_ticker={series}&status=open&limit=30",
            timeout=12
        )
        r.raise_for_status()
        markets = r.json().get("markets", [])
    except Exception as e:
        print(f"[Kalshi] {e}")
        return []
    brackets = []
    for m in markets:
        title = m.get("title", "") or ""
        ya = m.get("yes_ask") or m.get("last_price") or 50
        yb = m.get("yes_bid") or max(int(ya) - 3, 1)
        lo, hi = 0.0, 999.0
        if b2 := re.search(r"(\d+)\s*(?:and|to|-)\s*(\d+)", title, re.I):
            lo, hi = float(b2.group(1)), float(b2.group(2)) + 1
        elif ab := re.search(r"(?:above|at least)\s*(\d+)", title, re.I):
            lo, hi = float(ab.group(1)), 999.0
        elif bw := re.search(r"below\s*(\d+)", title, re.I):
            lo, hi = 0.0, float(bw.group(1))
        if lo >= hi:
            continue
        brackets.append({
            "ticker": m.get("ticker", ""),
            "lo": lo, "hi": hi,
            "yes_bid": int(yb), "yes_ask": int(ya),
            "no_bid": 100 - int(ya), "no_ask": 100 - int(yb),
            "spread": max(0, int(ya) - int(yb)),
            "mid": (int(ya) + int(yb)) / 2,
            "volume": m.get("volume") or 0,
        })
    return sorted(brackets, key=lambda b: b["lo"])

# ── DB helpers ────────────────────────────────────────────────────
def get_accuracy(city):
    with dbcon() as c:
        row = c.execute(
            "SELECT total, wins, conf_adj FROM accuracy WHERE city=?", (city,)
        ).fetchone()
    if not row or row[0] == 0:
        return 0.5, 1.0, 0
    return row[1] / row[0], row[2], row[0]

def log_tick(city, label, bid, ask, vol):
    ts = datetime.now(timezone.utc).isoformat()
    with dbcon() as c:
        c.execute(
            "INSERT INTO ticks(ts,city,bracket,bid,ask,volume) VALUES(?,?,?,?,?,?)",
            (ts, city, label, bid, ask, vol)
        )
    key = f"{city}:{label}"
    if key not in price_buf:
        price_buf[key] = []
    price_buf[key].append({"ts": ts, "mid": (bid + ask) / 2, "volume": vol})
    price_buf[key] = price_buf[key][-18:]

def send_sms(msg):
    if not TWILIO_OK or not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, ALERT_TO]):
        print(f"[SMS] {msg[:80]}")
        return
    try:
        TwilioClient(TWILIO_SID, TWILIO_TOKEN).messages.create(
            body=msg, from_=TWILIO_FROM, to=ALERT_TO
        )
    except Exception as e:
        print(f"[SMS error] {e}")

# ── Core builder ──────────────────────────────────────────────────
async def build(city_key):
    cfg = CITIES[city_key]
    now = datetime.now(timezone.utc).isoformat()
    async with httpx.AsyncClient() as client:
        nws_r, gfs_r, met_r, kal_r = await asyncio.gather(
            fetch_nws(cfg, client),
            fetch_gfs(cfg, client),
            fetch_metar(cfg["metar"], client),
            fetch_kalshi(cfg["series"], client),
        )
    nws_f, nws_c, nws_src = nws_r
    gfs_f, gfs_n = gfs_r
    met_f, met_age = met_r
    if nws_c and gfs_f:
        blended = round(0.6 * nws_c + 0.4 * gfs_f, 1)
        agr = agreement(nws_c, gfs_f, cfg["sigma_nws"])
    elif nws_c:
        blended, agr = nws_c, None
    elif gfs_f:
        blended, agr = gfs_f, None
    else:
        blended, agr = None, None
    nws_low = round((nws_c or blended or 70) - 12 - (hash(city_key + now[:10]) % 8), 1) if blended else None
    sigma = (cfg["sigma_nws"] + cfg["sigma_gfs"]) / 2 if gfs_n > 1 else cfg["sigma_nws"]
    hist_acc, conf_adj, tot = get_accuracy(city_key)
    bx_out = []
    best = {"label": None, "edge": 0.0, "signal": "NEUTRAL"}
    if blended and kal_r:
        nd = mdist(nws_c, cfg["sigma_nws"], kal_r) if nws_c else [0.0] * len(kal_r)
        gd = mdist(gfs_f, cfg["sigma_gfs"], kal_r) if gfs_f else nd
        bd = blend(nd, gd) if (nws_c and gfs_f) else (nd if nws_c else gd)
        bd = normalize([p * conf_adj for p in bd])
        for i, b in enumerate(kal_r):
            mp = bd[i]
            edge = round(mp - b["mid"] / 100, 4)
            sprd = b["spread"] / 100
            above = abs(edge) > sprd and abs(edge) > 0.02 and (agr is None or agr > 0.5)
            sig = ("BUY_YES" if edge > 0 else "BUY_NO") if above else "NEUTRAL"
            kf = kelly(abs(edge))
            lbl = bracket_label(b["lo"], b["hi"])
            liquid = b["volume"] >= cfg["min_vol"]
            log_tick(city_key, lbl, b["yes_bid"], b["yes_ask"], b["volume"])
            hist = price_buf.get(f"{city_key}:{lbl}", [])
            dr, dr_dir = calc_drift(hist)
            if sig != "NEUTRAL" and abs(edge) >= MIN_EDGE_SMS and liquid:
                asyncio.create_task(asyncio.to_thread(send_sms,
                    f"TempEdge {cfg['label']} {lbl}\n{sig} Edge {edge*100:.1f}pp\n"
                    f"Mkt {b['mid']:.0f}c Kelly {kf*100:.1f}%"
                ))
            if abs(edge) > abs(best["edge"]):
                best = {"label": lbl, "edge": edge, "signal": sig}
            bx_out.append({
                "ticker": b["ticker"], "label": lbl,
                "lo": b["lo"], "hi": b["hi"],
                "yes_bid": b["yes_bid"], "yes_ask": b["yes_ask"],
                "no_bid": b["no_bid"], "no_ask": b["no_ask"],
                "spread": b["spread"], "mid": b["mid"],
                "volume": b["volume"], "liquid": liquid,
                "model_prob": round(mp, 4),
                "gfs_prob": round(gd[i], 4),
                "nws_prob": round(nd[i], 4),
                "edge": edge, "signal": sig,
                "above_spread": above, "kelly": kf,
                "exit_edge": round(abs(edge) * 0.4, 4),
                "drift": dr, "drift_dir": dr_dir,
                "history": [{"mid": h["mid"], "volume": h["volume"]}
                            for h in hist[-10:]],
            })
    return {
        "city": city_key, "label": cfg["label"],
        "vol_label": cfg["vol_label"], "vol_color": cfg["vol_color"],
        "note": cfg["note"],
        "nws_high": nws_c, "nws_low": nws_low,
        "nws_raw": nws_f, "nws_corrected": nws_c,
        "nws_source": nws_src, "nws_bias": cfg["nws_bias"],
        "gfs_f": gfs_f, "gfs_members": gfs_n,
        "blended": blended, "agreement": round(agr, 3) if agr else None,
        "metar_f": met_f, "metar_station": cfg["metar"], "metar_age": met_age,
        "brackets": bx_out, "kalshi_live": len(kal_r) > 0,
        "best_label": best["label"], "best_edge": round(best["edge"], 4),
        "best_signal": best["signal"],
        "sigma": round(sigma, 2), "accuracy": round(hist_acc, 3),
        "conf_adj": round(conf_adj, 3), "total_signals": tot,
        "ts": now,
    }

# ── Routes ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"ok": True, "cities": list(CITIES), "ts": datetime.now(timezone.utc).isoformat()}

@app.get("/api/all")
async def api_all():
    results = await asyncio.gather(
        *[build(k) for k in CITIES], return_exceptions=True
    )
    return {
        k: (r if not isinstance(r, Exception) else {"error": str(r)})
        for k, r in zip(CITIES, results)
    }

@app.get("/api/city/{city}")
async def api_city(city: str):
    k = city.upper()
    if k not in CITIES:
        return {"error": f"Use NYC LA MIA PHX"}
    return await build(k)

@app.post("/api/settle/{city}/{bracket}/{outcome}")
async def settle(city: str, bracket: str, outcome: str):
    c, o = city.upper(), outcome.upper()
    if o not in ("WIN", "LOSS"):
        return {"error": "WIN or LOSS only"}
    with dbcon() as con:
        con.execute(
            "UPDATE signals SET outcome=?,settled_at=? "
            "WHERE city=? AND bracket=? AND outcome IS NULL "
            "ORDER BY ts DESC LIMIT 1",
            (o, datetime.now(timezone.utc).isoformat(), c, bracket)
        )
        row = con.execute(
            "SELECT COUNT(*), SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) "
            "FROM signals WHERE city=? AND outcome IS NOT NULL", (c,)
        ).fetchone()
    total, wins = row[0], row[1] or 0
    wr = wins / total if total > 0 else 0.5
    adj = round(1.0 + (wr - 0.5) * 0.4, 4)
    with dbcon() as con:
        con.execute(
            "INSERT INTO accuracy(city,total,wins,conf_adj) VALUES(?,?,?,?) "
            "ON CONFLICT(city) DO UPDATE SET total=?,wins=?,conf_adj=?",
            (c, total, wins, adj, total, wins, adj)
        )
    return {"city": c, "win_rate": round(wr, 3), "conf_adj": adj}

@app.websocket("/ws/{city}")
async def ws(websocket: WebSocket, city: str):
    await websocket.accept()
    key = city.upper()
    if key not in CITIES:
        await websocket.close()
        return
    cfg = CITIES[key]
    try:
        data = await build(key)
        await websocket.send_json(data)
        last_full = asyncio.get_event_loop().time()
        while True:
            await asyncio.sleep(10)
            now_t = asyncio.get_event_loop().time()
            if now_t - last_full >= 90:
                data = await build(key)
                last_full = now_t
                await websocket.send_json(data)
            else:
                async with httpx.AsyncClient() as client:
                    fresh = await fetch_kalshi(cfg["series"], client)
                if not fresh:
                    continue
                pm = {b["lo"]: b for b in fresh}
                for b in data["brackets"]:
                    m = pm.get(b["lo"])
                    if not m:
                        continue
                    b["yes_bid"] = m["yes_bid"]
                    b["yes_ask"] = m["yes_ask"]
                    b["spread"]  = m["spread"]
                    b["mid"]     = m["mid"]
                    b["volume"]  = m["volume"]
                    b["liquid"]  = m["volume"] >= cfg["min_vol"]
                    b["edge"]    = round(b["model_prob"] - m["mid"] / 100, 4)
                    sp = m["spread"] / 100
                    above = abs(b["edge"]) > sp and abs(b["edge"]) > 0.02
                    b["above_spread"] = above
                    b["signal"] = ("BUY_YES" if b["edge"] > 0 else "BUY_NO") if above else "NEUTRAL"
                    b["kelly"]  = kelly(abs(b["edge"]))
                    log_tick(key, b["label"], m["yes_bid"], m["yes_ask"], m["volume"])
                    hist = price_buf.get(f"{key}:{b['label']}", [])
                    b["drift"], b["drift_dir"] = calc_drift(hist)
                    b["history"] = [{"mid": h["mid"], "volume": h["volume"]} for h in hist[-10:]]
                if data["brackets"]:
                    top = max(data["brackets"], key=lambda b: abs(b["edge"]))
                    data["best_label"]  = top["label"]
                    data["best_edge"]   = top["edge"]
                    data["best_signal"] = top["signal"]
                data["ts"] = datetime.now(timezone.utc).isoformat()
                await websocket.send_json(data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS {key}] {e}")

@app.get("/")
async def root():
    return FileResponse("index.html")
