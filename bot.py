import os
import json
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

PAIRS = {
    "XAUUSD": {"t": "GC=F", "contract": 100, "dec": 2},
    "BTCUSD": {"t": "BTC-USD", "contract": 1, "dec": 2},
}
BALANCE = float(os.getenv("BALANCE") or 1000)
RISK_PCT = float(os.getenv("RISK_PCT") or 1)
RR = 2.0
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")
STATE_FILE = "state.json"
M5 = pd.Timedelta(minutes=5)
M30 = pd.Timedelta(minutes=30)
H1 = pd.Timedelta(hours=1)
TAP_WINDOW = 48  # candle M5 (4 jam)
MIN_Q = 55  # kualitas area minimal (0-100) untuk peringatan dan sinyal
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_BEFORE = pd.Timedelta(minutes=30)
NEWS_AFTER = pd.Timedelta(minutes=30)
CAL_REFRESH = pd.Timedelta(hours=6)
CAL_RETRY = pd.Timedelta(minutes=30)


def load(ticker):
    err = None
    for _ in range(2):
        try:
            df = yf.download(ticker, period="59d", interval="5m",
                             progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Volume" not in df.columns:
                df["Volume"] = 0.0
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            df["Volume"] = df["Volume"].fillna(0)
            if df.empty:
                raise ValueError("data kosong")
            break
        except Exception as e:
            err = e
            time.sleep(3)
    else:
        raise err
    df.columns = ["o", "h", "l", "c", "v"]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + M5 <= now]


def rs(df, rule):
    agg = {"o": "first", "h": "max", "l": "min", "c": "last"}
    if "v" in df.columns:
        agg["v"] = "sum"
    return df.resample(rule).agg(agg).dropna(subset=["o", "h", "l", "c"])


def atr(df, n=14):
    pc = df.c.shift()
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(n).mean()


def structure(df, k):
    """MSB: candle menutup melewati swing terakhir.
    Searah tren = BOS, berlawanan = CHOCH. Return (tren, daftar event)."""
    hi, lo, cl = df.h.values, df.l.values, df.c.values
    sh = sl = None
    trend = 0
    events = []
    for t in range(len(df)):
        p = t - k
        if p >= k:
            if hi[p] == hi[p - k:t + 1].max():
                sh = [hi[p], False]
            if lo[p] == lo[p - k:t + 1].min():
                sl = [lo[p], False]
        if sh and not sh[1] and cl[t] > sh[0]:
            kind = "CHOCH" if trend == -1 else "BOS"
            events.append((t, kind, 1, sh[0], sl[0] if sl else None))
            sh[1] = True
            trend = 1
        elif sl and not sl[1] and cl[t] < sl[0]:
            kind = "CHOCH" if trend == 1 else "BOS"
            events.append((t, kind, -1, sl[0], sh[0] if sh else None))
            sl[1] = True
            trend = -1
    return trend, events


def find_fvgs(df, min_size):
    h, l = df.h.values, df.l.values
    out = []
    for i in range(2, len(df)):
        if l[i] > h[i - 2] and l[i] - h[i - 2] >= min_size:
            out.append((1, h[i - 2], l[i], i))
        elif h[i] < l[i - 2] and l[i - 2] - h[i] >= min_size:
            out.append((-1, h[i], l[i - 2], i))
    return out


def zone_status(m5, lo, hi, dirn, t_start):
    after = m5[m5.index >= t_start]
    if after.empty:
        return "ok", None
    if dirn == 1:
        if (after.c < lo).any():
            return "dead", None
        touched = after[after.l <= hi]
    else:
        if (after.c > hi).any():
            return "dead", None
        touched = after[after.h >= lo]
    if touched.empty:
        return "ok", None
    t = touched.index[0]
    if t < m5.index[-1] - TAP_WINDOW * M5:
        return "dead", None
    return "ok", t


def score_poi(body_atr, fvg, bos, fresh, aligned):
    sc = (min(30, 15 * body_atr) + (20 if fvg else 0) + (15 if bos else 0)
          + (20 if fresh else 8) + (15 if aligned else 0))
    return int(min(100, sc))


def build_pois(h1, m5, a1, rl, rh):
    out = []
    a = atr(h1)

    def aligned(mid, d):
        if rh <= rl:
            return False
        p = (mid - rl) / (rh - rl)
        return p <= 0.5 if d == 1 else p >= 0.5

    for i in range(max(14, len(h1) - 400), len(h1) - 1):
        base, imp, A = h1.iloc[i], h1.iloc[i + 1], a.iloc[i]
        if np.isnan(A) or abs(base.c - base.o) > 0.5 * A:
            continue
        body = imp.c - imp.o
        if abs(body) < 1.5 * A:
            continue
        if body > 0 and imp.c > base.h:
            lo, hi, d, kind = base.l, max(base.o, base.c), 1, "demand"
        elif body < 0 and imp.c < base.l:
            lo, hi, d, kind = min(base.o, base.c), base.h, -1, "supply"
        else:
            continue
        if hi - lo > 2 * a1:
            continue
        st, tt = zone_status(m5, lo, hi, d, h1.index[i + 1] + H1)
        if st != "ok":
            continue
        fvg = i + 2 < len(h1) and (
            (d == 1 and h1.l.iloc[i + 2] > h1.h.iloc[i]) or
            (d == -1 and h1.h.iloc[i + 2] < h1.l.iloc[i]))
        prior = h1.iloc[max(0, i - 10):i + 1]
        bos = imp.c > prior.h.max() if d == 1 else imp.c < prior.l.min()
        sc = score_poi(abs(body) / A, fvg, bos, tt is None, aligned((lo + hi) / 2, d))
        out.append({"kind": kind, "dir": d, "lo": lo, "hi": hi,
                    "touched": tt, "score": sc})
    tail = h1.iloc[-400:]
    for d, lo, hi, i in find_fvgs(tail, 0.2 * a1):
        if hi - lo > 2 * a1:
            continue
        st, tt = zone_status(m5, lo, hi, d, tail.index[i] + H1)
        if st != "ok":
            continue
        mc = tail.iloc[i - 1]
        prior = tail.iloc[max(0, i - 11):i - 1]
        bos = False
        if len(prior):
            bos = mc.c > prior.h.max() if d == 1 else mc.c < prior.l.min()
        sc = score_poi(abs(mc.c - mc.o) / a1, True, bos, tt is None,
                       aligned((lo + hi) / 2, d))
        out.append({"kind": "FVG naik" if d == 1 else "FVG turun",
                    "dir": d, "lo": lo, "hi": hi, "touched": tt, "score": sc})
    return out


def eq_levels(df, tol):
    """Equal highs/lows yang belum tertembus. Return (daftar EQH, daftar EQL)."""
    hi, lo = df.h.values, df.l.values
    sh, sl = [], []
    for p in range(3, len(df) - 3):
        if hi[p] == hi[p - 3:p + 4].max():
            sh.append((p, hi[p]))
        if lo[p] == lo[p - 3:p + 4].min():
            sl.append((p, lo[p]))
    eqh, eql = [], []
    pts = sh[-8:]
    for x in range(len(pts)):
        for y in range(x + 1, len(pts)):
            if abs(pts[x][1] - pts[y][1]) <= tol:
                lv = max(pts[x][1], pts[y][1])
                if hi[pts[y][0] + 1:].max(initial=-np.inf) <= lv + tol:
                    eqh.append(lv)
    pts = sl[-8:]
    for x in range(len(pts)):
        for y in range(x + 1, len(pts)):
            if abs(pts[x][1] - pts[y][1]) <= tol:
                lv = min(pts[x][1], pts[y][1])
                if lo[pts[y][0] + 1:].min(initial=np.inf) >= lv - tol:
                    eql.append(lv)
    return eqh, eql


def rvol(h1):
    if "v" not in h1.columns or len(h1) < 30:
        return None
    base = h1.v.iloc[-49:-1].median()
    if not base or base <= 0 or h1.v.iloc[-1] <= 0:
        return None
    return h1.v.iloc[-1] / base


def rvol_label(rv):
    if rv is None:
        return "n/a"
    if rv < 0.7:
        return f"SEPI ({rv:.1f}x)"
    if rv > 1.3:
        return f"RAMAI ({rv:.1f}x)"
    return f"NORMAL ({rv:.1f}x)"


def killzone(now):
    h = now.hour + now.minute / 60
    if 7 <= h < 10:
        return "London"
    if 12 <= h < 15:
        return "New York"
    return None


def recent_event(m5, d, kind=None, back=2):
    _, ev = structure(m5, 2)
    n = len(m5) - 1
    if not ev:
        return None
    e = ev[-1]
    if e[0] < n - back or e[2] != d or (kind and e[1] != kind):
        return None
    return e


def recent_fvg(m5, d, a5, t_min=None):
    tail = m5.iloc[-12:]
    best = None
    for fd, lo, hi, i in find_fvgs(tail, 0.1 * a5):
        if fd != d:
            continue
        if t_min is not None and tail.index[i] < t_min:
            continue
        after = tail.iloc[i + 1:]
        if d == 1 and (after.c < lo).any():
            continue
        if d == -1 and (after.c > hi).any():
            continue
        best = (lo, hi)
    return best


def m5_trigger(m5, d, poi, a5, a1):
    t_tap = poi["touched"]
    if t_tap is None:
        return None
    e = recent_event(m5, d)
    if not e:
        return None
    t_ev = m5.index[e[0]]
    if t_ev < t_tap:
        return None
    seg = m5[m5.index >= t_tap - M5]
    fv = recent_fvg(m5, d, a5, max(t_tap, t_ev))
    if not fv:
        return None
    flo, fhi = fv
    entry = (flo + fhi) / 2
    price = m5.c.iloc[-1]
    if d == 1:
        sl = seg.l.min() - 0.2 * a5
        if not (sl < entry < price):
            return None
        tp = entry + RR * (entry - sl)
    else:
        sl = seg.h.max() + 0.2 * a5
        if not (price < entry < sl):
            return None
        tp = entry - RR * (sl - entry)
    risk = abs(entry - sl)
    if risk <= 0 or risk > 1.5 * a1:
        return None
    return {"entry": entry, "sl": sl, "tp": tp, "fvg": (flo, fhi),
            "kind": e[1], "level": e[3], "t": t_ev}


def m5_continue(m5, d, a5, a1):
    """Mode lanjutan: BOS searah bias di M5 + FVG, entry limit di retest FVG."""
    e = recent_event(m5, d, "BOS")
    if not e or e[4] is None:
        return None
    t_ev = m5.index[e[0]]
    fv = recent_fvg(m5, d, a5, t_ev)
    if not fv:
        return None
    flo, fhi = fv
    entry = (flo + fhi) / 2
    price = m5.c.iloc[-1]
    if d == 1:
        sl = e[4] - 0.2 * a5
        if not (sl < entry < price):
            return None
        tp = entry + RR * (entry - sl)
    else:
        sl = e[4] + 0.2 * a5
        if not (price < entry < sl):
            return None
        tp = entry - RR * (sl - entry)
    risk = abs(entry - sl)
    if risk <= 0 or risk > 1.5 * a1:
        return None
    return {"entry": entry, "sl": sl, "tp": tp, "fvg": (flo, fhi),
            "kind": "BOS", "level": e[3], "t": t_ev}


def lot_size(entry, sl, cfg):
    loss = abs(entry - sl) * cfg["contract"]
    risk_usd = BALANCE * RISK_PCT / 100
    lot = max(int(risk_usd / loss * 100) / 100, 0.01)
    return lot, lot * loss, risk_usd


def arah(t):
    return {1: "naik", -1: "turun"}.get(t, "belum jelas")


def wib(ts):
    return ts.tz_convert("Asia/Jakarta").strftime("%H:%M")


def analyze(name, cfg, m5):
    """Return (sinyal, peringatan, status). sinyal = [(kunci, pesan)], peringatan = [dict]"""
    if len(m5) < 1500:
        return [], [], f"{name}: data kurang"
    now = pd.Timestamp.now(tz="UTC")
    end = m5.index[-1] + M5
    h1 = rs(m5, "1h")
    h1 = h1[h1.index + H1 <= end]
    m30 = rs(m5, "30min")
    m30 = m30[m30.index + M30 <= end]
    a1, a5 = atr(h1).iloc[-1], atr(m5).iloc[-1]
    price = m5.c.iloc[-1]
    d = cfg["dec"]
    fx = lambda x: "-" if x is None else f"{x:.{d}f}"
    t1, ev1 = structure(h1, 3)
    t30, _ = structure(m30, 2)
    rh, rl = h1.h.iloc[-120:].max(), h1.l.iloc[-120:].min()
    pct = (price - rl) / (rh - rl) * 100 if rh > rl else 50.0
    zone = "PREMIUM" if pct > 50 else "DISCOUNT"
    pd_ok = (t1 == -1 and pct >= 50) or (t1 == 1 and pct <= 50)
    ideal = "premium" if t1 == -1 else "discount"
    tol = 0.15 * a1
    eqh_l, eql_l = eq_levels(h1, tol)
    eqh = min([x for x in eqh_l if x > price], default=None)
    eql = max([x for x in eql_l if x < price], default=None)
    h0, l0 = eq_levels(h1.iloc[:-6], tol)
    sweep = None
    for c in m30.iloc[-8:].itertuples():
        for lv in h0:
            if c.h > lv and c.c < lv:
                sweep = f"EQH {lv:.{d}f} tersapu"
        for lv in l0:
            if c.l < lv and c.c > lv:
                sweep = f"EQL {lv:.{d}f} tersapu"
    allp = build_pois(h1, m5, a1, rl, rh)
    n_up = sum(1 for p in allp if p["dir"] == 1)
    n_dn = len(allp) - n_up
    avg_q = sum(p["score"] for p in allp) / len(allp) if allp else 0
    kz = killzone(now)
    ctx = f"Sesi: {kz or 'luar killzone'} | RVOL: {rvol_label(rvol(h1))}\n"

    last_ev = f" ({ev1[-1][1]})" if ev1 else ""
    lines = [f"{name}: harga {price:.{d}f}",
             f"  Bias H1: {arah(t1)}{last_ev} | M30: {arah(t30)}",
             f"  Zona: {zone} ({pct:.0f}% dari range 5 hari)"]
    if ev1:
        t_e, _, dr, _, opp = ev1[-1]
        seg = h1.iloc[t_e:]
        if dr == -1:
            lines.append(f"  Strong high {fx(opp)} | Weak low {fx(seg.l.min())}")
        else:
            lines.append(f"  Strong low {fx(opp)} | Weak high {fx(seg.h.max())}")
    lines.append(f"  EQH {fx(eqh)} | EQL {fx(eql)}")
    lines.append(f"  Sweep: {sweep or '-'}")
    lines.append(f"  Area aktif: naik {n_up}, turun {n_dn}, kualitas rata-rata {avg_q:.0f}%")
    lines.append(f"  RVOL: {rvol_label(rvol(h1))} | Sesi: {kz or 'luar killzone'}")

    sigs, alerts = [], []
    if t1 == 0:
        lines.append("  Setup: TUNGGU (bias belum jelas)")
        return sigs, alerts, "\n".join(lines)

    near = []
    for p in allp:
        if p["dir"] != t1 or p["score"] < MIN_Q:
            continue
        if p["touched"] is None:
            dist = price - p["hi"] if t1 == 1 else p["lo"] - price
            if dist <= 0:
                continue
        else:
            dist = 0
        near.append((dist, p))
    near.sort(key=lambda x: x[0])
    if near:
        p = near[0][1]
        lines.append(f"  Area terdekat: {p['kind']} {p['lo']:.{d}f} - {p['hi']:.{d}f} "
                     f"(kualitas {p['score']}%)")
    else:
        lines.append(f"  Area terdekat: tidak ada (kualitas minimal {MIN_Q}%)")
    today = now.strftime("%Y-%m-%d")
    for dist, p in near:
        if p["touched"] is None:
            if dist <= 1.5 * a1:
                mid = (p["lo"] + p["hi"]) / 2
                alerts.append({
                    "name": name, "kind": p["kind"], "mid": mid,
                    "tol": 0.5 * a1, "day": today,
                    "key": f"A|{name}|{p['kind']}|{mid:.2f}|{today}",
                    "msg": (f"MENDEKATI AREA {p['kind'].upper()} {name}\n"
                            f"Harga {price:.{d}f}, area {p['lo']:.{d}f} - {p['hi']:.{d}f} "
                            f"(kualitas {p['score']}%)\n"
                            f"Bias H1 {arah(t1)} | Zona {zone} {pct:.0f}%\n"
                            f"Belum ada sinyal entri.")})
            continue
        if t30 != t1:
            continue
        tr = m5_trigger(m5, t1, p, a5, a1)
        if not tr:
            continue
        side = "BUY LIMIT" if t1 == 1 else "SELL LIMIT"
        lot, real, want = lot_size(tr["entry"], tr["sl"], cfg)
        key = f"S|{name}|{side}|{tr['t'].isoformat()}"
        msg = (f"SINYAL {side} {name} (SMC)\n"
               f"Bias H1: {arah(t1)} | M30: {arah(t30)} | Zona: {zone} {pct:.0f}%\n"
               f"Area H1: {p['kind']} {p['lo']:.{d}f} - {p['hi']:.{d}f} "
               f"(kualitas {p['score']}%, tersentuh {wib(p['touched'])} WIB)\n"
               f"M5: {tr['kind']} {arah(t1)}, tutup melewati {tr['level']:.{d}f}\n"
               f"Entry limit: {tr['entry']:.{d}f} "
               f"(FVG M5 {tr['fvg'][0]:.{d}f} - {tr['fvg'][1]:.{d}f})\n"
               f"SL: {tr['sl']:.{d}f}\nTP: {tr['tp']:.{d}f} (RR 1:{RR:g})\n"
               f"Lot: {lot:.2f} (risiko sekitar ${real:.2f} dari target ${want:.2f})\n"
               + ctx)
        if real > 1.5 * want:
            msg += "Peringatan: lot minimum 0.01 membuat risiko lebih besar dari target.\n"
        msg += ("Order limit bisa tidak terisi. Batalkan kalau harga sudah menembus SL "
                "sebelum terisi. Cek ukuran kontrak broker, berita, dan spread. "
                "Alat bantu, bukan jaminan.")
        sigs.append((key, msg))
    blocked = False
    if not sigs and t30 == t1:
        cn = m5_continue(m5, t1, a5, a1)
        if cn and not pd_ok:
            blocked = True
        elif cn:
            side = "BUY LIMIT" if t1 == 1 else "SELL LIMIT"
            lot, real, want = lot_size(cn["entry"], cn["sl"], cfg)
            hour = cn["t"].floor("1h").isoformat()
            key = f"C|{name}|{side}|{hour}"
            msg = (f"SINYAL {side} {name} (lanjutan BOS)\n"
                   f"Bias H1: {arah(t1)} | M30: {arah(t30)} | Zona: {zone} {pct:.0f}%\n"
                   f"M5: BOS {arah(t1)}, tutup melewati {cn['level']:.{d}f}\n"
                   f"Entry limit: {cn['entry']:.{d}f} "
                   f"(FVG M5 {cn['fvg'][0]:.{d}f} - {cn['fvg'][1]:.{d}f})\n"
                   f"SL: {cn['sl']:.{d}f} (di balik swing M5 terakhir)\n"
                   f"TP: {cn['tp']:.{d}f} (RR 1:{RR:g})\n"
                   f"Lot: {lot:.2f} (risiko sekitar ${real:.2f} dari target ${want:.2f})\n"
                   + ctx)
            if real > 1.5 * want:
                msg += "Peringatan: lot minimum 0.01 membuat risiko lebih besar dari target.\n"
            msg += ("Mode lanjutan tidak berada di area H1, jadi lebih berisiko. "
                    "Pertimbangkan lot lebih kecil. Order limit bisa tidak terisi. "
                    "Alat bantu, bukan jaminan.")
            sigs.append((key, msg))
    if sigs:
        setup = "SINYAL"
    elif alerts:
        setup = "MENDEKATI AREA"
    elif blocked:
        setup = (f"TUNGGU (ada BOS M5 searah bias, tapi harga di zona {zone.lower()}; "
                 f"tunggu pullback ke {ideal})")
    elif not pd_ok:
        setup = f"TUNGGU (harga di {zone.lower()}, idealnya pullback ke {ideal} dulu)"
    else:
        setup = "TUNGGU"
    lines.append(f"  Setup: {setup}")
    return sigs, alerts, "\n".join(lines)


def fetch_calendar():
    r = requests.get(CAL_URL, timeout=20,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    out = []
    for e in r.json():
        cur = e.get("country") or e.get("currency")
        if cur != "USD" or e.get("impact") != "High":
            continue
        t = pd.Timestamp(e["date"])
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        out.append({"t": t.tz_convert("UTC").isoformat(),
                    "title": str(e.get("title", "")),
                    "forecast": str(e.get("forecast", "")),
                    "previous": str(e.get("previous", ""))})
    return out


def get_calendar(state, now):
    """Return (events, catatan, berubah). Kalender dicache 6 jam di state."""
    cal = state.get("cal")
    try:
        if cal and now - pd.Timestamp(cal["at"]) < CAL_REFRESH:
            return cal["events"], "", False
    except Exception:
        cal = None
    try:
        if now - pd.Timestamp(state.get("cal_try", "2000-01-01T00:00:00+00:00")) < CAL_RETRY:
            raise RuntimeError("tunggu")
    except RuntimeError:
        return (cal["events"], "", False) if cal else ([], "kalender berita belum termuat", False)
    state["cal_try"] = now.isoformat()
    try:
        ev = fetch_calendar()
        state["cal"] = {"at": now.isoformat(), "events": ev}
        return ev, "", True
    except Exception as e:
        print("kalender gagal:", type(e).__name__)
        if cal:
            return cal["events"], "kalender berita belum diperbarui", True
        return [], "kalender berita gagal dimuat, filter berita nonaktif", True


def news_block(events, now):
    for e in events:
        t = pd.Timestamp(e["t"])
        if t - NEWS_BEFORE <= now <= t + NEWS_AFTER:
            return e
    return None


def next_news(events, now, within=pd.Timedelta(hours=3)):
    best = None
    for e in events:
        t = pd.Timestamp(e["t"])
        if now < t <= now + within and (best is None or t < pd.Timestamp(best["t"])):
            best = e
    return best


def news_text(e):
    t = pd.Timestamp(e["t"])
    extra = []
    if e.get("forecast"):
        extra.append(f"perkiraan {e['forecast']}")
    if e.get("previous"):
        extra.append(f"sebelumnya {e['previous']}")
    return f"{e['title']} (USD) jam {wib(t)} WIB" + (
        f" ({', '.join(extra)})" if extra else "")


def send(text):
    if not TOKEN or not CHAT:
        print("Secret Telegram belum terbaca")
        return False
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT, "text": text}, timeout=20)
    print("Telegram status:", r.status_code)
    return r.status_code == 200


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def alert_sent(sent, al):
    pre = f"A|{al['name']}|{al['kind']}|"
    for k in sent:
        if k.startswith(pre) and k.endswith("|" + al["day"]):
            try:
                if abs(float(k.split("|")[3]) - al["mid"]) <= al["tol"]:
                    return True
            except ValueError:
                pass
    return False


def main():
    manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"
    now = pd.Timestamp.now(tz="UTC")
    state = load_state()
    changed = state is None
    state = state or {}
    sent = list(state.get("sent", []))
    events, cal_note, cal_changed = get_calendar(state, now)
    changed = changed or cal_changed
    blk = news_block(events, now)
    nxt = next_news(events, now)

    for e in events:
        t = pd.Timestamp(e["t"])
        if pd.Timedelta(0) < t - now <= pd.Timedelta(minutes=35):
            key = f"N|{e['title']}|{e['t']}"
            if key not in sent and send(
                    "BERITA SEBENTAR LAGI\n" + news_text(e) + "\n"
                    "Sinyal entri dijeda dari 30 menit sebelum sampai "
                    "30 menit sesudah rilis, karena harga sering "
                    "bergerak liar dan menipu."):
                sent.append(key)
                changed = True

    stats, errs = [], []
    n_sig = n_al = n_skip = 0
    for name, cfg in PAIRS.items():
        try:
            sigs, alerts, st = analyze(name, cfg, load(cfg["t"]))
            stats.append(st)
            if blk:
                n_skip += len(sigs) + len(alerts)
                continue
            for key, msg in sigs:
                if key in sent:
                    continue
                if nxt:
                    msg += "\nBerita terdekat: " + news_text(nxt)
                if send(msg):
                    sent.append(key)
                    changed = True
                    n_sig += 1
            for al in alerts:
                if alert_sent(sent, al):
                    continue
                if send(al["msg"]):
                    sent.append(al["key"])
                    changed = True
                    n_al += 1
        except Exception as e:
            errs.append(f"{name} gagal: {type(e).__name__}")
            print(name, "gagal:", type(e).__name__, str(e)[:150])

    dkey = f"D|{now.strftime('%Y-%m-%d')}"
    if manual or (now.hour == 0 and dkey not in sent):
        lines = stats + errs
        if blk:
            lines.append("Jeda berita: " + news_text(blk))
        elif nxt:
            lines.append("Berita terdekat: " + news_text(nxt))
        elif not cal_note:
            lines.append("Tidak ada berita besar USD dalam 3 jam ke depan")
        if cal_note:
            lines.append(cal_note)
        if send("Bot aktif (analisis detail SMC + filter berita).\n" + "\n".join(lines)) and not manual:
            sent.append(dkey)
            changed = True
    if changed:
        state["sent"] = sent[-300:]
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    print("Selesai, sinyal:", n_sig, "peringatan:", n_al,
          "ditahan berita:", n_skip, "error:", len(errs))


if __name__ == "__main__":
    main()
