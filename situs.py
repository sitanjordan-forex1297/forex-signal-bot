import os
import json
import numpy as np
import pandas as pd
import bot
import teknikal
import riwayat

OUT = "data.json"
H4 = pd.Timedelta(hours=4)
STYLES = {
    "Scalp": {"rule": "30min", "tf": bot.M30, "k": 2, "rr": (1.5, 3.0), "tap": 2, "hz": "30 menit sampai 3 jam"},
    "Daytrade": {"rule": "1h", "tf": bot.H1, "k": 3, "rr": (2.0, 4.0), "tap": 4, "hz": "3 sampai 24 jam"},
    "Swing": {"rule": "4h", "tf": H4, "k": 3, "rr": (3.0, 5.0), "tap": 24, "hz": "2 sampai 7 hari"},
}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) or np.isinf(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def zstat(m5, lo, hi, d, t_start, tap_h):
    after = m5[m5.index >= t_start]
    if after.empty:
        return "ok", False
    if d == 1:
        if (after.c < lo).any():
            return "dead", False
        touched = after[after.l <= hi]
    else:
        if (after.c > hi).any():
            return "dead", False
        touched = after[after.h >= lo]
    if touched.empty:
        return "ok", False
    if touched.index[0] < m5.index[-1] - pd.Timedelta(hours=tap_h):
        return "dead", False
    return "ok", True


def pois_tf(df, m5, a, rl, rh, tf, tap_h):
    out = []
    atr_s = bot.atr(df)

    def aligned(mid, d):
        if rh <= rl:
            return False
        p = (mid - rl) / (rh - rl)
        return p <= 0.5 if d == 1 else p >= 0.5

    for i in range(max(14, len(df) - 300), len(df) - 1):
        base, imp, A = df.iloc[i], df.iloc[i + 1], atr_s.iloc[i]
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
        if hi - lo > 2 * a:
            continue
        st, tt = zstat(m5, lo, hi, d, df.index[i + 1] + tf, tap_h)
        if st != "ok":
            continue
        fvg = i + 2 < len(df) and ((d == 1 and df.l.iloc[i + 2] > df.h.iloc[i]) or
                                   (d == -1 and df.h.iloc[i + 2] < df.l.iloc[i]))
        prior = df.iloc[max(0, i - 10):i + 1]
        bos = imp.c > prior.h.max() if d == 1 else imp.c < prior.l.min()
        sc = bot.score_poi(abs(body) / A, fvg, bos, not tt, aligned((lo + hi) / 2, d))
        out.append({"kind": kind, "dir": d, "lo": lo, "hi": hi, "touched": tt, "score": sc})
    tail = df.iloc[-300:]
    for d, lo, hi, i in bot.find_fvgs(tail, 0.2 * a):
        if hi - lo > 2 * a:
            continue
        st, tt = zstat(m5, lo, hi, d, tail.index[i] + tf, tap_h)
        if st != "ok":
            continue
        mc = tail.iloc[i - 1]
        prior = tail.iloc[max(0, i - 11):i - 1]
        bos = bool(len(prior)) and (mc.c > prior.h.max() if d == 1 else mc.c < prior.l.min())
        sc = bot.score_poi(abs(mc.c - mc.o) / a, True, bos, not tt, aligned((lo + hi) / 2, d))
        out.append({"kind": "FVG naik" if d == 1 else "FVG turun", "dir": d,
                    "lo": lo, "hi": hi, "touched": tt, "score": sc})
    return out


def make_plan(name, cfg, m5, style, ctx):
    S = STYLES[style]
    end = m5.index[-1] + bot.M5
    df = bot.rs(m5, S["rule"])
    df = df[df.index + S["tf"] <= end]
    if len(df) < 60:
        return {"status": "DATA KURANG", "alasan": ["Data timeframe ini belum cukup"]}
    a = bot.atr(df).iloc[-1]
    price = m5.c.iloc[-1]
    t, ev = bot.structure(df, S["k"])
    seg = df.iloc[-120:]
    rh, rl = seg.h.max(), seg.l.min()
    pct = (price - rl) / (rh - rl) * 100 if rh > rl else 50.0
    if t == 0:
        return {"status": "TIDAK ADA", "alasan": ["Bias belum jelas di timeframe ini"], "horizon": S["hz"]}
    pois = [p for p in pois_tf(df, m5, a, rl, rh, S["tf"], S["tap"])
            if p["dir"] == t and p["score"] >= bot.MIN_Q]
    cands = []
    for p in pois:
        if p["touched"]:
            dist = 0.0
        else:
            dist = price - p["hi"] if t == 1 else p["lo"] - price
            if dist <= 0:
                continue
        cands.append((dist, p))
    if not cands:
        return {"status": "TIDAK ADA", "horizon": S["hz"],
                "alasan": [f"Bias {bot.arah(t)}, tapi belum ada area dengan kualitas minimal {bot.MIN_Q}%"]}
    dist, p = min(cands, key=lambda x: (x[0], -x[1]["score"]))
    buf = 0.3 * a
    if t == 1:
        entry, sl = p["hi"], p["lo"] - buf
    else:
        entry, sl = p["lo"], p["hi"] + buf
    risk = abs(entry - sl)
    r1, r2 = S["rr"]
    tp1, tp2 = entry + t * r1 * risk, entry + t * r2 * risk
    status = ("SIAP" if dist == 0 else "MENDEKATI" if dist <= 1.5 * a
              else "MENUNGGU" if dist <= 6 * a else "JAUH")
    pd_ok = (t == -1 and pct >= 50) or (t == 1 and pct <= 50)
    zone = "PREMIUM" if pct > 50 else "DISCOUNT"
    last_ev = f" ({ev[-1][1]})" if ev else ""
    why = [f"Bias {S['rule']} {bot.arah(t)}{last_ev}",
           f"Area {p['kind']} kualitas {p['score']}% ({'baru tersentuh' if p['touched'] else 'masih segar'})",
           f"Zona {zone} {pct:.0f}% ({'sesuai' if pd_ok else 'kurang sesuai'} dengan bias)",
           f"Teknikal Per Jam: {ctx['tech_h1']}"]
    warn = []
    if not pd_ok:
        warn.append(f"Harga di zona {zone.lower()}; idealnya pullback ke {'premium' if t == -1 else 'discount'} dulu")
    if (t == -1 and "Beli" in ctx["tech_h1"]) or (t == 1 and "Jual" in ctx["tech_h1"]):
        warn.append("Ringkasan teknikal Per Jam berlawanan dengan arah rencana")
    if ctx.get("news"):
        warn.append("Berita besar USD dalam 3 jam: " + ctx["news"])
    if "SEPI" in ctx.get("rvol", ""):
        warn.append("Volume sepi, gerakan bisa palsu")
    q = p["score"]
    return {"status": status, "side": "BUY LIMIT" if t == 1 else "SELL LIMIT", "dir": t,
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr1": r1, "rr2": r2,
            "quality": q, "qlabel": "TINGGI" if q >= 75 else "SEDANG",
            "area": {"kind": p["kind"], "lo": p["lo"], "hi": p["hi"]},
            "dist": dist, "dist_atr": dist / a if a else None, "horizon": S["hz"],
            "loss_per_lot": risk * cfg["contract"], "alasan": why, "peringatan": warn}


def pair_block(name, cfg, m5, news_txt):
    now = pd.Timestamp.now(tz="UTC")
    end = m5.index[-1] + bot.M5
    h1 = bot.rs(m5, "1h")
    h1 = h1[h1.index + bot.H1 <= end]
    m30 = bot.rs(m5, "30min")
    m30 = m30[m30.index + bot.M30 <= end]
    price = m5.c.iloc[-1]
    t1, ev1 = bot.structure(h1, 3)
    t30, _ = bot.structure(m30, 2)
    h4 = bot.rs(m5, "4h")
    h4 = h4[h4.index + H4 <= end]
    t4, _ = bot.structure(h4, 3)
    a1 = bot.atr(h1).iloc[-1]
    rh, rl = h1.h.iloc[-120:].max(), h1.l.iloc[-120:].min()
    pct = (price - rl) / (rh - rl) * 100 if rh > rl else 50.0
    eqh_l, eql_l = bot.eq_levels(h1, 0.15 * a1)
    eqh = min([x for x in eqh_l if x > price], default=None)
    eql = max([x for x in eql_l if x < price], default=None)
    sw = None
    if ev1:
        te, _, dr, _, opp = ev1[-1]
        seg = h1.iloc[te:]
        sw = (["Strong high", opp, "Weak low", seg.l.min()] if dr == -1
              else ["Strong low", opp, "Weak high", seg.h.max()])
    res = {lab: teknikal.analyze_tf(teknikal.closed(m5, rule, dl)) for lab, rule, dl in teknikal.TFS}
    tech = {}
    for lab, r in res.items():
        tech[lab] = {"all": r["all"], "ma_l": r["ma_l"], "ind_l": r["ind_l"], "mc": r["mc"], "ic": r["ic"]}
    hr = res["Per Jam"]
    tech["Per Jam"].update({"ma": hr["ma"], "ind": hr["ind"], "piv": hr["piv"]})
    ctx = {"tech_h1": hr["all"], "news": news_txt, "rvol": bot.rvol_label(bot.rvol(h1))}
    allp = bot.build_pois(h1, m5, a1, rl, rh)
    areas = []
    for p in allp:
        dist = 0 if p["touched"] else (price - p["hi"] if p["dir"] == 1 else p["lo"] - price)
        if dist < 0:
            continue
        areas.append((dist, p))
    areas.sort(key=lambda x: x[0])
    areas = [{"kind": p["kind"], "dir": p["dir"], "lo": p["lo"], "hi": p["hi"],
              "score": p["score"], "touched": bool(p["touched"])} for _, p in areas[:6]]
    kz = bot.killzone(now)
    ch = [[c.o, c.h, c.l, c.c] for c in h1.iloc[-48:].itertuples()]
    return {"price": price, "dec": cfg["dec"], "contract": cfg["contract"], "chart": ch,
            "bias": {"h1": bot.arah(t1), "h1_event": ev1[-1][1] if ev1 else None,
                     "m30": bot.arah(t30), "h4": bot.arah(t4)},
            "zone": {"name": "PREMIUM" if pct > 50 else "DISCOUNT", "pct": pct},
            "levels": {"sw": sw, "eqh": eqh, "eql": eql}, "areas": areas, "tech": tech,
            "rvol": ctx["rvol"], "session": kz,
            "plans": {s: make_plan(name, cfg, m5, s, ctx) for s in STYLES}}


def export(manual):
    now = pd.Timestamp.now(tz="UTC")
    if not manual and os.path.exists(OUT):
        try:
            old = pd.Timestamp(json.load(open(OUT))["updated"])
            if now - old < pd.Timedelta(minutes=14):
                return False
        except Exception:
            pass
    state = bot.load_state() or {}
    events = (state.get("cal") or {}).get("events", [])
    nxt = bot.next_news(events, now)
    news_txt = bot.news_text(nxt) if nxt else ""
    blk = bot.news_block(events, now)
    pairs = {}
    for name, cfg in bot.PAIRS.items():
        m5 = riwayat.DATA.get(cfg["t"])
        if m5 is None or len(m5) < 1500:
            continue
        try:
            pairs[name] = pair_block(name, cfg, m5, news_txt)
        except Exception as e:
            print("situs gagal:", name, type(e).__name__, str(e)[:120])
    hist = state.get("hist", [])
    items = sorted(hist, key=lambda s: s["t0"], reverse=True)[:40]
    news = [e for e in events if pd.Timestamp(e["t"]) > now - pd.Timedelta(hours=3)]
    data = {"updated": now.isoformat(), "balance": bot.BALANCE, "risk_pct": bot.RISK_PCT,
            "pairs": pairs, "news": news, "block": blk["title"] if blk else None,
            "history": {"stats": riwayat.stats(hist), "items": items}}
    with open(OUT, "w") as f:
        json.dump(clean(data), f, separators=(",", ":"))
    return True


if __name__ == "__main__":
    manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"
    bot.main()
    riwayat.finish(manual)
    try:
        export(manual)
    except Exception as e:
        print("export gagal:", type(e).__name__, str(e)[:150])
    for m in teknikal.DETAILS:
        bot.send(m)
