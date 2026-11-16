import os
import re
import json
import pandas as pd
import bot
import teknikal

DATA = {}
NEW = []
FINAL = ("TP", "SL", "habis waktu", "kedaluwarsa", "terlewat")
PEND_MAX = pd.Timedelta(hours=6)
RUN_MAX = pd.Timedelta(hours=72)

_orig_load = bot.load
_orig_send = bot.send


def load_cached(t):
    df = _orig_load(t)
    DATA[t] = df
    return df


def send_track(text):
    ok = _orig_send(text)
    if ok and text.startswith("SINYAL "):
        try:
            s = parse_signal(text)
            if s:
                NEW.append(s)
        except Exception as e:
            print("catat sinyal gagal:", type(e).__name__)
    return ok


bot.load = load_cached
bot.send = send_track


def parse_signal(text):
    m = re.match(r"SINYAL (BUY|SELL) LIMIT (\w+) \((.+?)\)", text)
    e = re.search(r"Entry limit: ([\d.]+)", text)
    s = re.search(r"\nSL: ([\d.]+)", text)
    t = re.search(r"\nTP: ([\d.]+)", text)
    if not (m and e and s and t):
        return None
    pair = m.group(2)
    cfg = bot.PAIRS.get(pair)
    df = DATA.get(cfg["t"]) if cfg else None
    if df is None:
        return None
    t0 = (df.index[-1] + bot.M5).isoformat()
    mode = "lanjutan BOS" if "lanjutan" in m.group(3) else "retest area"
    return {"id": f"{pair}|{m.group(1)}|{t0}|{mode}", "pair": pair,
            "side": m.group(1), "mode": mode, "entry": float(e.group(1)),
            "sl": float(s.group(1)), "tp": float(t.group(1)), "t0": t0,
            "status": "menunggu", "t_fill": None, "r": None, "notified": ""}


def evaluate(sig, m5):
    """Hitung ulang status dari awal memakai candle M5 setelah sinyal dikirim.
    Aturan konservatif: SL diprioritaskan bila TP dan SL ada di candle yang sama."""
    buy = sig["side"] == "BUY"
    e, sl, tp = sig["entry"], sig["sl"], sig["tp"]
    risk = abs(e - sl)
    t0 = pd.Timestamp(sig["t0"])
    after = m5[m5.index >= t0]
    status, t_fill, r = "menunggu", None, None
    for ts, c in after.iterrows():
        if t_fill is None:
            if (buy and c.h >= tp and c.l > e) or (not buy and c.l <= tp and c.h < e):
                status = "terlewat"
                break
            hit = c.l <= e if buy else c.h >= e
            if hit:
                t_fill = ts
                status = "berjalan"
                if (buy and c.l <= sl) or (not buy and c.h >= sl):
                    status, r = "SL", -1.0
                    break
                continue
            if ts - t0 > PEND_MAX:
                status = "kedaluwarsa"
                break
        else:
            if (buy and c.l <= sl) or (not buy and c.h >= sl):
                status, r = "SL", -1.0
                break
            if (buy and c.h >= tp) or (not buy and c.l <= tp):
                status = "TP"
                r = abs(tp - e) / risk if risk else None
                break
            if ts - t_fill > RUN_MAX:
                status = "habis waktu"
                r = ((c.c - e) if buy else (e - c.c)) / risk if risk else None
                break
    sig["status"] = status
    sig["t_fill"] = t_fill.isoformat() if t_fill is not None else None
    sig["r"] = r
    return sig


def stats(hist):
    cnt = lambda st: sum(1 for s in hist if s["status"] == st)
    tp, sl = cnt("TP"), cnt("SL")
    rs = [s["r"] for s in hist if s["status"] in ("TP", "SL", "habis waktu") and s["r"] is not None]
    return {"n": len(hist), "tp": tp, "sl": sl, "ht": cnt("habis waktu"),
            "wait": cnt("menunggu"), "run": cnt("berjalan"),
            "skip": cnt("kedaluwarsa") + cnt("terlewat"),
            "win": tp / (tp + sl) * 100 if tp + sl else None,
            "tot": sum(rs), "avg": sum(rs) / len(rs) if rs else None, "m": len(rs)}


def line(label, hist):
    s = stats(hist)
    if s["tp"] + s["sl"] == 0:
        return f"  {label}: belum ada hasil"
    return f"  {label}: TP {s['tp']}, SL {s['sl']}, win rate {s['win']:.0f}%"


def summary_text(hist):
    if not hist:
        return ("RIWAYAT SINYAL\nBelum ada sinyal tercatat. Win rate baru bisa dihitung "
                "setelah ada sinyal yang selesai (kena TP atau SL).")
    s = stats(hist)
    L = ["RIWAYAT SINYAL (dihitung otomatis)",
         f"Total {s['n']} sinyal | menunggu {s['wait']} | berjalan {s['run']}",
         f"TP {s['tp']} | SL {s['sl']} | habis waktu {s['ht']} | tidak terisi {s['skip']}"]
    if s["win"] is not None:
        L.append(f"Win rate (TP dibanding TP+SL): {s['win']:.0f}% ({s['tp']} dari {s['tp'] + s['sl']})")
    else:
        L.append("Win rate: belum ada sinyal yang selesai")
    if s["avg"] is not None:
        L.append(f"Total {s['tot']:+.1f}R | rata-rata {s['avg']:+.2f}R per trade ({s['m']} trade)")
    for p in bot.PAIRS:
        L.append(line(p, [x for x in hist if x["pair"] == p]))
    for md, nm in (("retest area", "Retest area"), ("lanjutan BOS", "Lanjutan BOS")):
        L.append(line(nm, [x for x in hist if x["mode"] == md]))
    done = s["tp"] + s["sl"]
    if done < 30:
        L.append(f"Sampel masih kecil ({done} hasil). Win rate belum bisa dipercaya sebelum puluhan hasil.")
    L.append("Simulasi: limit dianggap terisi saat harga menyentuh entry, tanpa spread dan slippage.")
    return "\n".join(L)


def result_text(s):
    head = f"HASIL SINYAL {s['pair']} {s['side']} ({s['mode']})\nEntry {s['entry']} | SL {s['sl']} | TP {s['tp']}\n"
    st = s["status"]
    if st == "berjalan":
        return head + "Harga menyentuh entry (limit terisi dalam simulasi)."
    if st == "TP":
        return head + f"Take profit tercapai ({s['r']:+.1f}R)."
    if st == "SL":
        return head + "Stop loss tersentuh (-1.0R)."
    if st == "habis waktu":
        return head + f"Ditutup setelah 72 jam tanpa TP atau SL ({s['r']:+.1f}R)."
    if st == "terlewat":
        return head + "Harga langsung ke TP tanpa menyentuh entry. Order limit tidak terisi."
    return head + "Order limit tidak terisi dalam 6 jam, sinyal dianggap kedaluwarsa."


def finish(manual):
    now = pd.Timestamp.now(tz="UTC")
    state = bot.load_state() or {}
    hist = state.get("hist", [])
    ids = {s["id"] for s in hist}
    changed = False
    for s in NEW:
        if s["id"] not in ids:
            hist.append(s)
            changed = True
    for s in hist:
        if s["status"] in FINAL:
            continue
        cfg = bot.PAIRS.get(s["pair"])
        df = DATA.get(cfg["t"]) if cfg else None
        if df is None:
            continue
        old = s["status"]
        evaluate(s, df)
        if s["status"] != old:
            changed = True
        if s["status"] not in ("menunggu", s.get("notified")):
            if bot.send(result_text(s)):
                s["notified"] = s["status"]
                changed = True
    day = now.strftime("%Y-%m-%d")
    if manual or (now.hour == 0 and state.get("hist_day") != day):
        if bot.send(summary_text(hist)) and not manual:
            state["hist_day"] = day
            changed = True
    openn = [s for s in hist if s["status"] not in FINAL]
    done = sorted([s for s in hist if s["status"] in FINAL], key=lambda s: s["t0"])[-150:]
    state["hist"] = done + openn
    if changed or manual:
        with open(bot.STATE_FILE, "w") as f:
            json.dump(state, f)


if __name__ == "__main__":
    bot.main()
    finish(os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch")
    for m in teknikal.DETAILS:
        bot.send(m)
