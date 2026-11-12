import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf

PAIRS = {
    "EURUSD": {"t": "EURUSD=X", "contract": 100000, "jpy": False, "dec": 5},
    "GBPUSD": {"t": "GBPUSD=X", "contract": 100000, "jpy": False, "dec": 5},
    "USDJPY": {"t": "JPY=X", "contract": 100000, "jpy": True, "dec": 3},
    "XAUUSD": {"t": "GC=F", "contract": 100, "jpy": False, "dec": 2},
}
BALANCE = float(os.getenv("BALANCE") or 1000)
RISK_PCT = float(os.getenv("RISK_PCT") or 1)
RR = 2.0
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")
H4 = pd.Timedelta(hours=4)


def load(ticker):
    df = yf.download(ticker, period="60d", interval="1h",
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.columns = ["o", "h", "l", "c"]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(hours=1) <= now]


def atr(df, n=14):
    pc = df.c.shift()
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(n).mean()


def to_h4(df):
    return df.resample("4h").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()


def snd_zones(h4, hist):
    a = atr(h4)
    zones = []
    for i in range(14, len(h4) - 1):
        base, imp, A = h4.iloc[i], h4.iloc[i + 1], a.iloc[i]
        if np.isnan(A) or abs(base.c - base.o) > 0.5 * A:
            continue
        body = imp.c - imp.o
        if abs(body) < 1.5 * A:
            continue
        t_end = h4.index[i + 1] + H4
        after = hist[hist.index >= t_end]
        if body > 0 and imp.c > base.h:
            lo, hi = base.l, max(base.o, base.c)
            if (after.l <= hi).any():
                continue
            zones.append(("demand", lo, hi))
        elif body < 0 and imp.c < base.l:
            lo, hi = min(base.o, base.c), base.h
            if (after.h >= lo).any():
                continue
            zones.append(("supply", lo, hi))
    return zones


def snr_levels(df, tol, k=3):
    pts = []
    for i in range(k, len(df) - k):
        w = df.iloc[i - k:i + k + 1]
        if df.h.iloc[i] == w.h.max():
            pts.append(df.h.iloc[i])
        if df.l.iloc[i] == w.l.min():
            pts.append(df.l.iloc[i])
    pts.sort()
    levels, grp = [], []
    for p in pts:
        if grp and p - grp[0] > tol:
            if len(grp) >= 2:
                levels.append(sum(grp) / len(grp))
            grp = []
        grp.append(p)
    if len(grp) >= 2:
        levels.append(sum(grp) / len(grp))
    return levels


def lot_size(entry, sl, cfg):
    loss = abs(entry - sl) * cfg["contract"]
    if cfg["jpy"]:
        loss /= entry
    risk_usd = BALANCE * RISK_PCT / 100
    lot = max(int(risk_usd / loss * 100) / 100, 0.01)
    return lot, lot * loss, risk_usd


def analyze(name, cfg, h1):
    if len(h1) < 200:
        return []
    cut = h1.index[-1]
    hist = h1.iloc[:-1]
    h4 = to_h4(hist)
    h4 = h4[h4.index + H4 <= cut]
    a4, a1 = atr(h4).iloc[-1], atr(h1).iloc[-1]
    last = h1.iloc[-1]
    levels = snr_levels(h4, 0.5 * a4)
    d = cfg["dec"]
    out = []
    for kind, lo, hi in snd_zones(h4, hist):
        if hi - lo > 1.5 * a4:
            continue
        if kind == "demand" and last.l <= hi and last.c > hi and last.c > last.o:
            side, entry, sl = "BUY", last.c, lo - 0.3 * a1
            tp = entry + RR * (entry - sl)
        elif kind == "supply" and last.h >= lo and last.c < lo and last.c < last.o:
            side, entry, sl = "SELL", last.c, hi + 0.3 * a1
            tp = entry - RR * (sl - entry)
        else:
            continue
        snr = any(lo - 0.5 * a4 <= L <= hi + 0.5 * a4 for L in levels)
        lot, real, want = lot_size(entry, sl, cfg)
        msg = (f"SINYAL {side} {name}\n"
               f"Zona {kind}: {lo:.{d}f} - {hi:.{d}f}"
               f"{' (+SNR)' if snr else ''}\n"
               f"Entry: {entry:.{d}f}\nSL: {sl:.{d}f}\nTP: {tp:.{d}f} (RR 1:{RR:g})\n"
               f"Lot: {lot:.2f} (risiko sekitar ${real:.2f} "
               f"dari target ${want:.2f})\n")
        if real > 1.5 * want:
            msg += "Peringatan: lot minimum 0.01 membuat risiko lebih besar dari target.\n"
        msg += "Cek berita dan spread dulu. Ini alat bantu, bukan jaminan."
        out.append(msg)
    return out


def send(text):
    if not TOKEN or not CHAT:
        print("Secret Telegram belum terbaca")
        return
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT, "text": text}, timeout=20)
    print("Telegram status:", r.status_code)


def main():
    msgs = []
    for name, cfg in PAIRS.items():
        try:
            msgs += analyze(name, cfg, load(cfg["t"]))
        except Exception as e:
            print(name, "gagal:", type(e).__name__)
    for m in msgs:
        send(m)
    if os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch" and not msgs:
        send("Bot aktif. Belum ada sinyal saat ini.")
    print("Selesai, sinyal:", len(msgs))


if __name__ == "__main__":
    main()
