import os
import numpy as np
import pandas as pd
import bot

TFS = [("30 Mnt", "30min", pd.Timedelta(minutes=30)),
       ("Per Jam", "1h", pd.Timedelta(hours=1)),
       ("4 Jam", "4h", pd.Timedelta(hours=4))]
MA_N = [5, 10, 20, 50, 100, 200]
DETAILS = []


def closed(m5, rule, delta):
    end = m5.index[-1] + bot.M5
    df = bot.rs(m5, rule)
    return df[df.index + delta <= end]


def sma(x, n):
    return x.rolling(n).mean()


def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def wilder(x, n):
    return x.ewm(alpha=1.0 / n, adjust=False).mean()


def isnan(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)


def sg_sign(v):
    if isnan(v):
        return "Netral"
    return "Beli" if v > 0 else "Jual" if v < 0 else "Netral"


def sg_band(v, lo, hi):
    """Di atas hi = jenuh beli (Jual), di bawah lo = jenuh jual (Beli),
    di antaranya: di atas 50 Beli, di bawah 50 Jual."""
    if isnan(v):
        return "Netral"
    if v > hi:
        return "Jual"
    if v < lo:
        return "Beli"
    return "Beli" if v > 50 else "Jual" if v < 50 else "Netral"


def indicators(df):
    """Return daftar (nama, nilai, sinyal). Sinyal: Beli, Jual, Netral, atau info volatilitas."""
    d = df.iloc[-600:]
    h, l, c = d.h, d.l, d.c
    pc = c.shift()
    out = []

    delta = c.diff()
    ag, al = wilder(delta.clip(lower=0), 14), wilder(-delta.clip(upper=0), 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + ag / al)
    out.append(("RSI(14)", rsi.iloc[-1], "Beli" if rsi.iloc[-1] > 50 else "Jual" if rsi.iloc[-1] < 50 else "Netral"))

    hh9, ll9 = h.rolling(9).max(), l.rolling(9).min()
    k = (100 * (c - ll9) / (hh9 - ll9)).rolling(6).mean()
    out.append(("STOCH(9,6)", k.iloc[-1], sg_band(k.iloc[-1], 20, 80)))

    mn, mx = rsi.rolling(14).min(), rsi.rolling(14).max()
    sr = 100 * (rsi - mn) / (mx - mn)
    out.append(("STOCHRSI(14)", sr.iloc[-1], sg_band(sr.iloc[-1], 20, 80)))

    macd = ema(c, 12) - ema(c, 26)
    out.append(("MACD(12,26)", macd.iloc[-1], sg_sign(macd.iloc[-1])))

    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_w = wilder(tr, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi, mdi = 100 * wilder(pdm, 14) / atr_w, 100 * wilder(mdm, 14) / atr_w
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    adx = wilder(dx.fillna(0), 14)
    a = adx.iloc[-1]
    s = "Netral" if a <= 25 else ("Beli" if pdi.iloc[-1] > mdi.iloc[-1] else "Jual")
    out.append(("ADX(14)", a, s))

    hh14, ll14 = h.rolling(14).max(), l.rolling(14).min()
    wr = -100 * (hh14 - c) / (hh14 - ll14)
    v = wr.iloc[-1]
    out.append(("Williams %R", v, "Netral" if isnan(v) else "Jual" if v > -20 else "Beli" if v < -80 else "Netral"))

    tp = (h + l + c) / 3
    md = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma(tp, 14)) / (0.015 * md)
    out.append(("CCI(14)", cci.iloc[-1], sg_sign(cci.iloc[-1])))

    atr14 = atr_w.iloc[-1]
    base = atr_w.rolling(50).mean().iloc[-1]
    ratio = atr14 / base if base and not isnan(base) else 1.0
    vol = "Volatilitas Kurang" if ratio < 0.8 else "Volatilitas Tinggi" if ratio > 1.2 else "Volatilitas Sedang"
    out.append(("ATR(14)", atr14, vol))

    hl = c - (sma(h, 14) + sma(l, 14)) / 2
    out.append(("Highs/Lows(14)", hl.iloc[-1], sg_sign(hl.iloc[-1])))

    lowmin = pd.concat([l, pc], axis=1).min(axis=1)
    highmax = pd.concat([h, pc], axis=1).max(axis=1)
    bp, trr = c - lowmin, highmax - lowmin
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = lambda n: bp.rolling(n).sum() / trr.rolling(n).sum()
        uo = 100 * (4 * avg(7) + 2 * avg(14) + avg(28)) / 7
    v = uo.iloc[-1]
    out.append(("Ultimate Oscillator", v, "Netral" if isnan(v) else "Jual" if v > 70 else "Beli" if v < 30 else "Netral"))

    roc = (c / c.shift(12) - 1) * 100
    out.append(("ROC", roc.iloc[-1], sg_sign(roc.iloc[-1])))

    bb = (h + l) / 2 - ema(c, 13)
    out.append(("Bull/Bear Power(13)", bb.iloc[-1], sg_sign(bb.iloc[-1])))
    return out


def mas(df):
    c = df.c
    price = c.iloc[-1]
    rows = []
    for n in MA_N:
        sv = sma(c, n).iloc[-1] if len(c) >= n else np.nan
        ev = ema(c, n).iloc[-1] if len(c) >= n else np.nan
        sg = lambda m: None if isnan(m) else ("Beli" if price > m else "Jual")
        rows.append((f"MA{n}", sv, sg(sv), ev, sg(ev)))
    return rows


def score(buy, sell, neu):
    t = buy + sell + neu
    return 0.0 if t == 0 else (buy - sell) / t


def label(sc):
    if sc >= 0.45:
        return "Sangat Beli"
    if sc >= 0.1:
        return "Beli"
    if sc <= -0.45:
        return "Sangat Jual"
    if sc <= -0.1:
        return "Jual"
    return "Netral"


def pivots(row):
    H, L, C, O = row.h, row.l, row.c, row.o
    R = H - L
    P = (H + L + C) / 3
    out = {"Klasik": (P, [L - 2 * (H - P), P - R, 2 * P - H], [2 * P - L, P + R, H + 2 * (P - L)]),
           "Fibonacci": (P, [P - R, P - 0.618 * R, P - 0.382 * R], [P + 0.382 * R, P + 0.618 * R, P + R]),
           "Camarilla": (P, [C - R * 1.1 / 4, C - R * 1.1 / 6, C - R * 1.1 / 12],
                         [C + R * 1.1 / 12, C + R * 1.1 / 6, C + R * 1.1 / 4])}
    pw = (H + L + 2 * C) / 4
    out["Woodie"] = (pw, [L - 2 * (H - pw), pw - R, 2 * pw - H], [2 * pw - L, pw + R, H + 2 * (pw - L)])
    x = H + 2 * L + C if C < O else 2 * H + L + C if C > O else H + L + 2 * C
    out["DeMark"] = (x / 4, [None, None, x / 2 - H], [x / 2 - L, None, None])
    return out


def analyze_tf(df):
    mrows = mas(df)
    irows = indicators(df)
    mb = sum(1 for r in mrows for s in (r[2], r[4]) if s == "Beli")
    ms = sum(1 for r in mrows for s in (r[2], r[4]) if s == "Jual")
    ib = sum(1 for r in irows if r[2] == "Beli")
    is_ = sum(1 for r in irows if r[2] == "Jual")
    inn = sum(1 for r in irows if r[2] == "Netral")
    sm_, si = score(mb, ms, 0), score(ib, is_, inn)
    return {"ma": mrows, "ind": irows, "mc": (mb, ms), "ic": (ib, is_, inn),
            "ma_l": label(sm_), "ind_l": label(si), "all": label((sm_ + si) / 2),
            "piv": pivots(df.iloc[-1])}


def fv(v, d):
    return "-" if isnan(v) else f"{v:.{d}f}"


def compact(name, cfg, res):
    d = cfg["dec"]
    h = res["Per Jam"]
    g = {n: v for n, _, v in h["ind"]}
    val = {n: v for n, v, _ in h["ind"]}
    atr_lab = g["ATR(14)"]
    P, S, R = h["piv"]["Klasik"]
    return (f"  Teknikal: 30 Mnt {res['30 Mnt']['all']} | Per Jam {h['all']} | 4 Jam {res['4 Jam']['all']}\n"
            f"  Per Jam: MA {h['ma_l']} (beli {h['mc'][0]}, jual {h['mc'][1]}) | "
            f"Indikator {h['ind_l']} (beli {h['ic'][0]}, jual {h['ic'][1]}, netral {h['ic'][2]})\n"
            f"  RSI {fv(val['RSI(14)'], 1)} | MACD {fv(val['MACD(12,26)'], d)} | "
            f"ADX {fv(val['ADX(14)'], 1)} | CCI {fv(val['CCI(14)'], 0)} | "
            f"ATR {fv(val['ATR(14)'], d)} ({atr_lab.lower()})\n"
            f"  Pivot klasik: S1 {fv(S[2], d)} | P {fv(P, d)} | R1 {fv(R[0], d)}")


def detail(name, cfg, res):
    d = cfg["dec"]
    h = res["Per Jam"]
    L = [f"ANALISIS TEKNIKAL {name} (Per Jam)", f"Rangkuman: {h['all']}",
         f"Moving Average: {h['ma_l']} (beli {h['mc'][0]}, jual {h['mc'][1]})  [sederhana | eksponensial]"]
    for n, sv, ss, ev, es in h["ma"]:
        L.append(f"  {n}: {fv(sv, d)} {ss or '-'} | {fv(ev, d)} {es or '-'}")
    L.append(f"Indikator: {h['ind_l']} (beli {h['ic'][0]}, jual {h['ic'][1]}, netral {h['ic'][2]})")
    for n, v, s in h["ind"]:
        L.append(f"  {n}: {fv(v, 3 if abs(v) < 100 else d)} {s}")
    L.append("Pivot points (dari candle Per Jam sebelumnya):")
    for k, (P, S, R) in h["piv"].items():
        L.append(f"  {k}: S3 {fv(S[0], d)} S2 {fv(S[1], d)} S1 {fv(S[2], d)} | "
                 f"P {fv(P, d)} | R1 {fv(R[0], d)} R2 {fv(R[1], d)} R3 {fv(R[2], d)}")
    for tf in ("30 Mnt", "4 Jam"):
        r = res[tf]
        L.append(f"{tf}: {r['all']} (MA {r['ma_l']}, Indikator {r['ind_l']})")
    L.append("Ini ringkasan indikator, bukan jaminan arah harga.")
    return "\n".join(L)


_orig = bot.analyze


def analyze_plus(name, cfg, m5):
    sigs, alerts, st = _orig(name, cfg, m5)
    try:
        res = {lab: analyze_tf(closed(m5, rule, dl)) for lab, rule, dl in TFS}
        st += "\n" + compact(name, cfg, res)
        if os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch":
            DETAILS.append(detail(name, cfg, res))
        h1 = res["Per Jam"]["all"]
        new = []
        for key, msg in sigs:
            msg += f"\nTeknikal Per Jam: {h1}"
            if (msg.startswith("SINYAL SELL") and "Beli" in h1) or (msg.startswith("SINYAL BUY") and "Jual" in h1):
                msg += "\nCatatan: ringkasan teknikal Per Jam berlawanan dengan arah sinyal."
            new.append((key, msg))
        sigs = new
    except Exception as e:
        st += f"\n  teknikal gagal: {type(e).__name__}"
        print("teknikal gagal:", type(e).__name__, str(e)[:150])
    return sigs, alerts, st


bot.analyze = analyze_plus

if __name__ == "__main__":
    bot.main()
    for m in DETAILS:
        bot.send(m)
