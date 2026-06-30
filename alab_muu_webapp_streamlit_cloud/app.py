\
import math
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="ALAB/MUU AI Web App", layout="wide")

DEFAULT_TICKERS = ["ALAB", "MUU", "MRVL", "NVDA", "CRDO", "QQQ", "SOXX", "SOXL"]

st.markdown("""
# ALAB/MUU AI Web App
恐怖やノイズではなく、**銘柄特性・移動平均線・出来高・RS・RR**で判断するためのWebアプリ。
""")

@st.cache_data(ttl=60 * 30)
def download_data(tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
    data = {}
    for t in tickers:
        df = yf.download(t, period=period, interval="1d", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna(how="all").copy()
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            data[t] = df
    return data

def add_indicators(df, qqq=None, soxx=None):
    d = df.copy()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()
    d["VolMA20"] = d["Volume"].rolling(20).mean()
    d["VolRatio20"] = d["Volume"] / d["VolMA20"]

    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev_close).abs(),
        (d["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()
    d["ATR14Pct"] = d["ATR14"] / d["Close"] * 100

    d["Ret1D"] = d["Close"].pct_change() * 100
    d["Ret5D"] = d["Close"].pct_change(5) * 100
    d["Ret20D"] = d["Close"].pct_change(20) * 100
    d["High52W"] = d["High"].rolling(252, min_periods=30).max()
    d["Drawdown52W"] = (d["Close"] / d["High52W"] - 1) * 100
    d["Is52WHigh"] = d["Close"] >= d["High52W"].shift(1)

    d["AboveMA20"] = d["Close"] > d["MA20"]
    d["AboveMA50"] = d["Close"] > d["MA50"]
    d["AboveMA200"] = d["Close"] > d["MA200"]

    if qqq is not None and not qqq.empty:
        q = qqq["Close"].reindex(d.index).ffill()
        d["RS_QQQ_20D"] = d["Ret20D"] - q.pct_change(20) * 100
        d["RS_QQQ_5D"] = d["Ret5D"] - q.pct_change(5) * 100
    else:
        d["RS_QQQ_20D"] = np.nan
        d["RS_QQQ_5D"] = np.nan

    if soxx is not None and not soxx.empty:
        s = soxx["Close"].reindex(d.index).ffill()
        d["RS_SOXX_20D"] = d["Ret20D"] - s.pct_change(20) * 100
        d["RS_SOXX_5D"] = d["Ret5D"] - s.pct_change(5) * 100
    else:
        d["RS_SOXX_20D"] = np.nan
        d["RS_SOXX_5D"] = np.nan

    return d

def streak_bool(s):
    vals = s.dropna().astype(bool).tolist()
    n = 0
    for v in vals[::-1]:
        if v:
            n += 1
        else:
            break
    return n

def ma_break_stats(d, ma_col):
    above = d["Close"] > d[ma_col]
    breaks = (~above) & (above.shift(1) == True) & d[ma_col].notna()

    recovery_days = []
    failed = 0

    for idx in np.where(breaks.values)[0]:
        ok = False
        for j in range(idx + 1, min(idx + 31, len(d))):
            if d["Close"].iloc[j] > d[ma_col].iloc[j]:
                recovery_days.append(j - idx)
                ok = True
                break
        if not ok:
            failed += 1

    return {
        "breaks": int(breaks.sum()),
        "current_above_streak": streak_bool(above),
        "avg_recovery": float(np.mean(recovery_days)) if recovery_days else None,
        "within_1d": int(sum(x <= 1 for x in recovery_days)),
        "within_3d": int(sum(x <= 3 for x in recovery_days)),
        "within_5d": int(sum(x <= 5 for x in recovery_days)),
        "failed_30d": failed,
    }

def pullback_stats(d):
    trend = d[d["AboveMA50"].fillna(False)].copy()
    if trend.empty:
        return {"avg": None, "median": None, "max": None}
    trend["Peak"] = trend["Close"].cummax()
    trend["DD"] = (trend["Close"] / trend["Peak"] - 1) * 100
    p = trend["DD"][trend["DD"] < -1]
    return {
        "avg": float(p.mean()) if not p.empty else 0.0,
        "median": float(p.median()) if not p.empty else 0.0,
        "max": float(p.min()) if not p.empty else 0.0
    }

def high_after_pullback(d, window=20):
    idxs = list(np.where(d["Is52WHigh"].fillna(False).values)[0])
    vals = []
    for idx in idxs:
        future = d.iloc[idx:min(idx + window + 1, len(d))]
        if len(future) > 1:
            vals.append((future["Low"].min() / d["Close"].iloc[idx] - 1) * 100)
    return {
        "count": len(idxs),
        "avg": float(np.mean(vals)) if vals else None,
        "median": float(np.median(vals)) if vals else None,
        "max": float(np.min(vals)) if vals else None
    }

def accumulation_distribution(d, lookback=20):
    x = d.tail(lookback)
    vol_up = x["Volume"] > x["Volume"].shift(1)
    acc = (x["Close"] > x["Close"].shift(1)) & vol_up
    dist = (x["Close"] < x["Close"].shift(1)) & vol_up
    return {
        "acc": int(acc.sum()),
        "dist": int(dist.sum()),
        "net": int(acc.sum() - dist.sum())
    }

def health_score(d):
    latest = d.iloc[-1]
    score = 50
    notes = []

    if bool(latest["AboveMA20"]):
        score += 10; notes.append("20MA上")
    else:
        score -= 10; notes.append("20MA下")

    if bool(latest["AboveMA50"]):
        score += 15; notes.append("50MA上")
    else:
        score -= 20; notes.append("50MA下")

    if bool(latest["AboveMA200"]):
        score += 10; notes.append("200MA上")
    else:
        score -= 15; notes.append("200MA下")

    rs = latest.get("RS_QQQ_20D", np.nan)
    if pd.notna(rs):
        if rs > 10:
            score += 10; notes.append("QQQ比かなり強い")
        elif rs > 0:
            score += 5; notes.append("QQQ比プラス")
        else:
            score -= 5; notes.append("QQQ比マイナス")

    vol = latest.get("VolRatio20", np.nan)
    if pd.notna(vol):
        if vol > 2 and latest["Ret1D"] > 0:
            score += 5; notes.append("出来高を伴う上昇")
        elif vol > 2 and latest["Ret1D"] < 0:
            score -= 12; notes.append("出来高を伴う下落")

    dd = latest.get("Drawdown52W", np.nan)
    if pd.notna(dd):
        if dd > -5:
            score += 5; notes.append("高値圏維持")
        elif dd < -20:
            score -= 10; notes.append("高値から大きく下落")

    ad = accumulation_distribution(d)
    if ad["net"] >= 3:
        score += 5; notes.append("蓄積優勢")
    elif ad["net"] <= -3:
        score -= 8; notes.append("分配優勢")

    return int(max(0, min(100, score))), notes

def ai_decision(d, ticker):
    latest = d.iloc[-1]
    score, notes = health_score(d)
    m20 = ma_break_stats(d, "MA20")
    m50 = ma_break_stats(d, "MA50")
    p = pullback_stats(d)
    h = high_after_pullback(d)
    ad = accumulation_distribution(d)

    price = float(latest["Close"])
    dd52 = latest.get("Drawdown52W", np.nan)
    vol = latest.get("VolRatio20", np.nan)
    rs = latest.get("RS_QQQ_20D", np.nan)

    pullback_is_normal = pd.notna(dd52) and -15 <= dd52 <= -3
    trend_ok = bool(latest["AboveMA50"]) and bool(latest["AboveMA200"])
    volume_not_distribution = not (pd.notna(vol) and vol > 2 and latest["Ret1D"] < 0)
    rs_ok = pd.isna(rs) or rs > -5

    if score >= 78 and pullback_is_normal and trend_ok and volume_not_distribution and rs_ok:
        decision, stars, action = "押し目買い候補", "★★★★★", "分割で買いを検討"
    elif score >= 75 and trend_ok:
        decision, stars, action = "保有優位", "★★★★☆", "慌てて売らない。押し目待ち"
    elif score >= 60 and trend_ok:
        decision, stars, action = "様子見", "★★★☆☆", "買い急がず、20MA/50MA反応を確認"
    elif score >= 45:
        decision, stars, action = "警戒", "★★☆☆☆", "新規買いは抑える。50MA回復待ち"
    else:
        decision, stars, action = "危険", "★☆☆☆☆", "トレンド修復まで待つ"

    reasons = []
    reasons.append("50MA・200MA上で中期トレンド維持" if trend_ok else "中期トレンドに傷あり")
    if pd.notna(rs):
        reasons.append(f"QQQ比RS 20日: {rs:.1f}%")
    if pd.notna(vol):
        reasons.append(f"出来高倍率: {vol:.2f}倍")
    if pd.notna(dd52):
        reasons.append(f"52週高値から: {dd52:.1f}%")

    buy_zones = {
        "5%押し": price * 0.95,
        "8%押し": price * 0.92,
        "12%押し": price * 0.88,
        "15%押し": price * 0.85,
    }

    high52 = latest.get("High52W", np.nan)
    if pd.notna(high52):
        targets = [float(high52), float(high52 * 1.08), float(high52 * 1.18)]
    else:
        targets = [price * 1.08, price * 1.15, price * 1.25]

    stop = float(latest["MA50"] * 0.97) if pd.notna(latest["MA50"]) else price * 0.9

    return {
        "decision": decision,
        "stars": stars,
        "action": action,
        "score": score,
        "notes": notes,
        "reasons": reasons,
        "ma20": m20,
        "ma50": m50,
        "pullback": p,
        "after_high": h,
        "ad": ad,
        "buy_zones": buy_zones,
        "targets": targets,
        "stop": stop,
    }

def rr_table(price, stop, targets):
    rows = []
    risk = price - stop
    for t in targets:
        reward = t - price
        rows.append({
            "現在値": price,
            "目標": t,
            "撤退": stop,
            "下値リスク%": (stop / price - 1) * 100,
            "上値余地%": (t / price - 1) * 100,
            "RR": reward / risk if risk > 0 else np.nan
        })
    return pd.DataFrame(rows)

def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    return f"{v:.{nd}f}"

with st.sidebar:
    st.header("設定")
    tickers_text = st.text_input("取得銘柄", "ALAB MUU MRVL NVDA CRDO QQQ SOXX SOXL")
    tickers = [x.strip().upper() for x in tickers_text.split() if x.strip()]
    period = st.selectbox("取得期間", ["1y", "2y", "5y"], index=1)
    selected = st.selectbox("表示銘柄", tickers, index=0)
    if st.button("最新データ取得"):
        st.cache_data.clear()

try:
    raw = download_data(tickers, period)
except Exception as e:
    st.error("データ取得でエラーが出ました。少し時間を置いて再読み込みしてください。")
    st.code(str(e))
    st.stop()

if selected not in raw or raw[selected].empty:
    st.error(f"{selected} のデータが取れませんでした。")
    st.stop()

qqq, soxx = raw.get("QQQ"), raw.get("SOXX")
data = {t: add_indicators(df, qqq=qqq, soxx=soxx) for t, df in raw.items()}

d = data[selected].dropna(subset=["Close"]).copy()
latest = d.iloc[-1]
ai = ai_decision(d, selected)

c1, c2, c3, c4 = st.columns(4)
c1.metric("AI判定", ai["decision"], ai["stars"])
c2.metric("行動", ai["action"])
c3.metric("健康スコア", f"{ai['score']}/100")
c4.metric("現在値", f"{latest['Close']:.2f}")
st.write("理由： " + " / ".join(ai["reasons"]))

tab1, tab2, tab3, tab4 = st.tabs(["AI判定", "銘柄カルテ", "チャート", "比較ランキング"])

with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("押し目買いライン")
        st.dataframe(pd.DataFrame([[k, v] for k, v in ai["buy_zones"].items()], columns=["条件", "価格"]), use_container_width=True, hide_index=True)

        st.subheader("目標価格とRR")
        st.dataframe(rr_table(float(latest["Close"]), ai["stop"], ai["targets"]), use_container_width=True, hide_index=True)

    with right:
        st.subheader("AIコメント")
        comments = []
        if ai["decision"] == "押し目買い候補":
            comments.append("トレンドが維持されたまま押している可能性が高い。分割買いの候補。")
        elif ai["decision"] == "保有優位":
            comments.append("今は無理に買い増すより、保有して伸ばす局面。売る理由は弱い。")
        elif ai["decision"] == "様子見":
            comments.append("悪くはないが、買い増しには少し根拠不足。反発確認を待つ。")
        else:
            comments.append("買いより防御。移動平均線と出来高の改善を待つ。")

        comments.append(f"20MA割れは過去{ai['ma20']['breaks']}回。平均回復は{fmt(ai['ma20']['avg_recovery'])}日。")
        comments.append(f"上昇トレンド中の平均押しは{fmt(ai['pullback']['avg'])}%、最大押しは{fmt(ai['pullback']['max'])}%。")
        comments.append(f"52週高値更新後の平均押しは{fmt(ai['after_high']['avg'])}%。")
        comments.append(f"直近20日の蓄積日は{ai['ad']['acc']}、分配日は{ai['ad']['dist']}。")

        st.info("\\n\\n".join(comments))

with tab2:
    st.subheader(f"{selected} 銘柄カルテ")
    rows = [
        ["終値", latest["Close"]],
        ["20MA", latest["MA20"]],
        ["50MA", latest["MA50"]],
        ["200MA", latest["MA200"]],
        ["20MA乖離%", (latest["Close"]/latest["MA20"] - 1)*100 if pd.notna(latest["MA20"]) else np.nan],
        ["50MA乖離%", (latest["Close"]/latest["MA50"] - 1)*100 if pd.notna(latest["MA50"]) else np.nan],
        ["出来高倍率", latest["VolRatio20"]],
        ["ATR14%", latest["ATR14Pct"]],
        ["52週高値から%", latest["Drawdown52W"]],
        ["QQQ比RS 20日", latest["RS_QQQ_20D"]],
        ["SOXX比RS 20日", latest["RS_SOXX_20D"]],
        ["20MA割れ回数", ai["ma20"]["breaks"]],
        ["20MA上連続日数", ai["ma20"]["current_above_streak"]],
        ["20MA割れ後 平均回復日数", ai["ma20"]["avg_recovery"]],
        ["50MA割れ回数", ai["ma50"]["breaks"]],
        ["50MA上連続日数", ai["ma50"]["current_above_streak"]],
        ["平均押し%", ai["pullback"]["avg"]],
        ["最大押し%", ai["pullback"]["max"]],
        ["52週高値更新後 平均押し%", ai["after_high"]["avg"]],
        ["蓄積日", ai["ad"]["acc"]],
        ["分配日", ai["ad"]["dist"]],
    ]
    st.dataframe(pd.DataFrame(rows, columns=["項目", "値"]), use_container_width=True, hide_index=True)

with tab3:
    lookback = st.selectbox("チャート期間", [90, 180, 365, 730], index=2)
    x = d.tail(lookback)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=x.index, open=x["Open"], high=x["High"], low=x["Low"], close=x["Close"], name=selected))
    fig.add_trace(go.Scatter(x=x.index, y=x["MA20"], name="20MA"))
    fig.add_trace(go.Scatter(x=x.index, y=x["MA50"], name="50MA"))
    fig.add_trace(go.Scatter(x=x.index, y=x["MA200"], name="200MA"))
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("比較ランキング")
    ranks = []
    for t, df0 in data.items():
        if t in ["QQQ", "SOXX"]:
            continue
        dd = df0.dropna(subset=["Close"])
        if dd.empty:
            continue
        a = ai_decision(dd, t)
        l = dd.iloc[-1]
        ranks.append({
            "銘柄": t,
            "AI判定": a["decision"],
            "健康スコア": a["score"],
            "終値": l["Close"],
            "1日%": l["Ret1D"],
            "20日%": l["Ret20D"],
            "QQQ比RS20日": l["RS_QQQ_20D"],
            "SOXX比RS20日": l["RS_SOXX_20D"],
            "出来高倍率": l["VolRatio20"],
            "52週高値から%": l["Drawdown52W"],
        })
    rank_df = pd.DataFrame(ranks).sort_values(["健康スコア", "QQQ比RS20日"], ascending=False)
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

st.caption("投資判断補助ツールです。売買を保証するものではありません。")
