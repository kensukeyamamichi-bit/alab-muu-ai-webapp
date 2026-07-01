
import math
from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="ALAB/MUU Buy-Sell Matrix v11", layout="wide")

st.markdown("""
# ALAB/MUU Buy-Sell Matrix v11
**買いポイント**と**利確ポイント**を同じ画面で見る版。  
目的：  
- 買い：押し目・期待値・トレンド維持  
- 売り：過熱度・5MA乖離・急騰後統計  
を同時に見て、**買う / 保有 / 一部利確 / 待つ**を判断する。

今回の修正：**5MA乖離率だけでは全利確にしない**。さらに押し目買いでは、個別銘柄だけでなく **市場全体のリバランス/指数連動の売り** を判定に入れる。

追加：QQQ/SOXXの下落、出来高、月末・四半期末、ウォッチ銘柄の同時下落率から、**市場リバランス押し目** か **個別悪化** かを分ける。

今回のv11追加：**S&P500（SPY）/ Tech100（QQQ）/ 半導体（SOXX）/ 長期国債（TLT）/ 金（GLD）/ 長期金利（^TNX）** をマクロ指標として組み込み、
「リスクオンの押し目」か「金利上昇・リスクオフの危険な下げ」かを分ける。
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
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA10"] = d["Close"].rolling(10).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()

    for ma in [5,10,20,50,200]:
        d[f"GapMA{ma}"] = (d["Close"] / d[f"MA{ma}"] - 1) * 100
        d[f"AboveMA{ma}"] = d["Close"] > d[f"MA{ma}"]

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

    for n in [1,3,5,10,20]:
        d[f"Ret{n}D"] = d["Close"].pct_change(n) * 100

    d["High52W"] = d["High"].rolling(252, min_periods=30).max()
    d["Drawdown52W"] = (d["Close"] / d["High52W"] - 1) * 100
    d["Is52WHigh"] = d["Close"] >= d["High52W"].shift(1)

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
        if v: n += 1
        else: break
    return n

def ma_break_stats(d, ma_col):
    above = d["Close"] > d[ma_col]
    breaks = (~above) & (above.shift(1) == True) & d[ma_col].notna()
    recovery_days, failed = [], 0
    for idx in np.where(breaks.values)[0]:
        ok = False
        for j in range(idx+1, min(idx+31, len(d))):
            if d["Close"].iloc[j] > d[ma_col].iloc[j]:
                recovery_days.append(j-idx)
                ok = True
                break
        if not ok:
            failed += 1
    return {
        "breaks": int(breaks.sum()),
        "current_above_streak": streak_bool(above),
        "avg_recovery": float(np.mean(recovery_days)) if recovery_days else None,
        "within_3d": int(sum(x <= 3 for x in recovery_days)),
        "within_5d": int(sum(x <= 5 for x in recovery_days)),
        "failed_30d": failed,
    }

def accumulation_distribution(d, lookback=20):
    x = d.tail(lookback)
    vol_up = x["Volume"] > x["Volume"].shift(1)
    acc = (x["Close"] > x["Close"].shift(1)) & vol_up
    dist = (x["Close"] < x["Close"].shift(1)) & vol_up
    return {"acc": int(acc.sum()), "dist": int(dist.sum()), "net": int(acc.sum()-dist.sum())}





def macro_risk_score(data: Dict[str, pd.DataFrame]):
    """S&P500/Tech100/半導体/国債/金/長期金利から、押し目の質を判定。
    高いほど「リスクオン環境の健全な押し目」。低いほど「金利上昇・リスクオフで危険な下げ」。
    """
    score = 50
    notes = []
    details = {}

    def one_day(ticker):
        df = data.get(ticker)
        if df is None or df.empty or len(df.dropna(subset=["Close"])) < 25:
            return np.nan, np.nan, np.nan
        x = df.dropna(subset=["Close"])
        ret1 = x["Close"].pct_change().iloc[-1] * 100
        ret5 = x["Close"].pct_change(5).iloc[-1] * 100
        volr = (x["Volume"] / x["Volume"].rolling(20).mean()).iloc[-1] if "Volume" in x else np.nan
        return ret1, ret5, volr

    spy1, spy5, spyv = one_day("SPY")
    qqq1, qqq5, qqqv = one_day("QQQ")
    soxx1, soxx5, soxxv = one_day("SOXX")
    tlt1, tlt5, _ = one_day("TLT")
    ief1, ief5, _ = one_day("IEF")
    gld1, gld5, _ = one_day("GLD")
    gold1, gold5, _ = one_day("GC=F")
    tnx1, tnx5, _ = one_day("^TNX")
    if pd.isna(tnx1):
        tnx1, tnx5, _ = one_day("TNX")

    # details
    for name, val in [("S&P500(SPY)当日%", spy1), ("Tech100(QQQ)当日%", qqq1), ("半導体(SOXX)当日%", soxx1),
                      ("長期国債(TLT)当日%", tlt1), ("中期国債(IEF)当日%", ief1), ("金(GLD/GC)当日%", gld1 if pd.notna(gld1) else gold1),
                      ("長期金利(^TNX)当日%", tnx1)]:
        if pd.notna(val): details[name] = val

    # 株式指数の地合い
    if pd.notna(spy1):
        if spy1 <= -1.0: score -= 8; notes.append("S&P500が弱い")
        elif spy1 >= 0.5: score += 6; notes.append("S&P500は底堅い")
    if pd.notna(qqq1):
        if qqq1 <= -1.3: score -= 10; notes.append("Tech100が弱い")
        elif qqq1 >= 0.7: score += 8; notes.append("Tech100は強い")
    if pd.notna(soxx1):
        if soxx1 <= -1.8: score -= 10; notes.append("半導体全体が弱い")
        elif soxx1 >= 1.0: score += 10; notes.append("半導体は強い")

    # 金利・国債：TLT下落＋TNX上昇はグロースに逆風
    if pd.notna(tlt1):
        if tlt1 <= -1.0: score -= 12; notes.append("長期国債下落＝金利上昇圧力")
        elif tlt1 >= 1.0: score += 10; notes.append("長期国債上昇＝金利低下追い風")
    if pd.notna(ief1):
        if ief1 <= -0.5: score -= 6; notes.append("中期国債も弱い")
        elif ief1 >= 0.5: score += 5; notes.append("中期国債は強い")
    if pd.notna(tnx1):
        if tnx1 >= 1.5: score -= 14; notes.append("長期金利が上昇")
        elif tnx1 <= -1.5: score += 12; notes.append("長期金利が低下")

    # 金：株安＋金高はリスクオフ、株高＋金高はインフレ/ドル要因で中立寄り
    gold_ret = gld1 if pd.notna(gld1) else gold1
    if pd.notna(gold_ret):
        if gold_ret >= 0.8 and (pd.notna(qqq1) and qqq1 < 0):
            score -= 8; notes.append("株安＋金高＝リスクオフ気味")
        elif gold_ret <= -0.8 and (pd.notna(qqq1) and qqq1 > 0):
            score += 4; notes.append("金弱く株強い＝リスクオン")

    # 重要な組み合わせ
    if pd.notna(qqq1) and pd.notna(tlt1) and qqq1 < 0 and tlt1 < 0:
        score -= 12; notes.append("株と債券が同時下落＝悪い押し目")
    if pd.notna(qqq1) and pd.notna(tlt1) and qqq1 < 0 and tlt1 > 0:
        score += 10; notes.append("株安でも債券高＝金利低下型の押し目")
    if pd.notna(qqq1) and pd.notna(soxx1) and qqq1 > 0 and soxx1 > 0:
        score += 8; notes.append("Tech/半導体が同時に強い")


    score = int(max(0, min(100, score)))
    if score >= 75:
        label = "リスクオン良好"
    elif score >= 60:
        label = "押し目許容の地合い"
    elif score >= 45:
        label = "中立"
    else:
        label = "金利/リスクオフ警戒"
    return score, label, notes, details


def market_rebalance_score(data: Dict[str, pd.DataFrame], selected: str):
    """押し目買い用の市場全体リバランス判定。
    個別悪化ではなく、指数・半導体全体・ウォッチ銘柄が同時に下げているかを見る。
    スコアが高いほど「市場全体の需給/リバランスによる押し目」の可能性が高い。
    """
    if selected not in data or data[selected].empty:
        return 50, "市場判定不可", [], {}

    d = data[selected].dropna(subset=["Close"])
    latest = d.iloc[-1]
    score = 50
    notes = []
    details = {}

    # 月末・四半期末は機械的なリバランス売り/買いが出やすい
    dt = d.index[-1]
    cal = pd.Timestamp(dt)
    month_end = cal + pd.tseries.offsets.BMonthEnd(0)
    days_to_month_end = (month_end.date() - cal.date()).days
    is_month_rebalance = days_to_month_end <= 3 or cal.day <= 3
    is_quarter_month = cal.month in [3, 6, 9, 12]
    details["月末まで日数"] = days_to_month_end
    if is_month_rebalance:
        score += 10; notes.append("月末/月初リバランス期間")
    if is_month_rebalance and is_quarter_month:
        score += 8; notes.append("四半期末リバランス期間")

    # 指数の動き：QQQ/SOXXが同時下落なら個別悪化ではなく市場要因の可能性
    q = data.get("QQQ")
    sx = data.get("SOXX")
    q_ret = sx_ret = np.nan
    q_vol = sx_vol = np.nan
    if q is not None and not q.empty:
        qd = q.dropna(subset=["Close"]).copy()
        q_ret = qd["Close"].pct_change().iloc[-1] * 100
        q_vol = (qd["Volume"] / qd["Volume"].rolling(20).mean()).iloc[-1]
        details["QQQ当日%"] = q_ret
        details["QQQ出来高倍率"] = q_vol
        if q_ret <= -1.0:
            score += 14; notes.append("QQQが1%以上下落")
        elif q_ret >= 1.0:
            score -= 6; notes.append("QQQは強い")
        if pd.notna(q_vol) and q_vol >= 1.25 and q_ret < 0:
            score += 8; notes.append("QQQ下落日に出来高増")

    if sx is not None and not sx.empty:
        sd = sx.dropna(subset=["Close"]).copy()
        sx_ret = sd["Close"].pct_change().iloc[-1] * 100
        sx_vol = (sd["Volume"] / sd["Volume"].rolling(20).mean()).iloc[-1]
        details["SOXX当日%"] = sx_ret
        details["SOXX出来高倍率"] = sx_vol
        if sx_ret <= -1.5:
            score += 16; notes.append("SOXXが1.5%以上下落")
        elif sx_ret >= 1.5:
            score -= 8; notes.append("SOXXは強い")
        if pd.notna(sx_vol) and sx_vol >= 1.25 and sx_ret < 0:
            score += 8; notes.append("SOXX下落日に出来高増")

    # ウォッチ銘柄全体の同時下落率・上昇率
    watch = [t for t in data.keys() if t not in ["QQQ", "SOXX", "SPY", "TLT", "IEF", "GLD", "GC=F", "^TNX", "TNX"]]
    rets, above20, above50 = [], [], []
    for t in watch:
        x = data[t].dropna(subset=["Close"])
        if len(x) < 60:
            continue
        r = x["Close"].pct_change().iloc[-1] * 100
        rets.append(r)
        above20.append(bool(x["Close"].iloc[-1] > x["MA20"].iloc[-1]))
        above50.append(bool(x["Close"].iloc[-1] > x["MA50"].iloc[-1]))
    if rets:
        down_ratio = sum(r < 0 for r in rets) / len(rets)
        big_down_ratio = sum(r <= -3 for r in rets) / len(rets)
        details["ウォッチ同時下落率%"] = down_ratio * 100
        details["ウォッチ3%以上下落率%"] = big_down_ratio * 100
        details["20MA上比率%"] = np.mean(above20) * 100 if above20 else np.nan
        details["50MA上比率%"] = np.mean(above50) * 100 if above50 else np.nan
        if down_ratio >= 0.65:
            score += 18; notes.append("ウォッチ銘柄が同時下落")
        elif down_ratio <= 0.35:
            score -= 10; notes.append("市場全体は崩れていない")
        if big_down_ratio >= 0.35:
            score += 10; notes.append("高ベータ銘柄に広範な売り")

    # 個別だけ極端に弱い場合はリバランスではなく個別悪化として減点
    sel_ret = latest.get("Ret1D", np.nan)
    if pd.notna(sel_ret):
        details[f"{selected}当日%"] = sel_ret
        idx_ref = np.nan
        if pd.notna(sx_ret): idx_ref = sx_ret
        elif pd.notna(q_ret): idx_ref = q_ret
        if pd.notna(idx_ref):
            rel = sel_ret - idx_ref
            details["個別-指数%"] = rel
            if rel <= -4:
                score -= 22; notes.append("個別が指数より大きく弱い")
            elif rel >= 2:
                score += 8; notes.append("個別は指数より耐えている")

    score = int(max(0, min(100, score)))
    if score >= 75:
        label = "市場リバランス押し目の可能性高い"
    elif score >= 60:
        label = "市場要因の押し目寄り"
    elif score >= 45:
        label = "市場/個別どちらもあり"
    else:
        label = "個別要因・弱さに注意"
    return score, label, notes, details

def distribution_days(d, lookback=25):
    """IBD風のDistribution Day: 下落＋出来高増＋終値が日中レンジ下位50%。直近25営業日でカウント。"""
    x = d.tail(lookback).copy()
    rng = (x["High"] - x["Low"]).replace(0, np.nan)
    close_pos = (x["Close"] - x["Low"]) / rng
    dist = (
        (x["Close"] < x["Close"].shift(1)) &
        (x["Volume"] > x["Volume"].shift(1)) &
        (close_pos <= 0.50)
    ).fillna(False)
    return int(dist.sum()), dist

def capital_flow_score(d):
    """過熱時に本当に売るべきかを見るための資金流入スコア。"""
    latest = d.iloc[-1]
    score = 50
    notes = []

    vol = latest.get("VolRatio20", np.nan)
    ret1 = latest.get("Ret1D", np.nan)
    gap5 = latest.get("GapMA5", np.nan)
    rs20 = latest.get("RS_QQQ_20D", np.nan)
    rs5 = latest.get("RS_QQQ_5D", np.nan)

    if pd.notna(vol):
        if vol >= 1.5 and ret1 > 0:
            score += 18; notes.append("出来高を伴う上昇")
        elif vol >= 1.5 and ret1 < 0:
            score -= 22; notes.append("出来高を伴う下落")
        elif vol >= 1.0:
            score += 5; notes.append("出来高は平均以上")
        else:
            score -= 5; notes.append("出来高は弱め")

    if pd.notna(rs20):
        if rs20 >= 15:
            score += 15; notes.append("QQQ比RSが非常に強い")
        elif rs20 > 0:
            score += 8; notes.append("QQQ比RSプラス")
        else:
            score -= 10; notes.append("QQQ比RSマイナス")

    if pd.notna(rs5):
        if rs5 > 0:
            score += 6; notes.append("短期RSも強い")
        else:
            score -= 6; notes.append("短期RSが鈍化")

    # 終値位置: 高値引けに近いほど買い継続、安値引けは売り圧力
    rng = latest["High"] - latest["Low"]
    if rng and pd.notna(rng) and rng > 0:
        close_pos = (latest["Close"] - latest["Low"]) / rng
        if close_pos >= 0.70:
            score += 10; notes.append("終値位置が強い")
        elif close_pos <= 0.35:
            score -= 12; notes.append("終値位置が弱い")

    ad = accumulation_distribution(d)
    dist_count, _dist_series = distribution_days(d)
    if ad["net"] >= 3:
        score += 10; notes.append("蓄積優勢")
    elif ad["net"] <= -3:
        score -= 12; notes.append("分配優勢")

    dist_count, _ = distribution_days(d)
    if dist_count <= 1:
        score += 6; notes.append("Distribution Day少ない")
    elif dist_count >= 4:
        score -= 18; notes.append("Distribution Day多い")
    elif dist_count >= 2:
        score -= 8; notes.append("Distribution Dayやや増加")

    return int(max(0, min(100, score))), notes, dist_count


def pullback_quality_score(d, market_rebalance=50):
    """週足/日足の押し目買い思想を日足データで近似。
    高値追いではなく、強いトレンド内で短期線〜中期線に近づいた時を高評価。
    """
    latest = d.iloc[-1]
    score = 45
    notes = []

    close = latest["Close"]
    gap5 = latest.get("GapMA5", np.nan)
    gap20 = latest.get("GapMA20", np.nan)
    gap50 = latest.get("GapMA50", np.nan)
    rs20 = latest.get("RS_QQQ_20D", np.nan)
    vol = latest.get("VolRatio20", np.nan)

    # トレンド前提
    if bool(latest.get("AboveMA50", False)) and bool(latest.get("AboveMA200", False)):
        score += 20; notes.append("中長期トレンド上")
    else:
        score -= 25; notes.append("中長期トレンド弱い")

    # 押し目の位置。5MA乖離が大きい時は追いかけ買いではない。
    if pd.notna(gap5):
        if -2 <= gap5 <= 3:
            score += 25; notes.append("5MA付近の浅い押し")
        elif 3 < gap5 <= 7:
            score += 10; notes.append("5MA上だがまだ許容")
        elif gap5 > 10:
            score -= 30; notes.append("5MAから上方乖離、追いかけ買い注意")
        elif gap5 < -5:
            score -= 10; notes.append("5MA下で短期弱含み")

    if pd.notna(gap20):
        if 0 <= gap20 <= 8:
            score += 18; notes.append("20MA上の良い押し目")
        elif gap20 > 20:
            score -= 15; notes.append("20MAからも過熱")
        elif gap20 < 0:
            score -= 10; notes.append("20MA割れ")

    if pd.notna(rs20):
        if rs20 > 10:
            score += 10; notes.append("RS強い")
        elif rs20 < 0:
            score -= 10; notes.append("RS弱い")

    if pd.notna(vol):
        if vol >= 1.2 and latest.get("Ret1D", 0) > 0:
            score += 7; notes.append("反発出来高あり")
        elif vol >= 1.5 and latest.get("Ret1D", 0) < 0:
            score -= 12; notes.append("下落出来高あり")

    # 市場全体のリバランス/指数連動売りなら、強い銘柄の押し目として加点。
    # 逆に市場が強いのに個別だけ下げている場合は減点。
    if market_rebalance >= 75:
        score += 18; notes.append("市場リバランス由来の押し目")
    elif market_rebalance >= 60:
        score += 10; notes.append("市場要因の押し目寄り")
    elif market_rebalance < 45 and latest.get("Ret1D", 0) < 0:
        score -= 15; notes.append("個別要因の下落に注意")

    score = int(max(0, min(100, score)))
    if score >= 80:
        label = "S級押し目"
    elif score >= 65:
        label = "A級押し目"
    elif score >= 50:
        label = "B級・様子見"
    else:
        label = "追いかけ/弱い押し目"
    return score, label, notes


def trend_continuation_score(d):
    """5MA乖離が高くても売らないケースを拾うためのトレンド継続スコア。
    ATH/52週高値、RS、MA上、出来高、Distribution Dayの少なさを評価。
    """
    latest = d.iloc[-1]
    score = 50
    notes = []

    # ATH/52週高値更新は価格発見局面として強く評価
    is_high = bool(latest.get("Is52WHigh", False))
    dd = latest.get("Drawdown52W", np.nan)
    if is_high or (pd.notna(dd) and dd >= -2):
        score += 18; notes.append("ATH/52週高値圏で価格発見")
    elif pd.notna(dd) and dd <= -10:
        score -= 12; notes.append("52週高値から距離あり")

    # 移動平均線の並び
    if bool(latest.get("AboveMA5", False)):
        score += 8; notes.append("5MA上を維持")
    else:
        score -= 8; notes.append("5MA割れ")
    if bool(latest.get("AboveMA20", False)):
        score += 10; notes.append("20MA上を維持")
    else:
        score -= 14; notes.append("20MA割れ")
    if bool(latest.get("AboveMA50", False)):
        score += 8; notes.append("50MA上")
    else:
        score -= 12; notes.append("50MA割れ")

    # RS
    rs20 = latest.get("RS_QQQ_20D", np.nan)
    rs5 = latest.get("RS_QQQ_5D", np.nan)
    if pd.notna(rs20):
        if rs20 >= 20:
            score += 16; notes.append("20日RSが極めて強い")
        elif rs20 >= 8:
            score += 10; notes.append("20日RSが強い")
        elif rs20 < 0:
            score -= 10; notes.append("20日RSが弱い")
    if pd.notna(rs5):
        if rs5 > 0:
            score += 6; notes.append("短期RSプラス")
        else:
            score -= 5; notes.append("短期RSマイナス")

    # 出来高と終値位置
    vol = latest.get("VolRatio20", np.nan)
    ret1 = latest.get("Ret1D", np.nan)
    if pd.notna(vol) and pd.notna(ret1):
        if vol >= 1.2 and ret1 > 0:
            score += 8; notes.append("上昇日に出来高あり")
        elif vol >= 1.5 and ret1 < 0:
            score -= 14; notes.append("下落日に出来高増")

    rng = latest["High"] - latest["Low"]
    if pd.notna(rng) and rng > 0:
        close_pos = (latest["Close"] - latest["Low"]) / rng
        if close_pos >= 0.60:
            score += 6; notes.append("終値位置が強い")
        elif close_pos <= 0.35:
            score -= 8; notes.append("終値位置が弱い")

    dist_count, _ = distribution_days(d)
    if dist_count <= 1:
        score += 8; notes.append("Distribution Day少ない")
    elif dist_count >= 4:
        score -= 18; notes.append("Distribution Day多い")
    elif dist_count >= 2:
        score -= 8; notes.append("Distribution Dayやや増加")

    score = int(max(0, min(100, score)))
    if score >= 80:
        label = "トレンド継続濃厚"
    elif score >= 65:
        label = "トレンド継続優位"
    elif score >= 50:
        label = "中立・観察"
    else:
        label = "トレンド鈍化"
    return score, label, notes


def market_context_label(d, heat, trend_score, flow, dist_days):
    """イベント出尽くし型か、トレンド継続型かを簡易分類。"""
    latest = d.iloc[-1]
    gap5 = latest.get("GapMA5", np.nan)
    is_high = bool(latest.get("Is52WHigh", False))
    if trend_score >= 75 and flow >= 70 and dist_days <= 2 and (is_high or (pd.notna(gap5) and gap5 >= 5)):
        return "トレンド型：過熱でも保有優位"
    if heat >= 85 and trend_score < 60 and (flow < 60 or dist_days >= 3):
        return "イベント出尽くし型：利確優位"
    if heat >= 85:
        return "過熱警戒：全売りではなく条件確認"
    return "通常判定"

def full_exit_conditions(d, heat, flow, dist_days, trend_score=50):
    """全利確優位は、5MA乖離や過熱だけでは出さない。
    強いトレンド銘柄は、複合悪化が出るまで全利確にしない。
    """
    latest = d.iloc[-1]
    conds = []
    conds.append(("利確過熱度90以上", heat >= 90))
    conds.append(("資金流入スコア50以下", flow <= 50))
    conds.append(("Distribution Day 4日以上", dist_days >= 4))
    trend_bad = False
    if pd.notna(latest.get("MA20", np.nan)):
        trend_bad = latest["Close"] < latest["MA20"]
    # 追加で5MA/20MAの短期崩れも補助条件にするが、単独では弱い
    conds.append(("20MA終値割れ", trend_bad))
    conds.append(("トレンド継続スコア60未満", trend_score < 60))
    count = sum(v for _, v in conds)
    return count, conds

def flow_label(flow, heat, dist_days):
    if heat >= 85 and flow >= 75 and dist_days <= 2:
        return "🟢 過熱だが機関はまだ買っている"
    if flow >= 75 and dist_days <= 2:
        return "🟢 資金流入は強い"
    if flow < 60 and dist_days >= 4:
        return "🔴 過熱＋機関売り増加"
    if dist_days >= 3:
        return "🟡 Distribution Day増加で警戒"
    if flow < 60:
        return "🟡 資金流入が鈍化"
    return "🟡 買いと売りが拮抗"

def final_action_v8(d, buy, heat, flow=80, dist_days=0, market_rebalance=50):
    # 全利確は5条件中4つ以上。5MA乖離・過熱だけでは絶対に全利確にしない。
    trend_score, trend_label, trend_notes = trend_continuation_score(d)
    context = market_context_label(d, heat, trend_score, flow, dist_days)
    exit_count, exit_conds = full_exit_conditions(d, heat, flow, dist_days, trend_score)
    pull_score, pull_label, pull_notes = pullback_quality_score(d, market_rebalance)

    if exit_count >= 4:
        return "全利確優位", "過熱だけでなく、資金流出・Distribution Day・20MA割れのうち複数が重なっています。トレンド終了リスクが高いため全利確を検討。", exit_count, exit_conds, pull_score, pull_label, pull_notes, trend_score, trend_label, trend_notes, context

    if heat >= 85 and flow >= 70 and dist_days <= 3:
        return "部分利確＋保有", "短期過熱。ただし資金流入はまだ残っています。全利確ではなく、20〜40%利確＋残りトレールが基本。", exit_count, exit_conds, pull_score, pull_label, pull_notes, trend_score, trend_label, trend_notes, context

    if heat >= 90:
        return "強い利確候補", "短期過熱は強いです。ただし全利確条件は未達。まず20〜40%の利益確定を検討。", exit_count, exit_conds, pull_score, pull_label, pull_notes, trend_score, trend_label, trend_notes, context

    if buy >= 75 and heat < 55 and flow >= 60:
        return "買い優位", "押し目買い候補。資金流入が維持されていれば分割で検討。", exit_count, exit_conds, pull_score, pull_label, pull_notes, trend_score, trend_label, trend_notes, context

    if pull_score >= 75 and heat < 70 and flow >= 60:
        return "押し目買い候補", f"{pull_label}。トレンド内の押し目として質が高いです。分割エントリー候補。", exit_count, exit_conds, pull_score, pull_label, pull_notes, trend_score, trend_label, trend_notes, context

    if buy >= 65 and heat < 70:
        return "分割買い候補", "買いすぎず分割。高値掴みを避ける。", exit_count, exit_conds, pull_score, pull_label, pull_notes, trend_score, trend_label, trend_notes, context

    if heat >= 70:
        return "一部利確", "利益の20〜30%確定を検討。残りは保有継続。", exit_count, exit_conds, pull_score, pull_label, pull_notes, trend_score, trend_label, trend_notes, context

    if buy < 45 and heat < 55:
        return "待機", "無理しない。明確な資金流入か押し目を待つ。", exit_count, exit_conds, pull_score, pull_label, pull_notes, trend_score, trend_label, trend_notes, context

    return "保有", "トレンド継続。売買せず観察。", exit_count, exit_conds, pull_score, pull_label, pull_notes, trend_score, trend_label, trend_notes, context

def action_box(action, comment, flow_text, flow_notes, dist_days, exit_count=None, exit_conds=None, pull_score=None, pull_label=None, pull_notes=None, trend_score=None, trend_label=None, trend_notes=None, context=None, market_score=None, market_label=None, market_notes=None):
    notes = " / ".join(flow_notes[:6]) if flow_notes else "-"
    cond_text = ""
    if exit_conds is not None:
        cond_rows = []
        for name, ok in exit_conds:
            mark = "✅" if ok else "—"
            cond_rows.append(f"<li>{mark} {name}</li>")
        cond_text = f"""
        <div class="small-title">全利確チェック：{exit_count}/5</div>
        <ul>{''.join(cond_rows)}</ul>
        """
    trend_text = ""
    if trend_score is not None:
        tnotes = " / ".join((trend_notes or [])[:6]) or "-"
        trend_text = f"""
        <div class="small-title">トレンド継続：{trend_score}/100｜{trend_label}</div>
        <div class="muted">局面判定：{context or '-'}<br>根拠：{tnotes}</div>
        """

    market_text = ""
    if market_score is not None:
        mnotes = " / ".join((market_notes or [])[:6]) or "-"
        market_text = f"""
        <div class="small-title">市場リバランス：{market_score}/100｜{market_label}</div>
        <div class="muted">{mnotes}</div>
        """

    pull_text = ""
    if pull_score is not None:
        pnotes = " / ".join((pull_notes or [])[:5]) or "-"
        pull_text = f"""
        <div class="small-title">押し目品質：{pull_score}/100｜{pull_label}</div>
        <div class="muted">{pnotes}</div>
        """

    bg = "#ecfdf5" if action in ["部分利確＋保有", "買い優位", "保有", "押し目買い候補"] else "#fffbeb"
    border = "#10b981" if action in ["部分利確＋保有", "買い優位", "保有", "押し目買い候補"] else "#f59e0b"
    if action == "全利確優位":
        bg, border = "#fef2f2", "#ef4444"

    st.markdown(f"""
    <style>
      .decision-card {{
        background:{bg}; border:1px solid {border}; border-radius:16px; padding:20px 22px;
        margin:12px 0 18px 0; line-height:1.75; white-space:normal; overflow-wrap:anywhere;
      }}
      .decision-title {{font-size:28px; font-weight:800; margin-bottom:8px;}}
      .small-title {{font-size:17px; font-weight:700; margin-top:12px;}}
      .muted {{color:#374151;}}
      .decision-card ul {{margin-top:4px;}}
    </style>
    <div class="decision-card">
      <div class="decision-title">最終判断：{action}</div>
      <div class="small-title">現在の需給判定</div>
      <div>{flow_text}</div>
      <div class="small-title">コメント</div>
      <div>{comment}</div>
      <div class="small-title">Distribution Day：{dist_days}日 / 直近25営業日</div>
      <div class="muted">資金フロー根拠：{notes}</div>
      {trend_text}
      {market_text}
      {cond_text}
      {pull_text}
    </div>
    """, unsafe_allow_html=True)

def health_score(d):
    latest = d.iloc[-1]
    score, notes = 50, []
    if bool(latest["AboveMA5"]): score += 5; notes.append("5MA上")
    else: score -= 5; notes.append("5MA下")
    if bool(latest["AboveMA20"]): score += 10; notes.append("20MA上")
    else: score -= 10; notes.append("20MA下")
    if bool(latest["AboveMA50"]): score += 15; notes.append("50MA上")
    else: score -= 20; notes.append("50MA下")
    if bool(latest["AboveMA200"]): score += 10; notes.append("200MA上")
    else: score -= 15; notes.append("200MA下")

    rs = latest.get("RS_QQQ_20D", np.nan)
    if pd.notna(rs):
        if rs > 10: score += 10; notes.append("QQQ比かなり強い")
        elif rs > 0: score += 5; notes.append("QQQ比プラス")
        else: score -= 5; notes.append("QQQ比マイナス")

    vol = latest.get("VolRatio20", np.nan)
    if pd.notna(vol):
        if vol > 2 and latest["Ret1D"] > 0: score += 5; notes.append("出来高上昇")
        elif vol > 2 and latest["Ret1D"] < 0: score -= 12; notes.append("出来高下落")

    ad = accumulation_distribution(d)
    dist_count, _dist_series = distribution_days(d)
    if ad["net"] >= 3: score += 5; notes.append("蓄積優勢")
    elif ad["net"] <= -3: score -= 8; notes.append("分配優勢")

    return int(max(0, min(100, score))), notes

def buy_score(d):
    latest = d.iloc[-1]
    score = 0
    reasons = []

    trend_ok = bool(latest["AboveMA50"]) and bool(latest["AboveMA200"])
    if trend_ok:
        score += 25; reasons.append("中期トレンド維持")
    if bool(latest["AboveMA20"]):
        score += 10; reasons.append("20MA上")
    dd = latest.get("Drawdown52W", np.nan)
    if pd.notna(dd):
        if -12 <= dd <= -5:
            score += 35; reasons.append("良い押し目ゾーン")
        elif -5 < dd <= -2:
            score += 20; reasons.append("浅い押し")
        elif dd < -20:
            score -= 20; reasons.append("深い下落に注意")
    rs = latest.get("RS_QQQ_20D", np.nan)
    if pd.notna(rs) and rs > 0:
        score += 15; reasons.append("QQQ比RSプラス")
    vol = latest.get("VolRatio20", np.nan)
    if pd.notna(vol) and not (vol > 2 and latest["Ret1D"] < 0):
        score += 10; reasons.append("分配的出来高ではない")
    gap5 = latest.get("GapMA5", np.nan)
    if pd.notna(gap5) and gap5 > 8:
        score -= 25; reasons.append("5MA乖離が大きく買いにくい")
    return int(max(0, min(100, score))), reasons

def sell_heat(d, gain_days=5):
    latest = d.iloc[-1]
    ret = latest.get(f"Ret{gain_days}D", np.nan)
    gap5 = latest.get("GapMA5", np.nan)
    gap20 = latest.get("GapMA20", np.nan)
    vol = latest.get("VolRatio20", np.nan)

    heat = 0
    reasons = []
    if pd.notna(ret):
        add = max(0, min(35, ret * 1.8)); heat += add
        if ret > 10: reasons.append(f"{gain_days}日で+{ret:.1f}%急騰")
    if pd.notna(gap5):
        add = max(0, min(35, gap5 * 3)); heat += add
        if gap5 > 8: reasons.append(f"5MA乖離+{gap5:.1f}%で過熱")
    if pd.notna(gap20):
        add = max(0, min(20, gap20 * 1.2)); heat += add
        if gap20 > 15: reasons.append(f"20MA乖離+{gap20:.1f}%で過熱")
    if pd.notna(vol) and vol > 2 and latest.get("Ret1D", 0) > 0:
        heat += 10; reasons.append("出来高を伴う急騰")
    heat = min(100, heat)

    if heat >= 85:
        judge = "過熱かなり強め（単独では売り判定にしない）"
    elif heat >= 70:
        judge = "過熱警戒・一部利確候補"
    elif heat >= 55:
        judge = "保有しつつ警戒"
    else:
        judge = "まだ保有優位"
    return judge, heat, reasons

def final_action(buy, heat):
    if buy >= 75 and heat < 55:
        return "買い優位", "押し目買い候補"
    if buy >= 65 and heat < 70:
        return "分割買い候補", "買いすぎず分割"
    if heat >= 85:
        return "利確優位", "強め利確候補"
    if heat >= 70:
        return "一部利確", "半分/一部を検討"
    if buy < 45 and heat < 55:
        return "待機", "無理しない"
    return "保有", "売買せず観察"

def entry_expectancy_table(d, entry_pcts=(0,3,5,8,12,15), lookahead=10, target_pct=10, stop_pct=5):
    latest_price = float(d["Close"].iloc[-1])
    rows = []
    for pct in entry_pcts:
        entry_price = latest_price * (1 - pct/100)
        mask = (d["Drawdown52W"] <= -pct) & (d["AboveMA50"].fillna(False))
        idxs = np.where(mask.fillna(False).values)[0]
        results, wins = [], 0
        for idx in idxs:
            if idx + 1 >= len(d): continue
            future = d.iloc[idx+1:min(idx+lookahead+1, len(d))]
            if future.empty: continue
            start = float(d["Close"].iloc[idx])
            max_up = (float(future["High"].max()) / start - 1) * 100
            max_down = (float(future["Low"].min()) / start - 1) * 100
            if max_up >= target_pct:
                wins += 1; results.append(target_pct)
            elif max_down <= -stop_pct:
                results.append(-stop_pct)
            else:
                results.append((float(future["Close"].iloc[-1]) / start - 1) * 100)
        n = len(results)
        rows.append({
            "価格": entry_price,
            "現在から": f"-{pct}%",
            "勝率": wins/n*100 if n else np.nan,
            "期待値": float(np.mean(results)) if n else np.nan,
            "サンプル数": n,
            "判定": ""
        })
    out = pd.DataFrame(rows)
    if out["期待値"].notna().any():
        out.loc[out["期待値"].idxmax(), "判定"] = "←最高"
    return out

def ma_gap_profit_stats(d, ma=5, thresholds=(3,5,8,10,12,15), lookahead=10, min_gap=5):
    gap_col = f"GapMA{ma}"
    rows = []
    for th in thresholds:
        idxs = np.where((d[gap_col] >= th).fillna(False).values)[0]
        vals, pulls, days_to_high, days_to_low = [], [], [], []
        last_idx, count = -10**9, 0
        for idx in idxs:
            if idx - last_idx < min_gap: continue
            if idx + 1 >= len(d): continue
            future = d.iloc[idx+1:min(idx+lookahead+1, len(d))]
            if future.empty: continue
            close = float(d["Close"].iloc[idx])
            max_up = (float(future["High"].max()) / close - 1) * 100
            min_down = (float(future["Low"].min()) / close - 1) * 100
            high_date = future["High"].idxmax()
            low_date = future["Low"].idxmin()
            vals.append(max_up); pulls.append(min_down)
            days_to_high.append(int(future.index.get_loc(high_date)+1))
            days_to_low.append(int(future.index.get_loc(low_date)+1))
            count += 1; last_idx = idx
        avg_up = np.mean(vals) if vals else np.nan
        avg_pull = np.mean(pulls) if pulls else np.nan
        if count == 0:
            judge = "-"
        elif pd.notna(avg_up) and pd.notna(avg_pull) and avg_up <= 3 and abs(avg_pull) >= 5:
            judge = "利確強め"
        elif pd.notna(avg_up) and pd.notna(avg_pull) and avg_up <= 5 and abs(avg_pull) >= 5:
            judge = "一部利確"
        elif pd.notna(avg_up) and avg_up > abs(avg_pull):
            judge = "保有余地"
        else:
            judge = "警戒"
        rows.append({
            f"{ma}MA乖離条件": f"+{th}%以上",
            "件数": count,
            f"{lookahead}日内 平均追加上昇%": avg_up,
            f"{lookahead}日内 平均最大押し%": avg_pull,
            "平均高値日数": np.mean(days_to_high) if days_to_high else np.nan,
            "平均押し日数": np.mean(days_to_low) if days_to_low else np.nan,
            "利確判定": judge
        })
    return pd.DataFrame(rows)

def surge_pullback_events(d, surge_pct=10, surge_days=5, lookahead=10, min_gap=5):
    ret_col = f"Ret{surge_days}D"
    candidates = np.where((d[ret_col] >= surge_pct).fillna(False).values)[0]
    events, last_idx = [], -10**9
    for idx in candidates:
        if idx - last_idx < min_gap: continue
        if idx + 1 >= len(d): continue
        future = d.iloc[idx+1:min(idx+lookahead+1, len(d))]
        if future.empty: continue
        signal_close = float(d["Close"].iloc[idx])
        min_low, max_high = float(future["Low"].min()), float(future["High"].max())
        min_date, max_date = future["Low"].idxmin(), future["High"].idxmax()
        events.append({
            "シグナル日": d.index[idx].strftime("%Y-%m-%d"),
            f"{surge_days}日上昇率%": float(d[ret_col].iloc[idx]),
            "シグナル終値": signal_close,
            f"{lookahead}日内最大押し%": (min_low/signal_close-1)*100,
            "押しまでの日数": int(future.index.get_loc(min_date)+1),
            f"{lookahead}日内最大上昇%": (max_high/signal_close-1)*100,
            "高値までの日数": int(future.index.get_loc(max_date)+1),
            "5MA乖離%": float(d["GapMA5"].iloc[idx]) if pd.notna(d["GapMA5"].iloc[idx]) else np.nan,
        })
        last_idx = idx
    return pd.DataFrame(events)

def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    return f"{v:.{nd}f}"

with st.sidebar:
    st.header("設定")
    tickers_text = st.text_input("取得銘柄", "ALAB MUU MRVL NVDA CRDO LITE COHR AAOI QQQ SPY SOXX SOXL TLT IEF GLD ^TNX IWM")
    tickers = [x.strip().upper() for x in tickers_text.split() if x.strip()]
    period = st.selectbox("取得期間", ["1y", "2y", "5y"], index=1)
    selected = st.selectbox("表示銘柄", tickers, index=0)

    st.header("売買判定設定")
    surge_days = st.selectbox("短期騰落期間", [1,3,5,10,20], index=2)
    lookahead = st.selectbox("検証する未来日数", [3,5,10,15,20], index=2)
    min_gap = st.selectbox("同シグナル間隔", [3,5,10,15], index=1)
    target_pct = st.number_input("期待値：利確幅%", value=10.0, step=1.0)
    stop_pct = st.number_input("期待値：損切幅%", value=5.0, step=1.0)

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

buy, buy_reasons = buy_score(d)
sell_judge, heat, sell_reasons = sell_heat(d, gain_days=surge_days)
flow, flow_notes, dist_days = capital_flow_score(d)
market_score, market_label, market_notes, market_details = market_rebalance_score(data, selected)
macro_score, macro_label, macro_notes, macro_details = macro_risk_score(data)
flow_text = flow_label(flow, heat, dist_days)
action, action_note, exit_count, exit_conds, pull_score, pull_label, pull_notes, trend_score, trend_label, trend_notes, context = final_action_v8(d, buy, heat, flow, dist_days, market_score)
health, health_notes = health_score(d)

c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = st.columns(10)
c1.metric("最終行動", action)
c2.metric("買いスコア", f"{buy}/100")
c3.metric("利確過熱度", f"{heat:.0f}/100", sell_judge)
c4.metric("健康スコア", f"{health}/100")
c5.metric("資金流入", f"{flow}/100")
c6.metric("Distribution", f"{dist_days}日")
c7.metric("押し目品質", f"{pull_score}/100", pull_label)
c8.metric("トレンド継続", f"{trend_score}/100", trend_label)
c9.metric("市場リバランス", f"{market_score}/100", market_label)
c10.metric("マクロ地合い", f"{macro_score}/100", macro_label)
action_box(action, action_note, flow_text, flow_notes, dist_days, exit_count, exit_conds, pull_score, pull_label, pull_notes, trend_score, trend_label, trend_notes, context, market_score, market_label, market_notes)
st.metric("現在値 / 5MA乖離", f"{latest['Close']:.2f}", f"{latest['GapMA5']:.2f}%")

st.markdown("### 売買マトリクス")
matrix = pd.DataFrame([
    ["買い", buy, "75以上=買い優位 / 65以上=分割買い候補", " / ".join(buy_reasons) if buy_reasons else "-"],
    ["過熱度", heat, "5MA乖離・急騰は警戒シグナル。単独では全利確にしない", " / ".join(sell_reasons) if sell_reasons else "-"],
    ["資金流入", flow, "75以上=機関買い継続 / 60未満=鈍化", " / ".join(flow_notes) if flow_notes else "-"],
    ["Distribution", dist_days, "0〜1日=問題小 / 2〜3日=警戒 / 4日以上=機関売り増加", flow_text],
    ["市場リバランス", market_score, "75以上=市場起因の押し目 / 45未満=個別悪化注意", market_label + "：" + (" / ".join(market_notes[:5]) if market_notes else "-")],
    ["マクロ地合い", macro_score, "75以上=リスクオン / 45未満=金利・リスクオフ警戒", macro_label + "：" + (" / ".join(macro_notes[:5]) if macro_notes else "-")],
    ["押し目品質", pull_score, "75以上=S/A級押し目 / 50未満=追いかけ注意", pull_label + "：" + (" / ".join(pull_notes[:5]) if pull_notes else "-")],
    ["トレンド継続", trend_score, "75以上=過熱でも保有優位 / 60未満=鈍化", trend_label + "：" + (" / ".join(trend_notes[:5]) if trend_notes else "-")],
    ["局面判定", "-", "イベント出尽くし型かトレンド型か", context],
    ["全利確条件", exit_count, "5条件中4つ以上で全利確優位", " / ".join([("✅" if ok else "—") + name for name, ok in exit_conds])],
], columns=["項目", "スコア", "判定基準", "根拠"])
st.dataframe(matrix, use_container_width=True, hide_index=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["同時判定", "買い期待値", "利確統計", "銘柄カルテ", "チャート", "比較ランキング"])

with tab1:
    st.subheader("買いポイントと利確ポイントを同時表示")
    st.markdown("#### 市場全体のリバランス判定")
    md = pd.DataFrame([[k, v] for k, v in market_details.items()], columns=["指標", "値"])
    if not md.empty:
        st.dataframe(md, use_container_width=True, hide_index=True)
    st.info(f"{market_label}：" + (" / ".join(market_notes[:6]) if market_notes else "判定材料不足"))
    st.markdown("#### マクロ地合い：S&P500 / Tech100 / 国債 / 金 / 長期金利")
    mac = pd.DataFrame([[k, v] for k, v in macro_details.items()], columns=["指標", "値"])
    if not mac.empty:
        st.dataframe(mac, use_container_width=True, hide_index=True)
    st.info(f"{macro_label}：" + (" / ".join(macro_notes[:6]) if macro_notes else "判定材料不足"))
    price = float(latest["Close"])
    buy_lines = [[f"{pct}%押し", price*(1-pct/100)] for pct in [3,5,8,12,15]]
    sell_lines = [[f"+{pct}%利確", price*(1+pct/100)] for pct in [5,8,10,15,20]]
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 買い候補ライン")
        st.dataframe(pd.DataFrame(buy_lines, columns=["条件", "価格"]), use_container_width=True, hide_index=True)
    with colB:
        st.markdown("#### 利確候補ライン")
        st.dataframe(pd.DataFrame(sell_lines, columns=["条件", "価格"]), use_container_width=True, hide_index=True)

    if action in ["買い優位", "分割買い候補", "押し目買い候補"]:
        st.success(f"判定：{action}。買いスコアが高く、利確過熱度はまだ高すぎません。")
    elif action in ["全利確優位", "強い利確候補", "一部利確", "部分利確＋保有"]:
        st.warning(f"判定：{action}。{action_note}")
    else:
        st.info(f"判定：{action}。今は無理に動かず観察優位です。")

with tab2:
    st.subheader("今買う / 押しを待つ期待値")
    et = entry_expectancy_table(d, entry_pcts=(0,3,5,8,12,15), lookahead=lookahead, target_pct=target_pct, stop_pct=stop_pct)
    show = et.copy()
    show["勝率"] = show["勝率"].map(lambda x: "-" if pd.isna(x) else f"{x:.0f}%")
    show["期待値"] = show["期待値"].map(lambda x: "-" if pd.isna(x) else f"{x:+.1f}%")
    show["価格"] = show["価格"].map(lambda x: f"{x:.2f}")
    st.dataframe(show, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("5MA乖離率で見る過熱統計（単独では売り判定にしない）")
    gap_stats = ma_gap_profit_stats(d, ma=5, thresholds=(3,5,8,10,12,15), lookahead=lookahead, min_gap=min_gap)
    st.dataframe(gap_stats, use_container_width=True, hide_index=True)
    st.markdown("#### 急騰後イベント")
    events = surge_pullback_events(d, surge_pct=10, surge_days=surge_days, lookahead=lookahead, min_gap=min_gap)
    st.dataframe(events.sort_values("シグナル日", ascending=False) if not events.empty else events, use_container_width=True, hide_index=True)

with tab4:
    st.subheader(f"{selected} 銘柄カルテ")
    m5, m20, m50 = ma_break_stats(d, "MA5"), ma_break_stats(d, "MA20"), ma_break_stats(d, "MA50")
    ad = accumulation_distribution(d)
    dist_count, _dist_series = distribution_days(d)
    rows = [
        ["終値", latest["Close"]],
        ["5MA", latest["MA5"]],
        ["20MA", latest["MA20"]],
        ["50MA", latest["MA50"]],
        ["200MA", latest["MA200"]],
        ["5MA乖離%", latest["GapMA5"]],
        ["20MA乖離%", latest["GapMA20"]],
        ["50MA乖離%", latest["GapMA50"]],
        ["出来高倍率", latest["VolRatio20"]],
        ["ATR14%", latest["ATR14Pct"]],
        ["52週高値から%", latest["Drawdown52W"]],
        ["QQQ比RS20日", latest["RS_QQQ_20D"]],
        ["5MA割れ回数", m5["breaks"]],
        ["5MA上連続日数", m5["current_above_streak"]],
        ["20MA割れ回数", m20["breaks"]],
        ["20MA上連続日数", m20["current_above_streak"]],
        ["20MA割れ後 平均回復日数", m20["avg_recovery"]],
        ["50MA割れ回数", m50["breaks"]],
        ["50MA上連続日数", m50["current_above_streak"]],
        ["蓄積日", ad["acc"]],
        ["分配日", ad["dist"]],
        ["Distribution Day(25日)", dist_count],
        ["資金流入スコア", flow],
        ["市場リバランススコア", market_score],
        ["市場リバランス", market_label],
        ["マクロ地合いスコア", macro_score],
        ["マクロ地合い", macro_label],
        ["押し目品質スコア", pull_score],
        ["押し目品質", pull_label],
        ["トレンド継続スコア", trend_score],
        ["トレンド継続", trend_label],
        ["局面判定", context],
        ["全利確条件", f"{exit_count}/5"],
    ]
    st.dataframe(pd.DataFrame(rows, columns=["項目", "値"]), use_container_width=True, hide_index=True)

with tab5:
    lookback_chart = st.selectbox("チャート期間", [90, 180, 365, 730], index=2)
    x = d.tail(lookback_chart)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=x.index, open=x["Open"], high=x["High"], low=x["Low"], close=x["Close"], name=selected))
    fig.add_trace(go.Scatter(x=x.index, y=x["MA5"], name="5MA"))
    fig.add_trace(go.Scatter(x=x.index, y=x["MA20"], name="20MA"))
    fig.add_trace(go.Scatter(x=x.index, y=x["MA50"], name="50MA"))
    fig.add_trace(go.Scatter(x=x.index, y=x["MA200"], name="200MA"))
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("比較ランキング")
    ranks = []
    for t, df0 in data.items():
        if t in ["QQQ", "SOXX", "SPY", "TLT", "IEF", "GLD", "GC=F", "^TNX", "TNX"]: continue
        dd = df0.dropna(subset=["Close"])
        if dd.empty: continue
        b, _ = buy_score(dd)
        sj, ht, _ = sell_heat(dd, gain_days=surge_days)
        fl, fl_notes, ds = capital_flow_score(dd)
        ms, ml, mn, md = market_rebalance_score(data, t)
        act, note, exn, exc, ps, pl, pn, ts, tl, tn, cx = final_action_v8(dd, b, ht, fl, ds, ms)
        h, _ = health_score(dd)
        l = dd.iloc[-1]
        ranks.append({
            "銘柄": t,
            "最終行動": act,
            "買いスコア": b,
            "利確過熱度": ht,
            "利確判定": sj,
            "健康スコア": h,
            "資金流入": fl,
            "Distribution": ds,
            "市場リバランス": ms,
            "市場判定": ml,
            "マクロ地合い": macro_score,
            "マクロ判定": macro_label,
            "押し目品質": ps,
            "トレンド継続": ts,
            "局面": cx,
            "全利確条件": exn,
            "終値": l["Close"],
            "5MA乖離%": l["GapMA5"],
            "20MA乖離%": l["GapMA20"],
            f"{surge_days}日%": l.get(f"Ret{surge_days}D", np.nan),
            "QQQ比RS20日": l["RS_QQQ_20D"],
            "出来高倍率": l["VolRatio20"],
        })
    st.dataframe(pd.DataFrame(ranks).sort_values(["買いスコア", "健康スコア"], ascending=False), use_container_width=True, hide_index=True)

st.caption("投資判断補助ツールです。全利確優位は「過熱＋資金流出＋Distribution Day＋20MA割れ＋トレンド鈍化」の複合条件でのみ出します。5MA乖離率は警戒シグナルであり、単独の売却理由ではありません。押し目買いは、市場リバランス/指数連動売りと個別悪化を分け、S&P500・Tech100・国債・金・長期金利でマクロ地合いも確認します。売買を保証するものではありません。")
