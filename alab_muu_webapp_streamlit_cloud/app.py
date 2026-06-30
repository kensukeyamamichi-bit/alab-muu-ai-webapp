
import math
from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="ALAB/MUU Expectancy AI v3", layout="wide")
st.markdown("""
# ALAB/MUU Expectancy AI v3
**買い期待値・利益確定ポイント・急騰後の押し統計**をまとめて見るアプリ。  
目的：恐怖やノイズではなく、銘柄特性と期待値で「買う・保有・売る」を判断する。
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
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev_close).abs(), (d["Low"] - prev_close).abs()], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()
    d["ATR14Pct"] = d["ATR14"] / d["Close"] * 100
    for n in [1, 3, 5, 10, 20]:
        d[f"Ret{n}D"] = d["Close"].pct_change(n) * 100
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
        d["RS_QQQ_20D"] = np.nan; d["RS_QQQ_5D"] = np.nan
    if soxx is not None and not soxx.empty:
        s = soxx["Close"].reindex(d.index).ffill()
        d["RS_SOXX_20D"] = d["Ret20D"] - s.pct_change(20) * 100
        d["RS_SOXX_5D"] = d["Ret5D"] - s.pct_change(5) * 100
    else:
        d["RS_SOXX_20D"] = np.nan; d["RS_SOXX_5D"] = np.nan
    return d

def streak_bool(s):
    n = 0
    for v in s.dropna().astype(bool).tolist()[::-1]:
        if v: n += 1
        else: break
    return n

def ma_break_stats(d, ma_col):
    above = d["Close"] > d[ma_col]
    breaks = (~above) & (above.shift(1) == True) & d[ma_col].notna()
    recovery_days, failed = [], 0
    for idx in np.where(breaks.values)[0]:
        ok = False
        for j in range(idx + 1, min(idx + 31, len(d))):
            if d["Close"].iloc[j] > d[ma_col].iloc[j]:
                recovery_days.append(j - idx); ok = True; break
        if not ok: failed += 1
    return {"breaks": int(breaks.sum()), "current_above_streak": streak_bool(above), "avg_recovery": float(np.mean(recovery_days)) if recovery_days else None, "within_3d": int(sum(x <= 3 for x in recovery_days)), "within_5d": int(sum(x <= 5 for x in recovery_days)), "failed_30d": failed}

def accumulation_distribution(d, lookback=20):
    x = d.tail(lookback)
    vol_up = x["Volume"] > x["Volume"].shift(1)
    acc = (x["Close"] > x["Close"].shift(1)) & vol_up
    dist = (x["Close"] < x["Close"].shift(1)) & vol_up
    return {"acc": int(acc.sum()), "dist": int(dist.sum()), "net": int(acc.sum() - dist.sum())}

def pullback_stats(d):
    trend = d[d["AboveMA50"].fillna(False)].copy()
    if trend.empty: return {"avg": None, "median": None, "max": None}
    trend["Peak"] = trend["Close"].cummax()
    trend["DD"] = (trend["Close"] / trend["Peak"] - 1) * 100
    p = trend["DD"][trend["DD"] < -1]
    return {"avg": float(p.mean()) if not p.empty else 0.0, "median": float(p.median()) if not p.empty else 0.0, "max": float(p.min()) if not p.empty else 0.0}

def high_after_pullback(d, window=20):
    idxs = list(np.where(d["Is52WHigh"].fillna(False).values)[0])
    vals = []
    for idx in idxs:
        future = d.iloc[idx:min(idx + window + 1, len(d))]
        if len(future) > 1:
            vals.append((future["Low"].min() / d["Close"].iloc[idx] - 1) * 100)
    return {"count": len(idxs), "avg": float(np.mean(vals)) if vals else None, "median": float(np.median(vals)) if vals else None, "max": float(np.min(vals)) if vals else None}

def health_score(d):
    latest = d.iloc[-1]; score = 50; notes = []
    for col, pts, name in [("AboveMA20",10,"20MA上"),("AboveMA50",15,"50MA上"),("AboveMA200",10,"200MA上")]:
        if bool(latest[col]): score += pts; notes.append(name)
        else: score -= pts; notes.append(name.replace("上","下"))
    rs = latest.get("RS_QQQ_20D", np.nan)
    if pd.notna(rs):
        if rs > 10: score += 10; notes.append("QQQ比かなり強い")
        elif rs > 0: score += 5; notes.append("QQQ比プラス")
        else: score -= 5; notes.append("QQQ比マイナス")
    vol = latest.get("VolRatio20", np.nan)
    if pd.notna(vol):
        if vol > 2 and latest["Ret1D"] > 0: score += 5; notes.append("出来高を伴う上昇")
        elif vol > 2 and latest["Ret1D"] < 0: score -= 12; notes.append("出来高を伴う下落")
    dd = latest.get("Drawdown52W", np.nan)
    if pd.notna(dd):
        if dd > -5: score += 5; notes.append("高値圏維持")
        elif dd < -20: score -= 10; notes.append("高値から大きく下落")
    ad = accumulation_distribution(d)
    if ad["net"] >= 3: score += 5; notes.append("蓄積優勢")
    elif ad["net"] <= -3: score -= 8; notes.append("分配優勢")
    return int(max(0, min(100, score))), notes

def ai_decision(d):
    latest = d.iloc[-1]; score, notes = health_score(d)
    dd52 = latest.get("Drawdown52W", np.nan); vol = latest.get("VolRatio20", np.nan); rs = latest.get("RS_QQQ_20D", np.nan)
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
    reasons = ["50MA・200MA上で中期トレンド維持" if trend_ok else "中期トレンドに傷あり"]
    if pd.notna(rs): reasons.append(f"QQQ比RS20日: {rs:.1f}%")
    if pd.notna(vol): reasons.append(f"出来高倍率: {vol:.2f}倍")
    if pd.notna(dd52): reasons.append(f"52週高値から: {dd52:.1f}%")
    return {"decision": decision, "stars": stars, "action": action, "score": score, "notes": notes, "reasons": reasons}

def surge_pullback_events(d, surge_pct=10, surge_days=5, lookahead=10, min_gap=5):
    ret_col = f"Ret{surge_days}D"
    candidates = np.where((d[ret_col] >= surge_pct).fillna(False).values)[0]
    events = []; last_idx = -10**9
    for idx in candidates:
        if idx - last_idx < min_gap or idx + 1 >= len(d): continue
        future = d.iloc[idx+1:min(idx+lookahead+1, len(d))]
        if future.empty: continue
        signal_close = float(d["Close"].iloc[idx]); min_low = float(future["Low"].min()); max_high = float(future["High"].max())
        min_date = future["Low"].idxmin(); max_date = future["High"].idxmax()
        events.append({"シグナル日": d.index[idx].strftime("%Y-%m-%d"), f"{surge_days}日上昇率%": float(d[ret_col].iloc[idx]), "シグナル終値": signal_close, f"{lookahead}日内最大押し%": (min_low/signal_close-1)*100, "押し安値": min_low, "押しまでの日数": int(future.index.get_loc(min_date)+1), f"{lookahead}日内最大上昇%": (max_high/signal_close-1)*100, "高値までの日数": int(future.index.get_loc(max_date)+1), "出来高倍率": float(d["VolRatio20"].iloc[idx]) if pd.notna(d["VolRatio20"].iloc[idx]) else np.nan, "20MA乖離%": float((d["Close"].iloc[idx]/d["MA20"].iloc[idx]-1)*100) if pd.notna(d["MA20"].iloc[idx]) else np.nan})
        last_idx = idx
    return pd.DataFrame(events)

def summarize_surge_events(events):
    if events.empty: return pd.DataFrame(), {}
    col = [c for c in events.columns if "最大押し%" in c][0]
    summary = {"件数": len(events), "平均押し%": events[col].mean(), "中央値押し%": events[col].median(), "最大押し%": events[col].min(), "平均押し日数": events["押しまでの日数"].mean(), "中央値押し日数": events["押しまでの日数"].median()}
    rows=[]
    for th in [5,8,12]:
        hit = events[col] <= -th
        rows.append({"押し条件": f"{th}%以上押す", "発生率": hit.mean()*100, "回数": int(hit.sum()), "平均押し日数": events.loc[hit,"押しまでの日数"].mean() if hit.any() else np.nan})
    return pd.DataFrame(rows), summary

def entry_expectancy_table(d, entry_pcts=(0,3,5,8,12,15), lookahead=10, target_pct=10, stop_pct=5):
    latest_price = float(d["Close"].iloc[-1]); rows=[]
    for pct in entry_pcts:
        entry_price = latest_price*(1-pct/100)
        mask = (d["Drawdown52W"] <= -pct) & d["AboveMA50"].fillna(False)
        idxs=np.where(mask.fillna(False).values)[0]; results=[]; wins=0
        for idx in idxs:
            if idx+1>=len(d): continue
            future=d.iloc[idx+1:min(idx+lookahead+1,len(d))]
            if future.empty: continue
            start=float(d["Close"].iloc[idx]); max_up=(float(future["High"].max())/start-1)*100; max_down=(float(future["Low"].min())/start-1)*100
            if max_up>=target_pct: wins+=1; results.append(target_pct)
            elif max_down<=-stop_pct: results.append(-stop_pct)
            else: results.append((float(future["Close"].iloc[-1])/start-1)*100)
        n=len(results); rows.append({"価格": entry_price, "現在から": f"-{pct}%", "勝率": wins/n*100 if n else np.nan, "期待値": float(np.mean(results)) if n else np.nan, "サンプル数": n, "判定": ""})
    out=pd.DataFrame(rows)
    if out["期待値"].notna().any(): out.loc[out["期待値"].idxmax(),"判定"]="←最高"
    return out

def profit_taking_table(d, gain_levels=(5,10,15,20,25,30), gain_days=5, lookahead=10, min_gap=5):
    rows=[]
    for g in gain_levels:
        events=surge_pullback_events(d, surge_pct=g, surge_days=gain_days, lookahead=lookahead, min_gap=min_gap)
        if events.empty:
            rows.append({"急騰条件": f"{gain_days}日で+{g}%", "件数":0, "平均追加上昇%":np.nan, "平均最大押し%":np.nan, "平均高値日数":np.nan, "利確判定":"-"}); continue
        up_col=[c for c in events.columns if "最大上昇%" in c][0]; pull_col=[c for c in events.columns if "最大押し%" in c][0]
        avg_up=events[up_col].mean(); avg_pull=events[pull_col].mean(); avg_high=events["高値までの日数"].mean()
        judge="利確強め" if avg_up<=3 and abs(avg_pull)>=6 else "一部利確" if avg_up<=5 and abs(avg_pull)>=5 else "保有余地"
        rows.append({"急騰条件": f"{gain_days}日で+{g}%", "件数":len(events), "平均追加上昇%":avg_up, "平均最大押し%":avg_pull, "平均高値日数":avg_high, "利確判定":judge})
    return pd.DataFrame(rows)

def profit_signal(d, gain_days=5):
    latest=d.iloc[-1]; price=float(latest["Close"]); ret=latest.get(f"Ret{gain_days}D",np.nan); ma20_gap=(price/latest["MA20"]-1)*100 if pd.notna(latest["MA20"]) else np.nan; dd52=latest.get("Drawdown52W",np.nan)
    heat=0
    if pd.notna(ret): heat += max(0,min(40,ret*2))
    if pd.notna(ma20_gap): heat += max(0,min(35,ma20_gap*2))
    if pd.notna(dd52) and dd52>-2: heat += 15
    if latest.get("VolRatio20",np.nan)>2 and latest.get("Ret1D",0)>0: heat += 10
    heat=min(100,heat)
    judge="利確かなり強め" if heat>=85 else "一部利確候補" if heat>=70 else "保有しつつ警戒" if heat>=55 else "まだ保有優位"
    return judge, heat

def fmt(v, nd=2):
    if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return "-"
    return f"{v:.{nd}f}"

with st.sidebar:
    st.header("設定")
    tickers_text=st.text_input("取得銘柄","ALAB MUU MRVL NVDA CRDO QQQ SOXX SOXL")
    tickers=[x.strip().upper() for x in tickers_text.split() if x.strip()]
    period=st.selectbox("取得期間",["1y","2y","5y"],index=1)
    selected=st.selectbox("表示銘柄",tickers,index=0)
    st.header("Swing統計")
    surge_pct=st.number_input("急騰判定：何％以上",value=10.0,step=1.0)
    surge_days=st.selectbox("急騰期間：何日で",[1,3,5,10,20],index=2)
    lookahead=st.selectbox("その後何日を見る",[3,5,10,15,20],index=2)
    min_gap=st.selectbox("同じ急騰を何日あけて数える",[3,5,10,15],index=1)
    st.header("期待値設定")
    target_pct=st.number_input("期待値計算：利確幅%",value=10.0,step=1.0)
    stop_pct=st.number_input("期待値計算：損切幅%",value=5.0,step=1.0)
    if st.button("最新データ取得"): st.cache_data.clear()

try:
    raw=download_data(tickers,period)
except Exception as e:
    st.error("データ取得でエラーが出ました。少し時間を置いて再読み込みしてください。"); st.code(str(e)); st.stop()
if selected not in raw or raw[selected].empty:
    st.error(f"{selected} のデータが取れませんでした。"); st.stop()
qqq,soxx=raw.get("QQQ"),raw.get("SOXX")
data={t:add_indicators(df,qqq=qqq,soxx=soxx) for t,df in raw.items()}
d=data[selected].dropna(subset=["Close"]).copy(); latest=d.iloc[-1]
ai=ai_decision(d); profit_judge,heat=profit_signal(d,gain_days=surge_days)

c1,c2,c3,c4,c5=st.columns(5)
c1.metric("AI総合判定",ai["decision"],ai["stars"]); c2.metric("利確判定",profit_judge); c3.metric("過熱度",f"{heat:.0f}/100"); c4.metric("健康スコア",f"{ai['score']}/100"); c5.metric("現在値",f"{latest['Close']:.2f}")
st.write("理由： "+" / ".join(ai["reasons"]))

tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["期待値テーブル","利益確定","急騰後統計","銘柄カルテ","チャート","比較ランキング"])
with tab1:
    st.subheader("今買う / 押しを待つ期待値")
    et=entry_expectancy_table(d,entry_pcts=(0,3,5,8,12,15),lookahead=lookahead,target_pct=target_pct,stop_pct=stop_pct)
    show=et.copy(); show["勝率"]=show["勝率"].map(lambda x:"-" if pd.isna(x) else f"{x:.0f}%"); show["期待値"]=show["期待値"].map(lambda x:"-" if pd.isna(x) else f"{x:+.1f}%"); show["価格"]=show["価格"].map(lambda x:f"{x:.2f}")
    st.dataframe(show,use_container_width=True,hide_index=True)
    if et["期待値"].notna().any():
        best=et.loc[et["期待値"].idxmax()]
        st.success(f"最適候補：{best['現在から']} 押し / 価格 {best['価格']:.2f} / 期待値 {best['期待値']:+.1f}% / 勝率 {best['勝率']:.0f}%")
with tab2:
    st.subheader("AI利益確定ポイント")
    pt=profit_taking_table(d,gain_levels=(5,10,15,20,25,30),gain_days=surge_days,lookahead=lookahead,min_gap=min_gap)
    st.dataframe(pt,use_container_width=True,hide_index=True)
    current_ret=latest.get(f"Ret{surge_days}D",np.nan)
    if profit_judge in ["利確かなり強め","一部利確候補"]: st.warning(f"{surge_days}日騰落率 {fmt(current_ret)}%。過熱度 {heat:.0f}/100。利確または一部利確を検討する局面。")
    elif profit_judge=="保有しつつ警戒": st.info(f"{surge_days}日騰落率 {fmt(current_ret)}%。過熱度 {heat:.0f}/100。まだ保有余地はあるが警戒。")
    else: st.success(f"{surge_days}日騰落率 {fmt(current_ret)}%。過熱度 {heat:.0f}/100。急いで利確する根拠は弱い。")
with tab3:
    events=surge_pullback_events(d,surge_pct=surge_pct,surge_days=surge_days,lookahead=lookahead,min_gap=min_gap); dist,summary=summarize_surge_events(events)
    if events.empty: st.warning("この条件に合う急騰イベントがありません。条件を緩めてください。")
    else:
        m1,m2,m3,m4=st.columns(4); m1.metric("急騰イベント数",summary["件数"]); m2.metric("急騰後の平均押し",f"{summary['平均押し%']:.2f}%"); m3.metric("最大押し",f"{summary['最大押し%']:.2f}%"); m4.metric("平均何日で押す",f"{summary['平均押し日数']:.1f}日")
        st.dataframe(dist,use_container_width=True,hide_index=True); st.dataframe(events.sort_values("シグナル日",ascending=False),use_container_width=True,hide_index=True)
with tab4:
    m20=ma_break_stats(d,"MA20"); m50=ma_break_stats(d,"MA50"); p=pullback_stats(d); h=high_after_pullback(d); ad=accumulation_distribution(d)
    rows=[["終値",latest["Close"]],["20MA",latest["MA20"]],["50MA",latest["MA50"]],["200MA",latest["MA200"]],["20MA乖離%",(latest["Close"]/latest["MA20"]-1)*100 if pd.notna(latest["MA20"]) else np.nan],["50MA乖離%",(latest["Close"]/latest["MA50"]-1)*100 if pd.notna(latest["MA50"]) else np.nan],["出来高倍率",latest["VolRatio20"]],["ATR14%",latest["ATR14Pct"]],["52週高値から%",latest["Drawdown52W"]],["QQQ比RS 20日",latest["RS_QQQ_20D"]],["SOXX比RS 20日",latest["RS_SOXX_20D"]],["20MA割れ回数",m20["breaks"]],["20MA上連続日数",m20["current_above_streak"]],["20MA割れ後 平均回復日数",m20["avg_recovery"]],["50MA割れ回数",m50["breaks"]],["50MA上連続日数",m50["current_above_streak"]],["平均押し%",p["avg"]],["最大押し%",p["max"]],["52週高値更新後 平均押し%",h["avg"]],["蓄積日",ad["acc"]],["分配日",ad["dist"]]]
    st.dataframe(pd.DataFrame(rows,columns=["項目","値"]),use_container_width=True,hide_index=True)
with tab5:
    lookback_chart=st.selectbox("チャート期間",[90,180,365,730],index=2); x=d.tail(lookback_chart)
    fig=go.Figure(); fig.add_trace(go.Candlestick(x=x.index,open=x["Open"],high=x["High"],low=x["Low"],close=x["Close"],name=selected)); fig.add_trace(go.Scatter(x=x.index,y=x["MA20"],name="20MA")); fig.add_trace(go.Scatter(x=x.index,y=x["MA50"],name="50MA")); fig.add_trace(go.Scatter(x=x.index,y=x["MA200"],name="200MA")); fig.update_layout(height=560,xaxis_rangeslider_visible=False,margin=dict(l=20,r=20,t=20,b=20)); st.plotly_chart(fig,use_container_width=True)
with tab6:
    ranks=[]
    for t,df0 in data.items():
        if t in ["QQQ","SOXX"]: continue
        dd=df0.dropna(subset=["Close"])
        if dd.empty: continue
        a=ai_decision(dd); pj,ht=profit_signal(dd,gain_days=surge_days); l=dd.iloc[-1]; et0=entry_expectancy_table(dd,entry_pcts=(0,3,5,8,12),lookahead=lookahead,target_pct=target_pct,stop_pct=stop_pct); best_ev=et0["期待値"].max() if et0["期待値"].notna().any() else np.nan
        ranks.append({"銘柄":t,"AI判定":a["decision"],"利確判定":pj,"過熱度":ht,"健康スコア":a["score"],"期待値最大%":best_ev,"終値":l["Close"],f"{surge_days}日%":l.get(f"Ret{surge_days}D",np.nan),"QQQ比RS20日":l["RS_QQQ_20D"],"出来高倍率":l["VolRatio20"],"52週高値から%":l["Drawdown52W"]})
    st.dataframe(pd.DataFrame(ranks).sort_values(["期待値最大%","健康スコア"],ascending=False),use_container_width=True,hide_index=True)
st.caption("投資判断補助ツールです。売買を保証するものではありません。")
