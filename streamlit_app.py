import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg
from psycopg.rows import dict_row
import streamlit as st

IST=ZoneInfo("Asia/Kolkata")
DB=os.getenv("NEON_DATABASE_URL","").strip()
if not DB:
    st.error("NEON_DATABASE_URL is missing")
    st.stop()

st.set_page_config(page_title="Index Early Detector", layout="wide")

@st.cache_data(ttl=20)
def q(sql,params=None):
    with psycopg.connect(DB,row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql,params or ())
            return pd.DataFrame(cur.fetchall())

def tist(x):
    if pd.isna(x): return None
    return pd.Timestamp(x).tz_convert("Asia/Kolkata") if pd.Timestamp(x).tzinfo else pd.Timestamp(x).tz_localize("UTC").tz_convert("Asia/Kolkata")

def tail_count(vals, pred):
    n=0
    for v in reversed(list(vals)):
        try: ok=pred(float(v))
        except Exception: ok=False
        if not ok: break
        n+=1
    return n

def option_points(score,persist):
    p=3 if score>=6 else 2.5 if score>=5 else 2 if score>=4 else 1 if score>=3 else 0
    return min(3,p+(0.5 if persist>=3 else 0))

@st.cache_data(ttl=20)
def load_all():
    date_df=q("SELECT MAX(trading_date) AS d FROM public.index_engine_snapshots")
    if date_df.empty or pd.isna(date_df.iloc[0]["d"]):
        return None,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    d=date_df.iloc[0]["d"]
    eng=q("SELECT * FROM public.index_engine_snapshots WHERE trading_date=%s ORDER BY symbol,ts",(d,))
    opt=q("SELECT * FROM public.index_option_snapshots WHERE trading_date=%s ORDER BY symbol,ts,option_type,wing_no",(d,))
    agg=q("SELECT * FROM public.index_futures_aggression_snapshots WHERE trading_date=%s ORDER BY symbol,ts",(d,))
    uni=q("SELECT * FROM public.index_money_flow_universe WHERE trading_date=%s ORDER BY symbol",(d,))
    return d,eng,opt,agg,uni

def build_option_basket(opt):
    if opt.empty: return pd.DataFrame()
    x=opt.copy()
    x["price_multiple"]=pd.to_numeric(x["price_multiple"],errors="coerce")
    rows=[]
    for (sym,ts),g in x.groupby(["symbol","ts"]):
        ce=g[g.option_type=="CE"]; pe=g[g.option_type=="PE"]
        bull=int((ce.price_multiple>1).sum()+(pe.price_multiple<1).sum())
        bear=int((pe.price_multiple>1).sum()+(ce.price_multiple<1).sum())
        rows.append({"symbol":sym,"ts":ts,"bull_option_score":bull,"bear_option_score":bear})
    return pd.DataFrame(rows).sort_values(["symbol","ts"])

def state_history(symbol,eng,optb,agg):
    eg=eng[eng.symbol.eq(symbol)].sort_values("ts").reset_index(drop=True)
    og=optb[optb.symbol.eq(symbol)].sort_values("ts").reset_index(drop=True)
    ag=agg[agg.symbol.eq(symbol)].sort_values("ts").reset_index(drop=True)
    times=sorted(set(pd.to_datetime(eg.ts).tolist()+pd.to_datetime(og.ts).tolist()+pd.to_datetime(ag.ts).tolist()))
    hist=[]
    for ts in times:
        oe=og[pd.to_datetime(og.ts)<=ts]
        aa=ag[pd.to_datetime(ag.ts)<=ts]
        ee=eg[pd.to_datetime(eg.ts)<=ts]
        bo=so=bp=sp=0
        if not oe.empty:
            bo=int(oe.iloc[-1].bull_option_score); so=int(oe.iloc[-1].bear_option_score)
            bp=tail_count(oe.bull_option_score,lambda x:x>=5)
            sp=tail_count(oe.bear_option_score,lambda x:x>=5)
        bull=option_points(bo,bp); bear=option_points(so,sp)
        imb=px=oi=None; buy_p=sell_p=long_p=short_p=0
        if not aa.empty:
            a=aa.iloc[-1]
            imb=pd.to_numeric(a.total_qty_imbalance,errors="coerce")
            px=pd.to_numeric(a.price_change_3m_pct,errors="coerce")
            oi=pd.to_numeric(a.oi_change_3m_pct,errors="coerce")
            buy_p=tail_count(aa.total_qty_imbalance,lambda x:x>=20)
            sell_p=tail_count(aa.total_qty_imbalance,lambda x:x<=-20)
            imbs=pd.to_numeric(aa.total_qty_imbalance,errors="coerce")
            pxs=pd.to_numeric(aa.price_change_3m_pct,errors="coerce")
            ois=pd.to_numeric(aa.oi_change_3m_pct,errors="coerce")
            lf=(imbs>=20)&(pxs>0)&(ois>0); sf=(imbs<=-20)&(pxs<0)&(ois>0)
            long_p=tail_count(lf.astype(int),lambda x:x==1)
            short_p=tail_count(sf.astype(int),lambda x:x==1)
            if pd.notna(imb):
                if imb>=20: bull+=2 if buy_p>=3 else 1
                elif imb<=-20: bear+=2 if sell_p>=3 else 1
            if pd.notna(px):
                if px>0: bull+=2 if long_p>=2 else 1
                elif px<0: bear+=2 if short_p>=2 else 1
            if pd.notna(oi) and oi>0 and pd.notna(px):
                if px>0: bull+=2 if long_p>=2 else 1
                elif px<0: bear+=2 if short_p>=2 else 1

        cumoi=None
        if not ee.empty:
            cumoi=pd.to_numeric(ee.iloc[-1].future_oi_change_pct_t0,errors="coerce")
            # Add one point once cumulative OI has reached 4%.
            if pd.notna(cumoi) and cumoi>=4:
                if bull>bear: bull+=1
                elif bear>bull: bear+=1

        bull=min(10,bull); bear=min(10,bear)
        direction="LONG" if bull>bear else "SHORT" if bear>bull else "MIXED"
        score=max(bull,bear)
        conflict=bull>=4 and bear>=4
        absorption=None
        if pd.notna(imb) and pd.notna(px):
            if imb<=-20 and px>=0: absorption="SELL ABSORPTION"
            elif imb>=20 and px<=0: absorption="BUY ABSORPTION"

        if conflict:
            state,conv="CONFLICT","CONFLICT"
        elif absorption and score<8:
            state,conv=absorption,"WARNING"
        elif score>=8:
            state=("ACCELERATING "+direction) if pd.notna(cumoi) and cumoi>=4 else ("CONFIRMED "+direction)
            conv="VERY HIGH"
        elif score>=6:
            state,conv="CONFIRMED "+direction,"HIGH"
        elif score>=4:
            state,conv="BUILDING "+direction,"MEDIUM"
        elif score>=2:
            state,conv="WATCH "+direction,"LOW"
        else:
            state,conv="NEUTRAL","LOW"
        hist.append({"symbol":symbol,"ts":ts,"state":state,"conviction":conv,"score":score,
                     "direction":direction,"bull_score":bull,"bear_score":bear,
                     "option_bull_score":bo,"option_bear_score":so,"option_persistence":max(bp,sp),
                     "qty_imbalance":imb,"aggression_persistence":max(buy_p,sell_p),
                     "price_3m_pct":px,"oi_3m_pct":oi,"cumulative_oi_pct":cumoi})
    return pd.DataFrame(hist)

d,eng,opt,agg,uni=load_all()
st.title("NIFTY + BANKNIFTY — Early Detector")
st.caption("Index Money Flow • Option basket • Futures aggression • Price/OI • OI acceleration")

if d is None:
    st.info("Waiting for index collector data.")
    st.stop()

st.caption(f"Trading date: {d}")
optb=build_option_basket(opt)
tabs=st.tabs(["State + Conviction","Index Snapshot","Frozen Options","Aggression","OI 50%"])

with tabs[0]:
    boards=[]
    histories={}
    for sym in ["NIFTY","BANKNIFTY"]:
        h=state_history(sym,eng,optb,agg)
        histories[sym]=h
        if h.empty: continue
        cur=h.iloc[-1]
        clean=h[~h.state.isin(["CONFLICT","SELL ABSORPTION","BUY ABSORPTION"])]
        peak=(clean if not clean.empty else h).sort_values(["score","ts"],ascending=[False,True]).iloc[0]
        # first confirmed and first accelerating in current/peak direction
        conf=h[h.state.str.startswith("CONFIRMED")]
        acc=h[h.state.str.startswith("ACCELERATING")]
        boards.append({
            "symbol":sym,
            "current_state":cur.state,
            "current_conviction":cur.conviction,
            "current_score":round(cur.score,1),
            "peak_state":peak.state,
            "peak_score":round(peak.score,1),
            "peak_time":tist(peak.ts).strftime("%H:%M") if pd.notna(peak.ts) else None,
            "opt_bull_6":cur.option_bull_score,
            "opt_bear_6":cur.option_bear_score,
            "opt_persist":cur.option_persistence,
            "qty_imbalance_pct":cur.qty_imbalance,
            "agg_persist":cur.aggression_persistence,
            "fut_price_3m_pct":cur.price_3m_pct,
            "fut_oi_3m_pct":cur.oi_3m_pct,
            "total_oi_build_pct":cur.cumulative_oi_pct,
            "first_confirmed":tist(conf.iloc[0].ts).strftime("%H:%M") if not conf.empty else None,
            "first_accelerating":tist(acc.iloc[0].ts).strftime("%H:%M") if not acc.empty else None,
        })
    b=pd.DataFrame(boards)
    st.dataframe(b,use_container_width=True,hide_index=True)

    for sym,h in histories.items():
        if h.empty: continue
        with st.expander(f"{sym} state lifecycle"):
            view=h[["ts","state","conviction","score","option_bull_score","option_bear_score",
                    "qty_imbalance","price_3m_pct","oi_3m_pct","cumulative_oi_pct"]].copy()
            view["ts"]=view["ts"].apply(lambda x:tist(x).strftime("%H:%M:%S"))
            st.dataframe(view,use_container_width=True,hide_index=True)

with tabs[1]:
    latest=eng.sort_values("ts").groupby("symbol").tail(1).copy()
    if not latest.empty:
        latest["ts"]=latest["ts"].apply(lambda x:tist(x).strftime("%H:%M:%S"))
    st.dataframe(latest,use_container_width=True,hide_index=True)

with tabs[2]:
    latest_ts=opt.groupby("symbol")["ts"].transform("max") if not opt.empty else []
    view=opt[opt.ts.eq(latest_ts)].copy() if not opt.empty else opt
    if not view.empty:
        view["ts"]=view["ts"].apply(lambda x:tist(x).strftime("%H:%M:%S"))
    st.dataframe(view,use_container_width=True,hide_index=True)

with tabs[3]:
    latest=agg.sort_values("ts").groupby("symbol").tail(1).copy()
    if not latest.empty:
        latest["ts"]=latest["ts"].apply(lambda x:tist(x).strftime("%H:%M:%S"))
    st.dataframe(latest,use_container_width=True,hide_index=True)

with tabs[4]:
    view=eng[["symbol","ts","call_oi_change_t0","put_oi_change_t0","oi_50pct_state"]].copy()
    if not view.empty:
        view["ts"]=view["ts"].apply(lambda x:tist(x).strftime("%H:%M:%S"))
    st.dataframe(view.sort_values(["symbol","ts"],ascending=[True,False]),use_container_width=True,hide_index=True)
