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
    # Six-of-six option breadth is worth 3 points. Persistence is a separate
    # confirmation point; previously it was added and then accidentally capped.
    p=3 if score>=6 else 2.5 if score>=5 else 2 if score>=4 else 1 if score>=3 else 0
    return min(4,p+(1 if persist>=3 else 0))

def session_move_points(change_pct):
    """Score cumulative movement from the 09:18 baseline, not index points."""
    if pd.isna(change_pct): return 0
    move=abs(float(change_pct))
    return 2 if move>=0.50 else 1 if move>=0.25 else 0

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
    ag=agg[agg.symbol.eq(symbol)].sort_values("ts").reset_index(drop=True)
    times=sorted(set(pd.to_datetime(eg.ts).tolist()+pd.to_datetime(ag.ts).tolist()))
    hist=[]
    for ts in times:
        ee=eg[pd.to_datetime(eg.ts)<=ts]; aa=ag[pd.to_datetime(ag.ts)<=ts]
        px=oi=imb=td=cumoi=session_px=None
        if not aa.empty:
            ar=aa.iloc[-1]; px=pd.to_numeric(ar.get("price_change_3m_pct"),errors="coerce"); oi=pd.to_numeric(ar.get("oi_change_3m_pct"),errors="coerce")
            imb=pd.to_numeric(ar.get("total_qty_imbalance"),errors="coerce"); td=pd.to_numeric(ar.get("delta_pct"),errors="coerce")
        if not ee.empty:
            er=ee.iloc[-1]
            if "future_price_change_3m_pct" in ee.columns:
                v=pd.to_numeric(er.get("future_price_change_3m_pct"),errors="coerce"); px=v if pd.notna(v) else px
            if "future_oi_change_3m_pct" in ee.columns:
                v=pd.to_numeric(er.get("future_oi_change_3m_pct"),errors="coerce"); oi=v if pd.notna(v) else oi
            cumoi=pd.to_numeric(er.get("future_oi_change_pct_t0"),errors="coerce"); session_px=pd.to_numeric(er.get("spot_change_pct_t0"),errors="coerce")

        # v3.1 INDEX PRICE PERSISTENCE:
        # Score the net move over the latest three 3-minute observations (about 9 minutes)
        # instead of requiring each individual bar to exceed +/-0.05%.
        recent=eg[pd.to_datetime(eg.ts)<=ts].tail(3)
        pxs=pd.to_numeric(recent.get("future_price_change_3m_pct",pd.Series(dtype=float)),errors="coerce").dropna()
        if pxs.empty and not aa.empty:
            pxs=pd.to_numeric(aa.tail(3).price_change_3m_pct,errors="coerce").dropna()

        bp=sp=0.0
        if len(pxs)>=3:
            net_9m=float(pxs.tail(3).sum())
            nonneg=int((pxs.tail(3)>=0).sum())
            nonpos=int((pxs.tail(3)<=0).sum())

            # 1 point: meaningful net 9m move in the direction.
            # 2 points: >=0.10% net move AND at least 2 of 3 bars aligned.
            if net_9m>=0.10 and nonneg>=2:
                bp=2.0
            elif net_9m>=0.05:
                bp=1.0

            if net_9m<=-0.10 and nonpos>=2:
                sp=2.0
            elif net_9m<=-0.05:
                sp=1.0

        # Index-calibrated fresh OI /2 is DIRECTION-NEUTRAL positioning evidence.
        # 1 point: current 3m OI >= +0.05%
        # 2 points: 3 consecutive 3m OI observations >= +0.05%
        oi_points=0.0
        fresh_oi_persist=0
        if not ee.empty and "fresh_oi_confirmation_points" in ee.columns:
            oi_points=pd.to_numeric(ee.iloc[-1].get("fresh_oi_confirmation_points"),errors="coerce")
            oi_points=float(oi_points) if pd.notna(oi_points) else 0.0
            fresh_oi_persist=pd.to_numeric(ee.iloc[-1].get("fresh_oi_persistence"),errors="coerce")
            fresh_oi_persist=int(fresh_oi_persist) if pd.notna(fresh_oi_persist) else 0
        else:
            ois_recent=pd.to_numeric(
                eg[pd.to_datetime(eg.ts)<=ts].get("future_oi_change_3m_pct",pd.Series(dtype=float)),
                errors="coerce"
            ).dropna().tail(3)
            if pd.notna(oi) and oi>=0.05:
                oi_points=1.0
            if len(ois_recent)>=3 and bool((ois_recent>=0.05).all()):
                oi_points=2.0
                fresh_oi_persist=3

        positioning_state = (
            "OI BUILDING — DIRECTION UNRESOLVED" if oi_points>=2
            else "FRESH OI" if oi_points>=1
            else "NO MATERIAL FRESH OI"
        )

        fstate="UNKNOWN"; persist=0; statepts=0.0
        if pd.notna(px) and pd.notna(oi):
            fstate="LONG_BUILDUP" if px>0 and oi>0 else "SHORT_BUILDUP" if px<0 and oi>0 else "SHORT_COVERING" if px>0 and oi<0 else "LONG_UNWINDING" if px<0 and oi<0 else "MIXED"
        src=eg[pd.to_datetime(eg.ts)<=ts]
        if {"future_price_change_3m_pct","future_oi_change_3m_pct"}.issubset(src.columns):
            ps=pd.to_numeric(src.future_price_change_3m_pct,errors="coerce"); os_=pd.to_numeric(src.future_oi_change_3m_pct,errors="coerce")
        else:
            src=aa; ps=pd.to_numeric(src.price_change_3m_pct,errors="coerce") if not src.empty else pd.Series(dtype=float); os_=pd.to_numeric(src.oi_change_3m_pct,errors="coerce") if not src.empty else pd.Series(dtype=float)
        lp=tail_count(((ps>0)&(os_>0)).astype(int),lambda x:x==1); shp=tail_count(((ps<0)&(os_>0)).astype(int),lambda x:x==1)
        persist=lp if fstate=="LONG_BUILDUP" else shp if fstate=="SHORT_BUILDUP" else 0; statepts=2.0 if persist>=3 else 1.0 if persist>=2 else 0.0
        bs=statepts if fstate=="LONG_BUILDUP" else 0.0; ss=statepts if fstate=="SHORT_BUILDUP" else 0.0

        flow=flowx=None; fp=0.0
        if not ee.empty and "total_flow_3m_cr" in ee.columns:
            fs=pd.to_numeric(ee.total_flow_3m_cr,errors="coerce"); flow=fs.iloc[-1]; prior=fs.iloc[:-1].dropna().tail(5)
            if pd.notna(flow) and len(prior)>=3 and prior.mean()>0: flowx=float(flow/prior.mean()); fp=1.5 if flowx>=2 else 1.0 if flowx>=1.5 else 0.0
        bf=fp if pd.notna(px) and px>0 else 0.0; sf=fp if pd.notna(px) and px<0 else 0.0

        pcrt=None; bpc=spc=0.0
        if not ee.empty and "pcr_change_3m" in ee.columns:
            pc=pd.to_numeric(ee.pcr_change_3m,errors="coerce").dropna().tail(3)
            if len(pc)>=2:
                pcrt=float(pc.sum()); bpc=1.0 if int((pc>0).sum())>=2 and pcrt>0 else 0.0; spc=1.0 if int((pc<0).sum())>=2 and pcrt<0 else 0.0

        # v3.1 GRADUATED EXECUTED AGGRESSION:
        # +/-20% to <30% earns 0.5; +/-30% or stronger earns 1.0.
        ba=sa=0.0
        if pd.notna(td):
            ba=1.0 if td>=30 else 0.5 if td>=20 else 0.0
            sa=1.0 if td<=-30 else 0.5 if td<=-20 else 0.0
        bi=si=0.0
        if pd.notna(imb): bi=0.5 if imb>=20 else 0.0; si=0.5 if imb<=-20 else 0.0

        # Directional score excludes direction-neutral OI points until price/state resolves.
        bull=min(8.0,bp+bs+bf+bpc+ba+bi)
        bear=min(8.0,sp+ss+sf+spc+sa+si)
        direction="LONG" if bull>bear else "SHORT" if bear>bull else "MIXED"

        # Full conviction score includes fresh-positioning OI /2, but do not call
        # it CONFIRMED directional unless directional evidence itself is strong.
        score=min(10.0,max(bull,bear)+oi_points)
        directional_score=max(bull,bear)

        if oi_points>=2 and directional_score<4:
            state="OI BUILDING — DIRECTION UNRESOLVED"
        elif score>=8.5 and directional_score>=6:
            state="HIGH CONVICTION "+direction
        elif score>=7 and directional_score>=5:
            state="CONFIRMED "+direction
        elif score>=6 and directional_score>=4:
            state="BUILDING "+direction
        elif directional_score>=4:
            state="WATCH "+direction
        elif oi_points>=1:
            state="FRESH OI — DIRECTION UNRESOLVED"
        else:
            state="NEUTRAL"
        conv="VERY HIGH" if score>=8.5 else "HIGH" if score>=7 else "MEDIUM" if score>=6 else "LOW"
        hist.append({"symbol":symbol,"ts":ts,"state":state,"conviction":conv,"score":score,"direction":direction,"bull_score":bull,"bear_score":bear,
                     "option_bull_score":0,"option_bear_score":0,"option_persistence":0,"qty_imbalance":imb,"aggression_persistence":0,
                     "session_price_pct":session_px,"price_3m_pct":px,"oi_3m_pct":oi,"cumulative_oi_pct":cumoi,
                     "price_persistence_points":max(bp,sp),"oi_confirmation_points":oi_points,
                     "fresh_oi_persistence":fresh_oi_persist,"positioning_state":positioning_state,
                     "directional_score":directional_score,
                     "futures_state":fstate,"futures_state_persistence":persist,
                     "futures_state_points":statepts,"total_flow_3m_cr":flow,"money_flow_acceleration_x":flowx,"money_flow_points":fp,
                     "pcr_trend_9m":pcrt,"pcr_trend_points":max(bpc,spc),"aggression_points":max(ba,sa),"imbalance_points":max(bi,si)})
    return pd.DataFrame(hist)

d,eng,opt,agg,uni=load_all()
st.title("NIFTY + BANKNIFTY — Early Detector v3.1 Index-Calibrated")
st.caption("Score /10: 9-minute price persistence 2 • Fresh OI positioning 2 (direction-neutral) • Futures-state persistence 2 • Money-flow expansion 1.5 • PCR trend 1 • Graduated aggression 1 • Qty imbalance 0.5. CONFIRMED still requires total >=7 and directional >=5.")

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
            "session_price_pct":cur.session_price_pct,
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
                    "qty_imbalance","session_price_pct","price_3m_pct","oi_3m_pct",
                    "cumulative_oi_pct","price_persistence_points","oi_confirmation_points","futures_state","futures_state_persistence","futures_state_points","total_flow_3m_cr","money_flow_acceleration_x","money_flow_points","pcr_trend_9m","pcr_trend_points","aggression_points","imbalance_points"]].copy()
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
