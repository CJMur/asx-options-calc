import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="TradersCircle Options")

# ==========================================
# 🛑 GOOGLE SHEET CONFIG
# ==========================================
# Using the export format for the ID you provided
SHEET_ID = "1d9FQ5mn--MSNJ_WJkU--IvoSRU0gQBqE0f9s9zEb0Q4"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .main-header {
        background-color: #0e1b32; padding: 1.5rem 2rem; color: white;
        border-radius: 8px; display: flex; justify_content: space-between; align_items: center;
        margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-badge {
        font-size: 12px; background-color: #f1f5f9; color: #475569; 
        padding: 4px 8px; border-radius: 4px; border: 1px solid #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "BHP"
if 'spot_price' not in st.session_state: st.session_state.spot_price = 0.0
if 'range_pct' not in st.session_state: st.session_state.range_pct = 0.05
if 'chain_obj' not in st.session_state: st.session_state.chain_obj = None
if 'ref_data' not in st.session_state: st.session_state.ref_data = None
if 'sheet_status' not in st.session_state: st.session_state.sheet_status = "Checking..."

# --- 3. DATA LOADING ---

@st.cache_data(ttl=300)
def load_sheet(url):
    try:
        # Only read first 5 cols to avoid "Expected 599 fields" error
        df = pd.read_csv(url, usecols=[0, 1, 2, 3, 4], on_bad_lines='skip')
        df.columns = ['Ticker', 'Expiry', 'Strike', 'Type', 'Code']
        
        # Clean
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
        df['Type'] = df['Type'].astype(str).str.title().str.strip()
        df['Strike'] = df['Strike'].astype(str).str.replace('$', '', regex=False).astype(float)
        df['Expiry'] = pd.to_datetime(df['Expiry'], dayfirst=True, errors='coerce')
        return df, f"✅ Active ({len(df)} codes)"
    except Exception as e:
        return pd.DataFrame(), f"❌ Connection Failed"

# Load Sheet
if st.session_state.ref_data is None:
    data, status = load_sheet(SHEET_URL)
    st.session_state.ref_data = data
    st.session_state.sheet_status = status

def lookup_code(ticker, expiry_str, strike, kind):
    df = st.session_state.ref_data
    if df is None or df.empty: return "ENTER_CODE"
    try:
        target_date = pd.to_datetime(expiry_str)
        mask = (
            (df['Ticker'] == ticker.replace(".AX","")) &
            (df['Type'] == kind) &
            (np.isclose(df['Strike'], strike, atol=0.01)) &
            (df['Expiry'] == target_date)
        )
        res = df[mask]
        if not res.empty: return res.iloc[0]['Code']
        return "ENTER_CODE"
    except: return "ERROR"

# --- 4. YAHOO ENGINE ---
def fetch_yahoo(t):
    clean = t.upper().replace(".AX", "").strip()
    sym = f"{clean}.AX"
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="1d")
        if not hist.empty:
            px = float(hist['Close'].iloc[-1])
            # Check if options exist
            if tk.options:
                return "OK", px, tk
            else:
                return "NO_CHAIN", px, None
    except: pass
    return "ERROR", 0.0, None

def get_bs_price(kind, spot, strike, time_days, vol_pct, rate_pct):
    try:
        t = max(0.001, time_days)
        c = mibian.BS([spot, strike, rate_pct, t], volatility=vol_pct)
        return c.callPrice if kind == 'Call' else c.putPrice
    except: return 0.0

def get_greeks(kind, spot, strike, time_days, vol_pct, rate_pct):
    try:
        t = max(0.001, time_days)
        c = mibian.BS([spot, strike, rate_pct, t], volatility=vol_pct)
        g = c.call if kind == 'Call' else c.put
        return {'delta': g.delta, 'gamma': g.gamma, 'theta': g.theta, 'vega': g.vega}
    except: return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

# --- 5. UI HEADER ---
st.markdown(f"""
<div class="main-header">
    <div style="font-size: 22px; font-weight: 700;">TradersCircle <span style="font-weight: 300; opacity: 0.7;">| PRO</span></div>
    <div style="text-align: right;">
        <div style="font-size: 12px; opacity: 0.8;">{st.session_state.ticker}</div>
        <div style="font-size: 24px; font-weight: 700; color: #4ade80;">${st.session_state.spot_price:.2f}</div>
        <div class="status-badge">Sheet: {st.session_state.sheet_status}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 6. SEARCH ---
c1, c2 = st.columns([3, 1])
query = c1.text_input("Ticker", st.session_state.ticker, label_visibility="collapsed")
if c2.button("Load Chain", type="primary", use_container_width=True) or (query.upper() != st.session_state.ticker):
    st.session_state.ticker = query.upper()
    with st.spinner("Connecting..."):
        status, px, obj = fetch_yahoo(st.session_state.ticker)
        st.session_state.spot_price = px
        st.session_state.chain_obj = obj
        if status == "NO_CHAIN":
            st.warning(f"Price found (${px:.2f}), but Yahoo has no Options Data for {query}. Using Manual Mode.")
        elif status == "ERROR":
            st.error("Ticker not found.")
        st.rerun()

# --- 7. MAIN DISPLAY ---

# A. OPTION CHAIN (If Available)
if st.session_state.chain_obj:
    tk = st.session_state.chain_obj
    exps = tk.options
    if exps:
        sel_exp = st.selectbox("Expiry", exps)
        try:
            chain = tk.option_chain(sel_exp)
            df = pd.merge(chain.calls, chain.puts, on="strike", how="outer", suffixes=("_c", "_p")).fillna(0)
            center = st.session_state.spot_price
            df = df[(df['strike'] > center*0.8) & (df['strike'] < center*1.2)]
            
            st.markdown(f"**Chain: {sel_exp}**")
            
            view = pd.DataFrame({
                'STRIKE': df['strike'],
                'C_Last': df['lastPrice_c'], 'C_Vol': df['impliedVolatility_c'],
                'P_Last': df['lastPrice_p'], 'P_Vol': df['impliedVolatility_p']
            })
            
            selection = st.dataframe(
                view,
                column_config={
                    "C_Last": st.column_config.NumberColumn("Call $", format="$%.2f"),
                    "STRIKE": st.column_config.NumberColumn("Strike", format="%.2f"),
                    "P_Last": st.column_config.NumberColumn("Put $", format="$%.2f"),
                    "C_Vol": None, "P_Vol": None
                },
                hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row"
            )
            
            if selection.selection['rows']:
                idx = selection.selection['rows'][0]
                row = view.iloc[idx]
                days = (datetime.strptime(sel_exp, "%Y-%m-%d") - datetime.now()).days
                
                st.info(f"Selected: **${row['STRIKE']} Strike**")
                
                def add(side, kind, px, iv):
                    if iv < 0.01: iv = 0.20
                    code = lookup_code(st.session_state.ticker, sel_exp, row['STRIKE'], kind)
                    st.session_state.legs.append({
                        "Qty": 1 if side=="Buy" else -1, "Type": kind, 
                        "Strike": row['STRIKE'], "Expiry": days, "Vol": iv*100, 
                        "Entry": px, "Code": code
                    })
                
                b1, b2, b3, b4 = st.columns(4)
                if b1.button("Buy Call"): add("Buy", "Call", row['C_Last'], row['C_Vol'])
                if b2.button("Sell Call"): add("Sell", "Call", row['C_Last'], row['C_Vol'])
                if b3.button("Buy Put"): add("Buy", "Put", row['P_Last'], row['P_Vol'])
                if b4.button("Sell Put"): add("Sell", "Put", row['P_Last'], row['P_Vol'])
        except Exception as e:
            st.error(f"Chain Error: {e}")
    else:
        st.warning("Yahoo returned empty option list.")

# B. MANUAL BUILDER (Always Visible Fallback)
with st.expander("🛠 Manual Leg Builder (Use if Chain missing)", expanded=not st.session_state.chain_obj):
    with st.form("manual"):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        code = c1.text_input("Code", "BHPS49")
        qty = c2.number_input("Qty", 1)
        kind = c3.selectbox("Type", ["Call", "Put"])
        strk = c4.number_input("Strike", value=round(st.session_state.spot_price, 2))
        days = c5.number_input("Days", 45)
        vol = c6.number_input("Vol%", 20.0)
        if st.form_submit_button("Add Manually"):
            px = get_bs_price(kind, st.session_state.spot_price, strk, days, vol, 4.0)
            st.session_state.legs.append({
                "Qty": qty, "Type": kind, "Strike": strk, 
                "Expiry": days, "Vol": vol, "Entry": px, "Code": code
            })
            st.rerun()

# --- 8. PORTFOLIO & CHARTS ---
if st.session_state.legs:
    st.markdown("---")
    st.subheader("📋 Active Portfolio")
    
    # Editor
    df_port = pd.DataFrame(st.session_state.legs)
    edited = st.data_editor(
        df_port,
        column_config={
            "Code": st.column_config.TextColumn("ASX Code", width="medium"),
            "Entry": st.column_config.NumberColumn("Entry", format="$%.3f"),
            "Vol": st.column_config.NumberColumn("Vol %", format="%.1f%%"),
            "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
        },
        use_container_width=True, num_rows="dynamic", key="editor"
    )
    st.session_state.legs = edited.to_dict('records')
    
    # Delta
    tot_delta = 0
    for leg in st.session_state.legs:
        g = get_greeks(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
        tot_delta += g['delta'] * leg['Qty'] * 100
    st.metric("Net Delta", f"{tot_delta:.1f}")
    
    # Chart
    st.markdown("---")
    c_ctrl, c_view = st.columns([1, 3])
    with c_ctrl:
        st.caption("Controls")
        if st.button("Zoom 5%"): st.session_state.range_pct = 0.05
        if st.button("Zoom 15%"): st.session_state.range_pct = 0.15
