import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="ASX Options Manager")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .main-header {
        background-color: #0e1b32;
        padding: 1rem 2rem;
        color: white;
        border-radius: 8px;
        display: flex; justify_content: space-between; align_items: center;
        margin-bottom: 1rem;
    }
    .stCard { background-color: #f9f9f9; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }
    /* Make tabs look professional */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 5px 5px 0 0; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #0e1b32; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "BHP.AX"
if 'spot_price' not in st.session_state: st.session_state.spot_price = 45.00 # Default fallback

# --- 3. MATH HELPERS ---
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

# --- 4. HEADER UI ---
st.markdown(f"""
<div class="main-header">
    <div style="font-size: 24px; font-weight: 600;">TradersCircle <span style="opacity:0.7">| PORTFOLIO</span></div>
    <div>
        <span style="font-size: 14px; opacity: 0.8; margin-right: 10px;">{st.session_state.ticker.replace('.AX','')}</span>
        <span style="font-size: 20px; font-weight: bold;">${st.session_state.spot_price:.2f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. HYBRID STRATEGY BUILDER ---
st.markdown("### 🛠 Strategy Builder")

# We use Tabs to separate the "Automatic" (Chain) from "Manual" (Ticket)
tab_manual, tab_search = st.tabs(["✍️ Manual Builder (Reliable)", "🔍 Chain Search (Auto)"])

# --- TAB 1: MANUAL BUILDER (The "Workhorse") ---
with tab_manual:
    with st.container():
        st.markdown("#### Configure Position Manually")
        
        # Ticker Update Logic
        c_tick, c_fetch = st.columns([3, 1])
        new_ticker = c_tick.text_input("Underlying Ticker", value=st.session_state.ticker).upper()
        if c_fetch.button("Update Price"):
            sym = f"{new_ticker}.AX" if not new_ticker.endswith(".AX") else new_ticker
            st.session_state.ticker = sym
            try:
                hist = yf.Ticker(sym).history(period="1d")
                if not hist.empty:
                    st.session_state.spot_price = float(hist['Close'].iloc[-1])
            except: pass
            st.rerun()

        # The "Trade Ticket" Form
        with st.form("manual_add"):
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1.5, 1.5, 1.5, 1])
            qty = c1.number_input("Qty", 1, 1000, 1)
            side = c2.selectbox("Side", ["Buy", "Sell"])
            kind = c3.selectbox("Type", ["Call", "Put", "Stock"])
            strike = c4.number_input("Strike", value=round(st.session_state.spot_price, 2))
            expiry = c5.number_input("Days", value=45)
            vol = c6.number_input("Vol %", value=20.0)
            
            if c7.form_submit_button("➕ Add", type="primary"):
                # Calc Entry Price automatically if user didn't specify (simplified here)
                entry_px = get_bs_price(kind, st.session_state.spot_price, strike, expiry, vol, 4.0)
                
                st.session_state.legs.append({
                    "Qty": qty * (1 if side == "Buy" else -1),
                    "Type": kind, "Strike": strike, "Expiry": expiry, "Vol": vol, "Entry": entry_px
                })
                st.rerun()

# --- TAB 2: OPTION CHAIN SEARCH (The "Nice to Have") ---
with tab_search:
    st.info("Note: Yahoo Finance data for ASX options can be patchy. Use Manual Builder if this returns no results.")
    search_ticker = st.text_input("Search Chain (e.g. BHP)", value="BHP").upper()
    
    if st.button("Load Chain"):
        sym = f"{search_ticker}.AX" if not search_ticker.endswith(".AX") else search_ticker
        try:
            tk = yf.Ticker(sym)
            dates = tk.options
            if dates:
                st.success(f"Found expirations: {dates}")
                # (Full chain logic omitted for brevity in this fix to ensure stability first)
                # If we find data, we would render the table here.
            else:
                st.warning("No option chain found. Please use the 'Manual Builder' tab.")
        except:
            st.error("Connection error. Use Manual Builder.")

# --- 6. PORTFOLIO TABLE (Always Visible if legs exist) ---
if st.session_state.legs:
    st.markdown("---")
    st
