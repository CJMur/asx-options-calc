import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="TradersCircle Options")
RAW_SHEET_URL = "https://docs.google.com/spreadsheets/d/1d9FQ5mn--MSNJ_WJkU--IvoSRU0gQBqE0f9s9zEb0Q4/edit?usp=sharing"

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
        cursor: help; white-space: nowrap;
    }
    .status-error { color: #dc2626; background-color: #fef2f2; border-color: #fecaca; }
    .status-success { color: #16a34a; background-color: #f0fdf4; border-color: #bbf7d0; }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "BHP"
if 'spot_price' not in st.session_state: st.session_state.spot_price = 0.0
if 'range_pct' not in st.session_state: st.session_state.range_pct = 0.05
if 'chain_obj' not in st.session_state: st.session_state.chain_obj = None
if 'ref_data' not in st.session_state: st.session_state.ref_data = None
if 'sheet_msg' not in st.session_state: st.session_state.sheet_msg = "Initializing..."

# --- 3. DATA LOADER (MATCHING YOUR SCREENSHOT) ---

@st.cache_data(ttl=600)
def load_sheet(raw_url):
    try:
        # Convert to export link
        if "/edit" in raw_url:
            base = raw_url.split("/edit")[0]
            csv_url = f"{base}/export?format=csv&gid=0"
        else:
            csv_url = raw_url

        # READ SPECIFIC COLUMNS BASED ON YOUR SCREENSHOT:
        # Col C (Index 2) = ASXCode
        # Col D (Index 3) = Underlying (Ticker)
        # Col E (Index 4) = OptType (C/P)
        # Col F (Index 5) = ExpDate
        # Col G (Index 6) = Strike
        df = pd.read_csv(csv_url, usecols=[2, 3, 4, 5, 6], on_bad_lines='skip', dtype=str)
        
        # Rename to standard internal names
        df.columns = ['Code', 'Ticker', 'Type', 'Expiry', 'Strike']
        
        # CLEANING
        # 1. Ticker
        df['Ticker'] = df['Ticker'].str.upper().str.strip()
        
        # 2. Type (Map "C" -> "Call", "P" -> "Put")
        df['Type'] = df['Type'].str.upper().str.strip().replace({'C': 'Call', 'P': 'Put'})
        
        # 3. Strike (Remove commas/$)
        df['Strike'] = pd.to_numeric(df['Strike'].str.replace(',', '').str.replace('$', ''), errors='coerce')
        
        # 4. Expiry (Parse dates like 19/03/2026)
        df['Expiry'] = pd.to_datetime(df['Expiry'], dayfirst=True, errors='coerce')
        
        # 5. Drop bad rows
        df = df.dropna(subset=['Code', 'Ticker', 'Strike', 'Expiry'])
        
        return df, f"success|{len(df)} Codes Loaded"
    
    except Exception as e:
        return pd.DataFrame(), f"error|{str(e)[:30]}"

# Initialize Data
if st.session_state.ref_data is None:
    data, msg = load_sheet(RAW_SHEET_URL)
    st.session_state.ref_data = data
    st.session_state.sheet_msg = msg

def lookup_code(ticker, expiry_str, strike, kind):
    df = st.session_state.ref_data
    if df is None or df.empty: return "ENTER_CODE"
    try:
        target_date = pd.to_datetime(expiry_str)
        # Robust Matching
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
            if tk.options: return "OK", px, tk
            else: return "NO_CHAIN", px, None
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
status_parts = st.session_state.sheet_msg.split("|")
status_cls = "status-success" if status_parts[0] == "success" else "status-error"
status_txt = status_parts[1] if len(status_parts) > 1 else status_parts[0]

st.markdown(f"""
<div class="main-header">
    <div style="font-size: 22px; font-weight: 700;">TradersCircle <span style="font-weight: 300; opacity: 0.7;">| PRO</span></div>
    <div style="text-align: right;">
        <div style="font-size: 12px; opacity: 0.8;">{st.session_state.ticker}</div>
        <div style="font-size: 24px; font-weight: 700; color: #4ade80;">${st.session_state.spot_price:.2f}</div>
        <span class="status-badge {status_cls}" title="Data Status">Sheet: {status_txt}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 6. SEARCH ---
c1, c2 = st.columns([3, 1])
query = c1.text_input("Ticker", st.session_state.ticker, label_visibility
