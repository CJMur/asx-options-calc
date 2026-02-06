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

# --- CSS STYLING ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    
    /* HEADER */
    .main-header {
        background-color: #0e1b32; 
        padding: 1rem 1.5rem; 
        color: white;
        border-radius: 8px; 
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title { font-size: 22px; font-weight: 700; white-space: nowrap; }
    
    /* STATUS BADGE */
    .status-badge {
        font-size: 11px; background-color: #f1f5f9; color: #475569; 
        padding: 2px 6px; border-radius: 4px; border: 1px solid #cbd5e1;
        display: inline-block; margin-top: 4px;
    }
    .status-success { color: #16a34a; background-color: #f0fdf4; border-color: #bbf7d0; }
    .status-error { color: #dc2626; background-color: #fef2f2; border-color: #fecaca; }
    
    /* BUTTONS */
    div[data-testid="stButton"] > button {
        background-color: #ef4444 !important; /* Red/Orange like screenshot */
        color: white !important;
        border: none;
        font-weight: 600;
    }
    
    /* TABLE TWEAKS */
    .stDataFrame { border: none !important; }
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
if 'data_source' not in st.session_state: st.session_state.data_source = "None"
if 'vol_manual' not in st.session_state: st.session_state.vol_manual = 33.5 # Matching screenshot

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=600)
def load_sheet(raw_url):
    try:
        if "/edit" in raw_url:
            base = raw_url.split("/edit")[0]
            csv_url = f"{base}/export?format=csv&gid=0"
        else:
            csv_url = raw_url

        df = pd.read_csv(csv_url, usecols=[2, 3, 4, 5, 6], on_bad_lines='skip', dtype=str)
        df.columns = ['Code', 'Ticker', 'Type', 'Expiry', 'Strike']
        
        df['Ticker'] = df['Ticker'].str.upper().str.strip()
        df['Type'] = df['Type'].str.upper().str.strip().replace({'C': 'Call', 'P': 'Put'})
        df['Strike'] = pd.to_numeric(df['Strike'].str.replace(',', '').str.replace('$', ''), errors='coerce')
        df['Expiry'] = pd.to_datetime(df['Expiry'], dayfirst=True, errors='coerce')
        
        df = df.dropna(subset=['Code', 'Ticker', 'Strike', 'Expiry'])
        return df, f"success|{len(df)} Codes Loaded"
    except Exception as e:
        return pd.DataFrame(), f"error|{str(e)[:30]}"

if st.session_state.ref_data is None:
    data, msg = load_sheet(RAW_SHEET_URL)
    st.session_state.ref_data = data
    st.session_state.sheet_msg = msg

# --- MATH ENGINE (ROBUST) ---
def get_bs_price(kind, spot, strike, time_days, vol_pct, rate_pct=4.0):
    try:
        # SAFETY: If expired, return Intrinsic Value
        if time_days <= 0:
            if kind == 'Call': return max(0.0, spot - strike)
            else: return max(0.0, strike - spot)
            
        # SAFETY: Mibian hates 0 days. Force min 0.5 days for math.
        safe_days = max(0.5, time_days)
        
        c = mibian.BS([spot, strike, rate_pct, safe_days], volatility=vol_pct)
        return c.callPrice if kind == 'Call' else c.putPrice
    except: return 0.0

def get_greeks(kind, spot, strike, time_days, vol_pct, rate_pct=4.0):
    try:
        if time_days <= 0: return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        safe_days = max(0.5, time_days)
        c = mibian.BS([spot, strike, rate_pct, safe_days], volatility=vol_pct)
        g = c.call if kind == 'Call' else c.put
        return {'delta': g.delta, 'gamma': g.gamma, 'theta': g.theta, 'vega': g.vega}
    except: return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

def fetch_data(t):
    clean = t.upper().replace(".AX", "").strip()
    sym = f"{clean}.AX"
    spot = 0.0
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="1d")
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
        else:
            return "ERROR", 0.0, None
        
        if tk.options: return "YAHOO", spot, tk
        else: return "SHEET", spot, None
    except: return "ERROR", 0.0, None

# --- 4. HEADER ---
status_parts = st.session_state.sheet_msg.split("|")
status_cls = "status-success" if status_parts[0] == "success" else "status-error"
status_txt = status_parts[1] if len(status_parts) > 1 else status_parts[0]

st.markdown(f"""
<div class="main-header">
    <div class="header-title">
        TradersCircle <span style="font-weight: 300; opacity: 0.7;">| PRO</span>
    </div>
    <div class="header-info">
        <div style="font-size: 14px; opacity: 0.9;">{st.session_state.ticker}</div>
        <div style="font-size: 28px; font-weight: 700; color: #4ade80;">${st.session_state.spot_price:.2f}</div>
        <div class="status-badge {status_cls}">{status_txt}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. CONTROLS ---
c_search, c_vol, c_btn = st.columns([2, 1, 1])
with c_search:
    query = st.text_input("Ticker", st.session_state.ticker, label_visibility="collapsed")
with c_vol:
    st.session_state.vol_manual = st.slider("Implied Volatility %", 10.0, 100.0, st.session_state.vol_manual, 0.5, label_visibility="collapsed")

if c_btn.button("Load Chain", type="primary", use_container_width=True) or (query.upper() != st.session_state.ticker):
    st.session_state.ticker = query.upper()
    with st.spinner("Analyzing Market..."):
        source, px, obj = fetch_data(st.session_state.ticker)
        st.session_state.spot_price = px
        st.session_state.chain_obj = obj
        st.session_state.data_source = source
        data, msg = load_sheet(RAW_SHEET_URL)
        st.session_state.ref_data = data
        st.session_state.sheet_msg = msg
        st.rerun()

# --- 6. ADVANCED CHAIN DISPLAY ---
df_view = pd.DataFrame()
current_exp = None

if st.session_state.data_source == "SHEET":
    ref = st.session_state.ref_data
    tkr = st.session_state.ticker.replace(".AX", "")
    subset = ref[ref['Ticker'] == tkr]
    
    if not subset.empty:
        valid_exps = sorted(subset['Expiry'].unique())
        # Filter out old dates just in case, but keep today
        # valid_exps = [d for d in valid_exps if d >= datetime.now() - timedelta(days=1)]
        
        exp_map = {d.strftime("%Y-%m-%d"): d for d in valid_exps}
        
        # Default to a future expiry if possible
        default_idx = 0
        if len(exp_map) > 0:
            current_exp = st.
