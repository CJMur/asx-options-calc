# ==========================================
# TradersCircle Options Calculator
# VERSION: 9.1 (UI Fixes: Teal Selection & Blue Sliders)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, time, timedelta
import pytz
import math

# --- 1. CONFIGURATION & THEME ---
st.set_page_config(layout="wide", page_title="TradersCircle Options v9.1")
RAW_SHEET_URL = "https://docs.google.com/spreadsheets/d/1d9FQ5mn--MSNJ_WJkU--IvoSRU0gQBqE0f9s9zEb0Q4/edit?usp=sharing"

# --- CSS STYLING ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; padding-bottom: 5rem !important; }
    
    /* Header Box */
    .header-box {
        padding: 1.5rem; background-color: #0e1b32; border-radius: 10px; color: white;
        margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-bottom: 4px solid #1DBFD2;
    }
    .header-title { font-size: 24px; font-weight: 700; margin: 0; }
    .header-sub { font-size: 14px; opacity: 0.8; margin: 0; }
    
    .status-tag {
        background-color: rgba(255,255,255,0.15); padding: 4px 10px; border-radius: 4px;
        font-size: 12px; font-family: monospace;
    }
    
    /* Primary Button (Teal) */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #1DBFD2 !important; border: none; color: white !important; font-weight: bold;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #16aebf !important;
    }
    
    /* Secondary Button */
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #f8fafc !important; color: #334155 !important; border: 1px solid #cbd5e1;
    }
    
    /* --- SLIDER COLOR FIX (Reverted to aggressive blue) --- */
    div[data-baseweb="slider"] > div > div > div { background-color: #0050FF !important; }
    div[role="slider"] { background-color: #0050FF !important; box-shadow: none !important; }
    div[data-testid="stSlider"] svg path { fill: #0050FF !important; stroke: #0050FF !important; }
    div[data-testid="stSlider"] p { color: white !important; }
    input[type=range] { accent-color: #0050FF !important; }
    
    /* Dataframe Row Selection Highlight (Teal) */
    [data-testid="stDataFrame"] [aria-selected="true"] > div {
        background-color: rgba(29, 191, 210, 0.4) !important;
        color: white !important;
    }
    
    .stDataFrame { border: none !important; }

    /* Clean Table Headers */
    .trade-header {
        font-weight: 700; color: #94a3b8; font-size: 12px; text-transform: uppercase;
        margin-bottom: 5px; cursor: help;
    }
    
    /* Delete Button Styling */
    button[kind="secondary"] {
        padding: 0rem 0.5rem !important;
        min-height: 0px !important;
        height: 32px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "" 
if 'spot_price' not in st.session_state: st.session_state.spot_price = 0.0
if 'range_pct' not in st.session_state: st.session_state.range_pct = 0.05
if 'chain_obj' not in st.session_state: st.session_state.chain_obj = None
if 'ref_data' not in st.session_state: st.session_state.ref_data = None
if 'sheet_msg' not in st.session_state: st.session_state.sheet_msg = "Initializing..."
if 'manual_spot' not in st.session_state: st.session_state.manual_spot = False
if 'is_market_open' not in st.session_state: st.session_state.is_market_open = True
if 'div_info' not in st.session_state: st.session_state.div_info = None
if 'matrix_vol_mod' not in st.session_state: st.session_state.matrix_vol_mod = 0.0

# Smart Search States
if 'preselect_code' not in st.session_state: st.session_state.preselect_code = None
if 'preselect_expiry' not in st.session_state: st.session_state.preselect_expiry = None
if 'preselect_strike' not in st.session_state: st.session_state.preselect_strike = None

# --- TOOLTIP DEFINITIONS ---
TOOLTIPS = {
    "Theo": "The theoretical fair value of the option calculated using the Black-Scholes (European) or Bjerksund-Stensland (American) pricing model.",
    "IV": "Implied Volatility: The market's forecast of a likely movement in the security's price. Higher IV means more expensive options.",
    "Delta": "The rate of change of the option price with respect to the underlying asset's price. A delta of 0.50 means the option moves $0.50 for every $1.00 move in the stock.",
    "Strike": "The set price at which the option contract can be exercised (bought or sold).",
    "Code": "The unique ASX exchange ticker symbol for this specific option contract.",
    "Entry": "The price you entered the trade at (or current market price).",
    "Premium": "The total cost or credit for the trade. Calculated as Price × Quantity × 100.",
    "Margin": "The estimated collateral required to hold this position, calculated based on the worst-case scenario from the exchange's risk analysis."
}

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=600)
def load_sheet(raw_url):
    try:
        if "/edit" in raw_url:
            base = raw_url.split("/edit")[0]
            csv_url = f"{base}/export?format=csv&gid=0"
        else:
            csv_url = raw_url

        df = pd.read_csv(csv_url, on_bad_lines='skip', dtype=str)
        
        header_map = {
            'ASXCode': 'Code', 'Underlying': 'Ticker', 'OptType': 'Type', 
            'ExpDate': 'Expiry', 'Strike': 'Strike', 'Volatility': 'Vol', 
            'Settlement': 'Settlement', 'Style': 'Style', 'Lookup Key': 'LookupKey'
        }
        df = df.rename(columns=header_map)
        
        required = ['Code', 'Ticker', 'Strike', 'Expiry']
        if not all(col in df.columns for col in required):
            return pd.DataFrame(), f"error|Missing columns: {list(df.columns)}"

        df['Ticker'] = df['Ticker'].str.upper().str.strip()
        df['Code'] = df['Code'].str.upper().str.strip()
        df['Type'] = df['Type'].str.upper().str.strip().replace({'C': 'Call', 'P': 'Put'})
        
        if 'Style' in df.columns:
            df['Style'] = df['Style'].str.upper().str.strip().replace({'A': 'American', 'E': 'European'})
        else:
            df['Style'] = 'American'
            
        df['Strike'] = pd.to_numeric(df['Strike'].str.replace(',', '').str.replace('$', ''), errors='coerce')
        df['Expiry'] = pd.to_datetime(df['Expiry'], dayfirst=True, errors='coerce')
        
        if 'Vol' in df.columns:
            df['Vol'] = df['Vol'].str.replace('%', '').astype(float)
            mask = df['Vol'] <= 1.0 
            df.loc[mask, 'Vol'] = df.loc[mask, 'Vol'] * 100
        else:
            df['Vol'] = 30.0
            
        if 'Settlement' in df.columns:
            df['Settlement'] = pd.to_numeric(df['Settlement'].str.replace('$', ''), errors='coerce')
        else:
            df['Settlement'] = 0.0

        scen_cols = [c for c in df.columns if 'Scenario' in c]
        if scen_cols:
            scen_df = df[scen_cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce'))
            df['UnitMargin'] = scen_df.min(axis=1).fillna(0.0)
        else:
            df['UnitMargin'] = 0.0

        df = df.dropna(subset=['Code', 'Ticker', 'Strike', 'Expiry'])
        return df, f"success|{len(df)} Codes Loaded"
    except Exception as e:
        return pd.DataFrame(), f"error|{str(e)[:30]}"

if st.session_state.ref_data is None:
    data, msg = load_sheet(RAW_SHEET_URL)
    st.session_state.ref_data = data
    st.session_state.sheet_msg = msg

# --- 4. MATH ENGINE ---
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes_european(S, K, T, r, sigma, option_type):
    if T <= 0: return max(0, S - K) if option_type == 'Call' else max(0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'Call':
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bjerksund_stensland_american(S, K, T, r, sigma, option_type):
    bs_price = black_scholes_european(S, K, T, r, sigma, option_type)
    if option_type == 'Call': return bs_price 
    else: return max(bs_price, max(0, K - S))

def calculate_price_and_delta(style, kind, spot, strike, time_days, vol_pct):
    r = 0.04 
    try:
        T = max(0.001, time_days / 365.0)
        v = vol_pct / 100.0
        S = float(spot)
        K = float(strike)
        
        if st.session_state.div_info:
            d_info = st.session_state.div_info
            if d_info['amount'] > 0 and d_info['date']:
                days_to_div = (d_info['date'] - datetime.now()).days
                if 0 <= days_to_div < time_days:
                    t_div = days_to_div / 365.0
                    div_pv = d_info['amount'] * math.exp(-r * t_div)
                    S = max(0.01, S - div_pv)
        
        if style.upper() == 'EUROPEAN':
            price = black_scholes_european(S, K, T, r, v, kind)
        else:
            price = bjerksund_stensland_american(S, K, T, r, v, kind)
            
        d1 = (math.log(S / K) + (r + 0.5 * v ** 2) * T) / (v * math.sqrt(T))
        if kind == 'Call': delta = norm_cdf(d1)
        else: delta = norm_cdf(d1) - 1
            
        return price, delta
    except: return 0.0, 0.0

def check_market_hours():
    sydney_tz = pytz.timezone('Australia/Sydney')
    now = datetime.now(sydney_tz)
    if now.weekday() >= 5: return False
    return time(10, 0) <= now.time() <= time(16, 10)

st.session_state.is_market_open = check_market_hours()

def fetch_data(t):
    clean = t.upper().replace(".AX", "").strip()
    sym = f"{clean}.AX"
    div_info = None
    spot = 0.0
    
    if st.session_state.manual_spot:
        spot = st.session_state.spot_price
        try:
            tk = yf.Ticker(sym)
            info = tk.info
            if 'exDividendDate' in info and info['exDividendDate']:
                ex_ts = info['exDividendDate']
                ex_date = datetime.fromtimestamp(ex_ts)
                amt = info.get('lastDividendValue', info.get('dividendRate', 0)/2)
                if ex_date > datetime.now(): div_info = {'amount': amt, 'date': ex_date}
        except: pass
        return "MANUAL", spot, div_info

    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="1d")
        if not hist.empty: spot = float(hist['Close'].iloc[-1])
        else: return "ERROR", 0.0, None
            
        try:
            info = tk.info
            if 'exDividendDate' in info and info['exDividendDate']:
                ex_ts = info['exDividendDate']
                ex_date = datetime.fromtimestamp(ex_ts)
                if ex_date > datetime.now():
                    amt = info.get('lastDividendValue', 0)
                    if amt == 0: amt = info.get('dividendRate', 0) / 2
                    div_info = {'amount': amt, 'date': ex_date}
        except: pass
        
        return "YAHOO", spot, div_info
    except: return "ERROR", 0.0, None

# --- 6. HEADER ---
status_parts = st.session_state.sheet_msg.split("|")
status_txt = status_parts[1] if len(status_parts) > 1 else status_parts[0]
mkt_status = "🟢 OPEN" if st.session_state.is_market_open else "🔴 CLOSED"

div_display_txt = ""
if st.session_state.div_info:
    d = st.session_state.div_info
    d_date = d['date'].strftime("%d %b")
    div_display_txt = f" | 💰 Auto Div: ${d['amount']:.2f} on {d_date}"

with st.container():
    st.markdown(f"""
    <div class="header-box">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="header-title">TradersCircle Options Calculator</div>
                <div class="header-sub">Option Strategy Builder v9.1</div>
            </div>
            <div style="text-align: right;">
                <div class="header-title" style="color: #4ade80;">${st.session_state.spot_price:.2f}</div>
                <div class="header-sub">{st.session_state.ticker if st.session_state.ticker else "---"}</div>
                <span class="status-tag">{mkt_status}{div_display_txt}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 7. CONTROLS ---
c1, c2, c3 = st.columns([1, 1, 2], gap="medium")
with c1: 
    display_val = st.session_state.preselect_code if st.session_state.preselect_code else st.session_state.ticker
    query = st.text_input("Ticker or Option Code", value=display_val, placeholder="e.g., BHP or BHPJ84")

with c2:
    if st.session_state.ticker:
        new_spot = st.number_input("Spot Price ($)", value=float(st.session_state.spot_price), format="%.2f", step=0.01)
        if new_spot != st.session_state.spot_price:
            st.session_state.spot_price = new_spot
            st.session_state.manual_spot = True
    else:
        st.write("")

with c3:
    st.write("") 
    st.write("")
    if st.button("LOAD OPTIONS", type="primary", use_container_width=True) or (query and query.upper() != display_val):
        if not query:
            st.warning("Please enter a ticker or option code.")
        else:
            query_upper = query.upper().strip()
            st.session_state.manual_spot = False
            
            # --- SMART SEARCH EXTRACTION LOGIC ---
            ref = st.session_state.ref_data
            ticker_to_fetch = query_upper # Fallback
            
            if ref is not None:
                match = ref[ref['Code'] == query_upper]
                if not match.empty:
                    # Found exact option code in sheet
                    ticker_to_fetch = str(match.iloc[0]['Ticker']).strip()
                    st.session_state.preselect_expiry = match.iloc[0]['Expiry'].strftime("%Y-%m-%d")
                    st.session_state.preselect_strike = float(match.iloc
