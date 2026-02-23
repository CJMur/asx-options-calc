# ==========================================
# TradersCircle Options Calculator
# VERSION: 10.22 (Forward Curve & Live RBA Rate)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, time, timedelta
import pytz
import math
import uuid
import requests
import re

# --- 1. CONFIGURATION & THEME ---
st.set_page_config(layout="wide", page_title="TradersCircle Options")
RAW_SHEET_URL = "https://docs.google.com/spreadsheets/d/1d9FQ5mn--MSNJ_WJkU--IvoSRU0gQBqE0f9s9zEb0Q4/edit?usp=sharing"

# --- CSS STYLING ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; padding-bottom: 5rem !important; }
    
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
    
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #1DBFD2 !important; border: none; color: white !important; font-weight: bold;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #16aebf !important;
    }
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #f8fafc !important; color: #334155 !important; border: 1px solid #cbd5e1;
    }
    
    /* --- SLIDER COLOR FIX (Electric Blue) --- */
    div[data-baseweb="slider"] > div > div > div { background-color: #0050FF !important; }
    div[role="slider"] { background-color: #0050FF !important; border: none !important; box-shadow: none !important; }
    div[data-testid="stSlider"] svg path { fill: #0050FF !important; stroke: #0050FF !important; }
    div[data-testid="stSlider"] p { color: white !important; }
    input[type=range] { accent-color: #0050FF !important; }
    
    /* Dataframe Row Selection Highlight (Teal) */
    [data-testid="stDataFrame"] [aria-selected="true"] > div {
        background-color: rgba(29, 191, 210, 0.4) !important;
        color: white !important;
    }
    
    .stDataFrame { border: none !important; }
    .trade-header {
        font-weight: 700; color: #94a3b8; font-size: 12px; text-transform: uppercase;
        margin-bottom: 5px; cursor: help; user-select: none;
    }
    
    /* Center align markdown text and allow background shading */
    .strategy-text { 
        user-select: none; 
        display: flex; 
        align-items: center; 
        min-height: 40px; 
        padding: 0 8px;
        border-radius: 4px;
        width: 100%;
    }
    
    /* Shrink the +/- buttons so rows stay slim */
    button[kind="secondary"] {
        padding: 0rem 0.5rem !important; min-height: 0px !important; height: 32px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "" 
if 'spot_price' not in st.session_state: st.session_state.spot_price = 0.0
if 'forward_price' not in st.session_state: st.session_state.forward_price = 0.0
if 'chain_obj' not in st.session_state: st.session_state.chain_obj = None
if 'ref_data' not in st.session_state: st.session_state.ref_data = None
if 'sheet_msg' not in st.session_state: st.session_state.sheet_msg = "Initializing..."
if 'manual_spot' not in st.session_state: st.session_state.manual_spot = False
if 'is_market_open' not in st.session_state: st.session_state.is_market_open = True
if 'div_info' not in st.session_state: st.session_state.div_info = None
if 'matrix_vol_mod' not in st.session_state: st.session_state.matrix_vol_mod = 0.0

if 'preselect_code' not in st.session_state: st.session_state.preselect_code = None
if 'preselect_expiry' not in st.session_state: st.session_state.preselect_expiry = None
if 'preselect_strike' not in st.session_state: st.session_state.preselect_strike = None

TOOLTIPS = {
    "Theo": "The theoretical fair value of the option calculated using the Black-Scholes or Bjerksund-Stensland pricing model.",
    "IV": "Implied Volatility: The market's forecast of a likely movement in the security's price.",
    "Delta": "The rate of change of the option price with respect to the underlying asset's price.",
    "Strike": "The set price at which the option contract can be exercised.",
    "Code": "The unique ASX exchange ticker symbol for this specific option contract.",
    "Entry": "The price you entered the trade at (or current market price).",
    "Premium": "The total cost or credit for the trade. Calculated as Price × Quantity × Contract Multiplier.",
    "Margin": "The estimated collateral required to hold this position."
}

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=86400)
def fetch_rba_cash_rate():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get("https://www.rba.gov.au/", headers=headers, timeout=5)
        match = re.search(r'Cash rate target.*?(\d+\.\d+)\s*%', res.text, re.IGNORECASE | re.DOTALL)
        if match: return float(match.group(1))
        return 3.85
    except:
        return 3.85

global_rba_rate = fetch_rba_cash_rate()

@st.cache_data(ttl=600)
def load_sheet(raw_url):
    try:
        csv_url = f"{raw_url.split('/edit')[0]}/export?format=csv&gid=0" if "/edit" in raw_url else raw_url
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

        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip().replace('NAN', np.nan).replace('', np.nan)
        df['Code'] = df['Code'].astype(str).str.upper().str.strip().replace('NAN', np.nan).replace('', np.nan)
        
        if 'Type' in df.columns:
            raw_type = df['Type'].astype(str).str.strip().str.upper()
            df['Type'] = np.where(raw_type.str.startswith('C'), 'Call', 'Put')
        else:
            df['Type'] = 'Call'
            
        if 'Style' in df.columns:
            raw_style = df['Style'].astype(str).str.strip().str.upper()
            df['Style'] = np.where(raw_style.str.startswith('E'), 'European', 'American')
        else:
            df['Style'] = 'American'
            
        df['Strike'] = pd.to_numeric(df['Strike'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').round(3)
        df['Expiry'] = pd.to_datetime(df['Expiry'], dayfirst=True, errors='coerce')
        
        if 'Vol' in df.columns:
            df['Vol'] = pd.to_numeric(df['Vol'].astype(str).str.replace('%', ''), errors='coerce')
            mask = df['Vol'] <= 1.0 
            df.loc[mask, 'Vol'] = df.loc[mask, 'Vol'] * 100
        else:
            df['Vol'] = 30.0
            
        df['Settlement'] = pd.to_numeric(df['Settlement'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce') if 'Settlement' in df.columns else 0.0

        scen_cols = [c for c in df.columns if 'Scenario' in str(c)]
        if scen_cols:
            scen_df = df[scen_cols].copy()
            for col in scen_cols:
                clean_str = scen_df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                scen_df[col] = pd.to_numeric(clean_str, errors='coerce')
            df['UnitMargin'] = scen_df.min(axis=1, skipna=True).fillna(0.0)
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

def black_scholes_european(S, K, T, r, sigma, option_type, q=0.0):
    if T <= 0: return max(0, S - K) if option_type == 'Call' else max(0, K - S)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'Call':
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)

def bjerksund_stensland_american(S, K, T, r, sigma, option_type):
    bs_price = black_scholes_european(S, K, T, r, sigma, option_type, q=0.0)
    if option_type == 'Call': return bs_price 
    else: return max(bs_price, max(0, K - S))

def calculate_price_and_delta(style, kind, simulated_spot, strike, time_days, vol_pct, current_spot, forward_override):
    if simulated_spot <= 0 or strike <= 0 or time_days < 0:
        return 0.0, 0.0
        
    r = global_rba_rate / 100.0
    q = 0.0
    is_xjo = (st.session_state.ticker == 'XJO')
    
    try:
        S = float(simulated_spot)
        K = float(strike)
        T = time_days / 365.0
        
        # Hard expiry value logic
        if T <= 0.0:
            price = max(0.0, S - K) if kind == 'Call' else max(0.0, K - S)
            delta = 0.0
            if kind == 'Call' and S > K: delta = 1.0
            elif kind == 'Put' and S < K: delta = -1.0
            return price, delta

        v = vol_pct / 100.0
        
        if is_xjo:
            style = 'EUROPEAN' 
            # If user provided a forward price, back-calculate the implied dividend yield (q)
            if forward_override and forward_override > 0 and current_spot > 0:
                implied_q = r - (math.log(forward_override / current_spot) / T)
                q = implied_q
            else:
                q = 0.04 # Fallback generic 4% yield
        elif st.session_state.div_info:
            d_info = st.session_state.div_info
            if d_info['amount'] > 0 and d_info['date']:
                days_to_div = (d_info['date'] - datetime.now()).days
                if 0 <= days_to_div < time_days:
                    t_div = days_to_div / 365.0
                    div_pv = d_info['amount'] * math.exp(-r * t_div)
                    S = max(0.01, S - div_pv)
        
        if style.upper() == 'EUROPEAN':
            price = black_scholes_european(S, K, T, r, v, kind, q)
        else:
            price = bjerksund_stensland_american(S, K, T, r, v, kind)
            
        d1 = (math.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * math.sqrt(T))
        
        if kind == 'Call':
            delta = math.exp(-q * T) * norm_cdf(d1)
        else:
            delta = math.exp(-q * T) * (norm_cdf(d1) - 1)
            
        return price, delta
    except: 
        return 0.0, 0.0

def check_market_hours():
    sydney_tz = pytz.timezone('Australia/Sydney')
    now = datetime.now(sydney_tz)
    return False if now.weekday() >= 5 else time(10, 0) <= now.time() <= time(16, 10)

st.session_state.is_market_open = check_market_hours()

def fetch_data(t):
    clean = t.upper().replace(".AX", "").strip()
    sym = "^AXJO" if clean == "XJO" else f"{clean}.AX"
    div_info, spot = None, 0.0
    
    if st.session_state.manual_spot:
        spot = st.session_state.spot_price
        try:
            info = yf.Ticker(sym).info
            if 'exDividendDate' in info and info['exDividendDate']:
                ex_ts = info['exDividendDate']
                if isinstance(ex_ts, (int, float)):
                    ex_date = datetime.fromtimestamp(ex_ts)
                    amt = info.get('lastDividendValue', info.get('dividendRate', 0)/2)
                    if ex_date > datetime.now(): div_info = {'amount': amt, 'date': ex_date}
        except: pass
        return "MANUAL", spot, div_info

    try:
        tk = yf.Ticker(sym)
        info = tk.info
        
        spot = float(info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose', 0.0))))
        
        if spot == 0.0:
            hist = tk.history(period="1d")
            if not hist.empty: 
                spot = float(hist['Close'].iloc[-1])
            
        if 'exDividendDate' in info and info['exDividendDate']:
            ex_ts = info['exDividendDate']
            if isinstance(ex_ts, (int, float)):
                ex_date = datetime.fromtimestamp(ex_ts)
                if ex_date > datetime.now():
                    amt = info.get('lastDividendValue', 0)
                    if amt == 0: amt = info.get('dividendRate', 0) / 2
                    div_info = {'amount': amt, 'date': ex_date}
                    
        return "YAHOO", spot, div_info
    except: 
        return "ERROR", 0.0, None

# --- 6. HEADER ---
mkt_status = "🟢 OPEN" if st.session_state.is_market_open else "🔴 CLOSED"

if st.session_state.ticker == 'XJO':
    div_display_txt = f" | 🏦 RBA Rate: {global_rba_rate}%"
elif st.session_state.div_info:
    div_display_txt = f" | 💰 Auto Div: ${st.session_state.div_info['amount']:.2f}"
else:
    div_display_txt = ""

st.markdown(f"""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="header-title">TradersCircle Options Calculator</div>
            <div class="header-sub">Option Strategy Builder v10.22</div>
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
        # Layout Spot and Forward Inputs
        s1, s2 = st.columns(2)
        new_spot = s1.number_input("Spot Price ($)", value=float(st.session_state.spot_price), format="%.2f", step=0.01)
        if new_spot != st.session_state.spot_price:
            st.session_state.spot_price = new_spot
            st.session_state.manual_spot = True
            
        new_fwd = s2.number_input("Fwd Price (Opt)", value=float(st.session_state.forward_price), format="%.2f", step=1.0)
        st.session_state.forward_price = new_fwd
    else: st.write("")

with c3:
    st.write(""); st.write("")
    if st.button("LOAD OPTIONS", type="primary", use_container_width=True) or (query and query.upper() != display_val):
        if not query: st.warning("Please enter a ticker or option code.")
        else:
            query_upper = query.upper().strip()
            
            ref = st.session_state.ref_data
            ticker_to_fetch = query_upper
            
            if ref is not None:
                match = ref[ref['Code'] == query_upper]
                if not match.empty:
                    ticker_to_fetch = str(match.iloc[0]['Ticker']).strip()
                    st.session_state.preselect_expiry = match.iloc[0]['Expiry'].strftime("%Y-%m-%d")
                    st.session_state.preselect_strike = float(match.iloc[0]['Strike'])
                    st.session_state.preselect_code = query_upper
                else:
                    tickers = ref['Ticker'].unique()
                    possible_matches = [t for t in tickers if query_upper.startswith(t)]
                    if possible_matches:
                        best_match = max(possible_matches, key=len)
                        if len(query_upper) > len(best_match):
                            ticker_to_fetch = best_match
                            st.session_state.preselect_code = query_upper
                            st.session_state.preselect_expiry = None
                            st.session_state.preselect_strike = None
                    else:
                        st.session_state.preselect_expiry = None
                        st.session_state.preselect_strike = None
                        st.session_state.preselect_code = None

            st.session_state.ticker = ticker_to_fetch

            with st.spinner("Fetching Market Data..."):
                source, px, div_data = fetch_data(st.session_state.ticker)
                
                if px > 0:
                    st.session_state.spot_price = px
                    st.session_state.manual_spot = False
                elif not st.session_state.manual_spot:
                    st.warning(f"Could not fetch live price for {st.session_state.ticker}. Please enter it manually.")
                
                st.session_state.div_info = div_data
                st.session_state.data_source = source
                data, msg = load_sheet(RAW_SHEET_URL)
                st.session_state.ref_data = data
                st.session_state.sheet_msg = msg
                st.session_state.is_market_open = check_market_hours()
                st.rerun()

# --- 9. CHAIN DISPLAY ---
df_view = pd.DataFrame()
current_exp = None

if st.session_state.ref_data is not None and st.session_state.ticker:
    ref = st.session_state.ref_data
    tkr = st.session_state.ticker.replace(".AX", "")
    subset = ref[ref['Ticker'] == tkr]
    
    today = datetime.now().replace(hour=0, minute=0, second=0)
    subset = subset[subset['Expiry'] >= today]
    
    if not subset.empty:
        valid_exps = sorted(subset['Expiry'].unique())
        exp_map = {d.strftime("%Y-%m-%d"): d for d in valid_exps}
        exp_list = list(exp_map.keys())
        
        default_idx = exp_list.index(st.session_state.preselect_expiry) if st.session_state.preselect_expiry in exp_list else None
        current_exp = st.selectbox("Expiry", exp_list, index=default_idx, placeholder="Select Expiry")
        
        if current_exp:
            target_dt = exp_map[current_exp]
            days_diff = (target_dt - today).days
            day_chain = subset[subset['Expiry'] == target_dt].copy()
            
            def calc_row_metrics(row):
                vol = float(row['Vol']) if pd.notna(row['Vol']) else 30.0
                style = row.get('Style', 'American')
                margin = float(row['UnitMargin']) if 'UnitMargin' in row else 0.0
                
                px, delta = calculate_price_and_delta(
                    style, row['Type'], st.session_state.spot_price, row['Strike'], 
                    days_diff, vol, st.session_state.spot_price, st.session_state.forward_price
                )
                
                return pd.Series([px, delta, vol, margin])

            metrics = day_chain.apply(calc_row_metrics, axis=1)
            metrics.columns = ['Calc_Price', 'Calc_Delta', 'Calc_Vol', 'Calc_Margin']
            day_chain = pd.concat([day_chain, metrics], axis=1)
            
            calls = day_chain[day_chain['Type'] == 'Call'].set_index('Strike')
            puts = day_chain[day_chain['Type'] == 'Put'].set_index('Strike')
            
            all_strikes = sorted(list(set(calls.index) | set(puts.index)))
            df_view = pd.DataFrame({'STRIKE': all_strikes})
            
            df_view['C_Code'] = df_view['STRIKE'].map(calls['Code'])
            df_view['C_Style_Full'] = df_view['STRIKE'].map(calls['Style']).fillna('American')
            df_view['C_Price'] = df_view['STRIKE'].map(calls['Calc_Price'])
            df_view['C_Vol'] = df_view['STRIKE'].map(calls['Calc_Vol'])
            df_view['C_Delta'] = df_view['STRIKE'].map(calls['Calc_Delta'])
            df_view['C_Margin'] = df_view['STRIKE'].map(calls['Calc_Margin'])
            
            df_view['P_Code'] = df_view['STRIKE'].map(puts['Code'])
            df_view['P_Style_Full'] = df_view['STRIKE'].map(puts['Style']).fillna('American')
            df_view['P_Price'] = df_view['STRIKE'].map(puts['Calc_Price'])
            df_view['P_Vol'] = df_view['STRIKE'].map(puts['Calc_Vol'])
            df_view['P_Delta'] = df_view['STRIKE'].map(puts['Calc_Delta'])
            df_view['P_Margin'] = df_view['STRIKE'].map(puts['Calc_Margin'])

if not df_view.empty and current_exp:
    center = st.session_state.preselect_strike if (st.session_state.preselect_strike and current_exp == st.session_state.preselect_expiry) else st.session_state.spot_price
        
    if center > 0:
        df_view['Diff'] = abs(df_view['STRIKE'] - center)
        atm_idx = df_view['Diff'].idxmin()
        df_view = df_view.iloc[max(0, atm_idx - 12):min(len(df_view), atm_idx + 13)].drop(columns=['Diff'])
    
    st.markdown(f"**Chain: {current_exp}**")
    
    disp = df_view[['C_Code', 'C_Price', 'C_Vol', 'C_Delta', 'STRIKE', 'P_Price', 'P_Vol', 'P_Delta', 'P_Code']].copy()
    
    def highlight_itm(row):
        spot = st.session_state.spot_price
        strike = row['STRIKE']
        target_code = str(st.session_state.preselect_code)
        
        styles = []
        for col in row.index:
            s = ""
            if col in ['C_Code', 'C_Price', 'C_Vol', 'C_Delta'] and strike < spot:
                s += "background-color: rgba(74, 222, 128, 0.10); "
            elif col in ['P_Code', 'P_Price', 'P_Vol', 'P_Delta'] and strike > spot:
                s += "background-color: rgba(74, 222, 128, 0.10); "
            
            if col == 'STRIKE':
                s += "font-weight: bold; background-color: rgba(255,255,255,0.05); "
            
            if col in ['C_Code', 'P_Code'] and str(row[col]) == target_code and target_code != "None":
                s += "color: white; border: 1px solid #1DBFD2; background-color: rgba(29, 191, 210, 0.4); "
                
            styles.append(s)
        return styles

    styled_disp = disp.style.apply(highlight_itm, axis=1).format({
        'C_Price': '{:.3f}', 'C_Vol': '{:.1f}', 'C_Delta': '{:.3f}', 'STRIKE': '{:.3f}',
        'P_Price': '{:.3f}', 'P_Vol': '{:.1f}', 'P_Delta': '{:.3f}'
    })

    selection = st.dataframe(
        styled_disp,
        column_config={
            "C_Code": st.column_config.TextColumn("Call Code", help=TOOLTIPS["Code"]),
            "C_Price": st.column_config.NumberColumn("Theo", format="%.3f", help=TOOLTIPS["Theo"]),
            "C_Vol": st.column_config.NumberColumn("IV %", format="%.1f", help=TOOLTIPS["IV"]),
            "C_Delta": st.column_config.NumberColumn("Delta", format="%.3f", help=TOOLTIPS["Delta"]),
            "STRIKE": st.column_config.NumberColumn("Strike", format="%.3f", help=TOOLTIPS["Strike"]),
            "P_Price": st.column_config.NumberColumn("Theo", format="%.3f", help=TOOLTIPS["Theo"]),
            "P_Vol": st.column_config.NumberColumn("IV %", format="%.1f", help=TOOLTIPS["IV"]),
            "P_Delta": st.column_config.NumberColumn("Delta", format="%.3f", help=TOOLTIPS["Delta"]),
            "P_Code": st.column_config.TextColumn("Put Code", help=TOOLTIPS["Code"]),
        },
        hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row"
    )
    
    if selection.selection['rows']:
        idx = selection.selection['rows'][0]
        row = df_view.iloc[idx]
        days_diff = (datetime.strptime(current_exp, "%Y-%m-%d") - datetime.now()).days
        
        st.write("")
        q_c, b1_c, b2_c, b3_c, b4_c, _ = st.columns([1.5, 1, 1, 1, 1, 3], gap="small")

        with q_c:
            trade_qty = st.number_input("Trade Quantity", min_value=1, value=1, step=1)
        
        def add(side, kind, px, code_hint, delta_val, qty_val, style_full):
            st.session_state.legs.append({
                "id": str(uuid.uuid4()),
                "Qty": qty_val if side == "Buy" else -qty_val, 
                "Type": kind, 
                "Style": str(style_full),
                "Strike": float(row['STRIKE']), 
                "Expiry": days_diff, 
                "ExpDateStr": current_exp, 
                "Vol": float(row['C_Vol'] if kind == 'Call' else row['P_Vol']), 
                "Entry": float(px), 
                "Code": str(code_hint), 
                "Delta": float(delta_val), 
                "MarginUnit": float(row['C_Margin'] if kind == 'Call' else row['P_Margin'])
            })
            st.rerun()
            
        c_c = str(row['C_Code']) if pd.notna(row['C_Code']) else "N/A"
        p_c = str(row['P_Code']) if pd.notna(row['P_Code']) else "N/A"
        c_s = str(row['C_Style_Full'])
        p_s = str(row['P_Style_Full'])
        
        btn_spacer = "<div style='height: 28px;'></div>"
        
        with b1_c:
             st.markdown(btn_spacer, unsafe_allow_html=True)
             if st.button(f"Buy Call", use_container_width=True): add("Buy", "Call", row['C_Price'], c_c, row['C_Delta'], trade_qty, c_s)
        with b2_c:
             st.markdown(btn_spacer, unsafe_allow_html=True)
             if st.button(f"Sell Call", use_container_width=True): add("Sell", "Call", row['C_Price'], c_c, row['C_Delta'], trade_qty, c_s)
        with b3_c:
             st.markdown(btn_spacer, unsafe_allow_html=True)
             if st.button(f"Buy Put", use_container_width=True): add("Buy", "Put", row['P_Price'], p_c, row['P_Delta'], trade_qty, p_s)
        with b4_c:
             st.markdown(btn_spacer, unsafe_allow_html=True)
             if st.button(f"Sell Put", use_container_width=True): add("Sell", "Put", row['P_Price'], p_c, row['P_Delta'], trade_qty, p_s)

# --- 10. STRATEGY ---
if st.session_state.legs:
    st.markdown("---")
    st.subheader("Strategy")
    
    contract_multiplier = 10 if st.session_state.ticker == 'XJO' else 100
    
    h_col_spec = [0.8, 1.0, 0.7, 0.9, 1.2, 2.7, 1.2, 1.0, 1.0, 1.0, 1.2, 1.5, 0.5]
    cols_header = st.columns(h_col_spec)
    
    with cols_header[0]: st.markdown('<div class="trade-header" title="Quantity (Editable)">Qty</div>', unsafe_allow_html=True)
    with cols_header[1]: st.markdown(f'<div class="trade-header" title="{TOOLTIPS["Code"]}">Code</div>', unsafe_allow_html=True)
    with cols_header[2]: st.markdown('<div class="trade-header" title="American or European">Style</div>', unsafe_allow_html=True)
    with cols_header[3]: st.markdown('<div class="trade-header" title="Call or Put">Type</div>', unsafe_allow_html=True)
    with cols_header[4]: st.markdown('<div class="trade-header" title="Date of Expiry">Expiry</div>', unsafe_allow_html=True)
    with cols_header[5]: st.markdown(f'<div class="trade-header" title="Smart Step Strike">Strike</div>', unsafe_allow_html=True)
    with cols_header[6]: st.markdown(f'<div class="trade-header" title="Implied Volatility (Editable)">Vol</div>', unsafe_allow_html=True)
    with cols_header[7]: st.markdown(f'<div class="trade-header" title="{TOOLTIPS["Entry"]}">Entry</div>', unsafe_allow_html=True)
    with cols_header[8]: st.markdown(f'<div class="trade-header" title="{TOOLTIPS["Theo"]}">Theo</div>', unsafe_allow_html=True)
    with cols_header[9]: st.markdown(f'<div class="trade-header" title="{TOOLTIPS["Delta"]}">Delta</div>', unsafe_allow_html=True)
    with cols_header[10]: st.markdown(f'<div class="trade-header" title="{TOOLTIPS["Premium"]}">Premium</div>', unsafe_allow_html=True)
    with cols_header[11]: st.markdown(f'<div class="trade-header" title="{TOOLTIPS["Margin"]}">Expected Margin</div>', unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 0 0 10px 0; border-top: 1px solid #334155;'>", unsafe_allow_html=True)

    total_delta, total_premium, raw_theo_sum, total_margin = 0, 0, 0, 0
    max_qty = max(abs(leg['Qty']) for leg in st.session_state.legs) if st.session_state.legs else 1
    
    for i, leg in enumerate(st.session_state.legs):
        if 'id' not in leg: leg['id'] = str(uuid.uuid4())
        if 'Style' not in leg: leg['Style'] = 'American'
        if 'ExpDateStr' not in leg: leg['ExpDateStr'] = (datetime.now() + timedelta(days=leg['Expiry'])).strftime("%Y-%m-%d")
        
        new_theo, new_delta = calculate_price_and_delta(
            leg['Style'], leg['Type'], st.session_state.spot_price, leg['Strike'], 
            leg['Expiry'], leg['Vol'], st.session_state.spot_price, st.session_state.forward_price
        )
        
        net_delta = leg['Qty'] * new_delta * contract_multiplier
        premium = -(leg['Qty'] * leg['Entry'] * contract_multiplier)
        row_margin = leg.get('MarginUnit', 0.0) * abs(leg['Qty']) 
        
        total_delta += net_delta
        total_premium += premium
        total_margin += row_margin
        raw_theo_sum += leg['Qty'] * new_theo
        
        p_color = '#4ade80' if premium >= 0 else '#f87171'
        m_color = '#4ade80' if row_margin >= 0 else '#f87171'
        
        row_bg = "rgba(74, 222, 128, 0.10)" if leg['Qty'] > 0 else "rgba(248, 113, 113, 0.10)"
        
        c = st.columns(h_col_spec)
        
        with c[0]: 
            new_qty = st.number_input("Qty", value=int(leg['Qty']), step=1, key=f"qty_{leg['id']}", label_visibility="collapsed")
            if new_qty != leg['Qty']:
                st.session_state.legs[i]['Qty'] = new_qty
                st.rerun()
                
        with c[1]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>{leg['Code']}</div>", unsafe_allow_html=True)
        with c[2]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>{str(leg['Style'])[0]}</div>", unsafe_allow_html=True)
        with c[3]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg}; font-weight:600;'>{leg['Type']}</div>", unsafe_allow_html=True)
        with c[4]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>{leg['ExpDateStr']}</div>", unsafe_allow_html=True)
        
        with c[5]: 
            tkr = st.session_state.ticker.replace(".AX", "")
            subset = st.session_state.ref_data[
                (st.session_state.ref_data['Ticker'] == tkr) & 
                (st.session_state.ref_data['Type'] == leg['Type']) & 
                (st.session_state.ref_data['Expiry'].dt.strftime("%Y-%m-%d") == leg['ExpDateStr'])
            ]
            
            available_strikes = sorted(subset['Strike'].unique().tolist())
            current_strike = float(leg['Strike'])
            
            if available_strikes:
                closest_strike = min(available_strikes, key=lambda x: abs(x - current_strike))
                if abs(closest_strike - current_strike) < 0.01:
                    current_strike = closest_strike
                elif current_strike not in available_strikes:
                    available_strikes.append(current_strike)
                    available_strikes = sorted(available_strikes)
            else:
                available_strikes = [current_strike]
                
            current_idx = available_strikes.index(current_strike)
            
            sc1, sc2, sc3, _ = st.columns([1.2, 0.6, 0.6, 1.0], gap="small")
            with sc1:
                st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>{current_strike:.2f}</div>", unsafe_allow_html=True)
            with sc2:
                dec = st.button("▼", key=f"dn_{leg['id']}", use_container_width=True)
            with sc3:
                inc = st.button("▲", key=f"up_{leg['id']}", use_container_width=True)
                
            new_strike = None
            if dec and current_idx > 0:
                new_strike = available_strikes[current_idx - 1]
            elif inc and current_idx < len(available_strikes) - 1:
                new_strike = available_strikes[current_idx + 1]
                
            if new_strike is not None and new_strike != current_strike:
                st.session_state.legs[i]['Strike'] = new_strike
                match = subset[subset['Strike'] == new_strike]
                if not match.empty:
                    new_vol = float(match.iloc[0]['Vol'])
                    new_style = match.iloc[0].get('Style', 'American')
                    
                    st.session_state.legs[i]['Code'] = str(match.iloc[0]['Code'])
                    st.session_state.legs[i]['Vol'] = new_vol
                    st.session_state.legs[i]['Style'] = new_style
                    st.session_state.legs[i]['MarginUnit'] = float(match.iloc[0]['UnitMargin'])
                    
                    matched_theo, _ = calculate_price_and_delta(
                        new_style, leg['Type'], st.session_state.spot_price, new_strike, 
                        leg['Expiry'], new_vol, st.session_state.spot_price, st.session_state.forward_price
                    )
                    st.session_state.legs[i]['Entry'] = matched_theo
                else:
                    st.session_state.legs[i]['Code'] = "N/A"
                st.rerun()
                
        with c[6]: 
            new_vol_input = st.number_input("Vol", value=float(leg['Vol']), step=0.5, format="%.1f", key=f"vol_{leg['id']}", label_visibility="collapsed")
            if new_vol_input != leg['Vol']:
                st.session_state.legs[i]['Vol'] = new_vol_input
                calibrated_theo, _ = calculate_price_and_delta(
                    leg['Style'], leg['Type'], st.session_state.spot_price, leg['Strike'], 
                    leg['Expiry'], new_vol_input, st.session_state.spot_price, st.session_state.forward_price
                )
                st.session_state.legs[i]['Entry'] = calibrated_theo
                st.rerun()
                
        with c[7]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>${leg['Entry']:.3f}</div>", unsafe_allow_html=True)
        with c[8]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>${new_theo:.3f}</div>", unsafe_allow_html=True)
        with c[9]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>{net_delta:.2f}</div>", unsafe_allow_html=True)
        with c[10]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg}; color:{p_color}; font-weight:600;'>${premium:.2f}</div>", unsafe_allow_html=True)
        with c[11]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg}; color:{m_color}; font-weight:600;'>${row_margin:.2f}</div>", unsafe_allow_html=True)
        with c[12]:
            if st.button("✕", key=f"d_{leg['id']}"):
                st.session_state.legs.pop(i)
                st.rerun()
                
        st.markdown("<hr style='margin: 5px 0; border-top: 1px solid #1e293b;'>", unsafe_allow_html=True)

    strategy_net_theo = raw_theo_sum / max_qty if max_qty != 0 else 0.0

    with st.container():
        f = st.columns(h_col_spec)
        with f[1]: st.markdown("<div class='strategy-text' style='font-weight:bold;'>TOTAL STRATEGY</div>", unsafe_allow_html=True)
        with f[8]: st.markdown(f"<div class='strategy-text' style='font-weight:bold;'>${strategy_net_theo:.3f}</div>", unsafe_allow_html=True)
        with f[9]: st.markdown(f"<div class='strategy-text' style='font-weight:bold;'>{total_delta:,.2f}</div>", unsafe_allow_html=True)
        with f[10]: st.markdown(f"<div class='strategy-text' style='color:{'#4ade80' if total_premium >= 0 else '#f87171'}; font-weight:bold'>${total_premium:,.2f}</div>", unsafe_allow_html=True)
        with f[11]: st.markdown(f"<div class='strategy-text' style='color:{'#4ade80' if total_margin >= 0 else '#f87171'}; font-weight:bold'>${total_margin:,.2f}</div>", unsafe_allow_html=True)

    # --- MATRIX ---
    st.markdown("---")
    st.subheader("Payoff Matrix")
    m1, m2 = st.columns(2)
    time_step = m1.slider("Step (Days)", 1, 30, 1)
    range_pct = m2.select_slider("Price Step (% per row)", options=[0.005, 0.01, 0.02, 0.03, 0.05], value=0.01, format_func=lambda x: f"{x*100:.1f}%")
    
    with m1:
        st.caption("Simulate Volatility Shift:")
        c_v1, c_v2, c_v3 = st.columns(3)
        if c_v1.button("IV -10%"): st.session_state.matrix_vol_mod -= 10
        if c_v2.button("IV Flat"): st.session_state.matrix_vol_mod = 0
        if c_v3.button("IV +10%"): st.session_state.matrix_vol_mod += 10
        st.caption(f"Current Shift: {st.session_state.matrix_vol_mod:+}%")

    spot = st.session_state.spot_price
    prices = [spot * (1 + range_pct * i) for i in range(6, -7, -1)]
    dates = [d * time_step for d in range(7)]
    
    matrix_data = []
    for p in prices:
        is_spot = math.isclose(p, spot, rel_tol=1e-5)
        row_label = f"» ${p:.2f} (SPOT) «" if is_spot else f"${p:.2f}"
        
        row = {"Price": row_label}
        for d in dates:
            pnl = 0
            for leg in st.session_state.legs:
                sim_vol = max(1.0, leg['Vol'] + st.session_state.matrix_vol_mod)
                rem_days = max(0, leg['Expiry'] - d)
                exit_px, _ = calculate_price_and_delta(
                    leg['Style'], leg['Type'], p, leg['Strike'], 
                    rem_days, sim_vol, st.session_state.spot_price, st.session_state.forward_price
                )
                pnl += (exit_px - leg['Entry']) * leg['Qty'] * contract_multiplier
            
            col_name = (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
            if d == 0: col_name = f"Today ({col_name})"
            row[col_name] = pnl
        matrix_data.append(row)
        
    df_mx = pd.DataFrame(matrix_data).set_index("Price")
    
    capital_at_risk = total_margin if total_premium >= 0 else abs(total_premium)
    
    def format_pnl(val):
        if capital_at_risk <= 0.001:
            return f"${val:,.0f} (0.0%)"
        pct = (val / capital_at_risk) * 100
        sign = "+" if val > 0 else ""
        return f"${val:,.0f} ({sign}{pct:.1f}%)"

    def make_heatmap(df):
        max_val = df.max().max()
        min_val = df.min().min()
        abs_max = max(abs(max_val), abs(min_val), 1)
        
        def style_cell(val):
            if val > 0:
                intensity = min(val / abs_max, 1.0)
                alpha = 0.05 + 0.35 * intensity
                return f"background-color: rgba(74, 222, 128, {alpha:.2f});"
            elif val < 0:
                intensity = min(abs(val) / abs_max, 1.0)
                alpha = 0.05 + 0.35 * intensity
                return f"background-color: rgba(248, 113, 113, {alpha:.2f});"
            return ""
        
        return df.applymap(style_cell)

    st.dataframe(df_mx.style.apply(make_heatmap, axis=None).format(format_pnl), use_container_width=True, height=500)

    # CHART
    st.markdown("### Payoff Chart")
    
    chart_spread = range_pct * 6 * 1.2
    chart_prices = np.linspace(spot * (1 - chart_spread), spot * (1 + chart_spread), 100)
    
    pnl_today = []
    pnl_expiry = []
    for p in chart_prices:
        val_t0 = 0
        val_tF = 0
        for leg in st.session_state.legs:
            price_t0, _ = calculate_price_and_delta(
                leg['Style'], leg['Type'], p, leg['Strike'], 
                leg['Expiry'], leg['Vol'], st.session_state.spot_price, st.session_state.forward_price
            )
            val_t0 += (price_t0 - leg['Entry']) * leg['Qty'] * contract_multiplier
            price_tf = max(0, p - leg['Strike']) if leg['Type'] == 'Call' else max(0, leg['Strike'] - p)
            val_tF += (price_tf - leg['Entry']) * leg['Qty'] * contract_multiplier
        pnl_today.append(val_t0)
        pnl_expiry.append(val_tF)
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=chart_prices, y=pnl_today, name="Today", 
        line=dict(color='#0050FF', width=3),
        hovertemplate="Price: $%{x:.2f}<br>P&L: $%{y:.2f}"
    ))
    
    fig.add_trace(go.Scatter(
        x=chart_prices, y=pnl_expiry, name="Expiry", 
        line=dict(color='#1DBFD2', dash='dash', width=3),
        hovertemplate="Price: $%{x:.2f}<br>P&L: $%{y:.2f}"
    ))
    
    fig.add_vline(x=spot, line_dash="dot", line_color="grey")
    
    fig.update_layout(
        height=450, 
        template="plotly_white", 
        margin=dict(t=30, b=30),
        xaxis=dict(title="Stock Price @ Expiry", tickprefix="$"),
        yaxis=dict(title="Profit / Loss ($)", tickprefix="$")
    )
    st.plotly_chart(fig, use_container_width=True)
