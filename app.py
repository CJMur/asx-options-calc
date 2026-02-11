# ==========================================
# TradersCircle Options Calculator
# VERSION: 7.6 (Slider Color Fix)
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
st.set_page_config(layout="wide", page_title="TradersCircle Options v7.6")
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
    
    /* --- SLIDER COLOR FIX (Aggressive Override) --- */
    /* 1. The Thumb (Circle) */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #0050FF !important;
        border-color: #0050FF !important;
        box-shadow: none !important;
    }
    
    /* 2. The Filled Track (The line to the left of the thumb) */
    /* This targets the first child div inside the slider track container, which is usually the filled part */
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
        background-color: #0050FF !important;
    }
    
    /* 3. The Value Text (The numbers '7' or '5%' above) */
    div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
        color: #0050FF !important;
    }
    
    .stDataFrame { border: none !important; }

    /* Clean Table Headers */
    .trade-header {
        font-weight: 700; color: #94a3b8; font-size: 12px; text-transform: uppercase;
        margin-bottom: 5px;
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
if 'vol_manual' not in st.session_state: st.session_state.vol_manual = 30.0

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
    r = 0.04 # Fixed 4% Risk Free Rate
    
    try:
        T = max(0.001, time_days / 365.0)
        v = vol_pct / 100.0
        S = float(spot)
        K = float(strike)
        
        # Discrete Dividend Logic
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
                <div class="header-title">TradersCircle <span style="font-weight: 300;">PRO</span></div>
                <div class="header-sub">Option Strategy Builder v7.6</div>
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
    query = st.text_input("Ticker", value=st.session_state.ticker, placeholder="Enter Stock Code")

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
    if st.button("LOAD OPTIONS", type="primary", use_container_width=True) or (query and query.upper() != st.session_state.ticker):
        if not query:
            st.warning("Please enter a ticker symbol.")
        else:
            if query.upper() != st.session_state.ticker: st.session_state.manual_spot = False
            st.session_state.ticker = query.upper()
            with st.spinner("Fetching Market Data..."):
                source, px, div_data = fetch_data(st.session_state.ticker)
                if not st.session_state.manual_spot: st.session_state.spot_price = px
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
        
        current_exp = st.selectbox("Expiry", list(exp_map.keys()), index=None, placeholder="Select Expiry")
        
        if current_exp:
            target_dt = exp_map[current_exp]
            days_diff = (target_dt - today).days
            day_chain = subset[subset['Expiry'] == target_dt].copy()
            
            def calc_row_metrics(row):
                vol = float(row['Vol']) if pd.notna(row['Vol']) else 30.0
                style = row.get('Style', 'American')
                
                if st.session_state.is_market_open:
                    px, delta = calculate_price_and_delta(style, row['Type'], st.session_state.spot_price, row['Strike'], days_diff, vol)
                else:
                    px = float(row['Settlement']) if pd.notna(row['Settlement']) else 0.0
                    _, delta = calculate_price_and_delta(style, row['Type'], st.session_state.spot_price, row['Strike'], days_diff, vol)
                return pd.Series([px, delta, vol])

            metrics = day_chain.apply(calc_row_metrics, axis=1)
            metrics.columns = ['Calc_Price', 'Calc_Delta', 'Calc_Vol']
            day_chain = pd.concat([day_chain, metrics], axis=1)
            
            calls = day_chain[day_chain['Type'] == 'Call'].set_index('Strike')
            puts = day_chain[day_chain['Type'] == 'Put'].set_index('Strike')
            
            all_strikes = sorted(list(set(calls.index) | set(puts.index)))
            df_view = pd.DataFrame({'STRIKE': all_strikes})
            
            df_view['C_Code'] = df_view['STRIKE'].map(calls['Code'])
            df_view['C_Price'] = df_view['STRIKE'].map(calls['Calc_Price'])
            df_view['C_Vol'] = df_view['STRIKE'].map(calls['Calc_Vol'])
            df_view['C_Delta'] = df_view['STRIKE'].map(calls['Calc_Delta'])
            
            df_view['P_Code'] = df_view['STRIKE'].map(puts['Code'])
            df_view['P_Price'] = df_view['STRIKE'].map(puts['Calc_Price'])
            df_view['P_Vol'] = df_view['STRIKE'].map(puts['Calc_Vol'])
            df_view['P_Delta'] = df_view['STRIKE'].map(puts['Calc_Delta'])

if not df_view.empty and current_exp:
    center = st.session_state.spot_price
    if center > 0:
        df_view['Diff'] = abs(df_view['STRIKE'] - center)
        atm_idx = df_view['Diff'].idxmin()
        start_idx = max(0, atm_idx - 12)
        end_idx = min(len(df_view), atm_idx + 13)
        df_view = df_view.iloc[start_idx:end_idx].drop(columns=['Diff'])
    
    st.markdown(f"**Chain: {current_exp}**")
    
    disp = df_view[['C_Code', 'C_Price', 'C_Vol', 'C_Delta', 'STRIKE', 'P_Price', 'P_Vol', 'P_Delta', 'P_Code']].copy()
    
    styled_disp = disp.style.applymap(
        lambda x: "font-weight: bold; background-color: rgba(255,255,255,0.05);", 
        subset=['STRIKE']
    ).format({
        'C_Price': '{:.3f}', 'C_Vol': '{:.1f}', 'C_Delta': '{:.3f}',
        'STRIKE': '{:.3f}',
        'P_Price': '{:.3f}', 'P_Vol': '{:.1f}', 'P_Delta': '{:.3f}'
    })

    selection = st.dataframe(
        styled_disp,
        column_config={
            "C_Code": st.column_config.TextColumn("Call Code"),
            "C_Price": st.column_config.NumberColumn("Price", format="%.3f"),
            "C_Vol": st.column_config.NumberColumn("IV %", format="%.1f"),
            "C_Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
            "STRIKE": st.column_config.NumberColumn("Strike", format="%.3f"),
            "P_Price": st.column_config.NumberColumn("Price", format="%.3f"),
            "P_Vol": st.column_config.NumberColumn("IV %", format="%.1f"),
            "P_Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
            "P_Code": st.column_config.TextColumn("Put Code"),
        },
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    if selection.selection['rows']:
        idx = selection.selection['rows'][0]
        row = disp.iloc[idx]
        days_diff = (datetime.strptime(current_exp, "%Y-%m-%d") - datetime.now()).days
        
        q_col, _ = st.columns([1, 5])
        trade_qty = q_col.number_input("Trade Quantity", min_value=1, value=1, step=1)
        
        def add(side, kind, px, code_hint, delta_val, qty_val):
            final_qty = qty_val if side == "Buy" else -qty_val
            row_vol = row['C_Vol'] if kind == 'Call' else row['P_Vol']
            st.session_state.legs.append({
                "Qty": final_qty, "Type": kind, "Strike": row['STRIKE'], 
                "Expiry": days_diff, "Vol": row_vol, "Entry": px, 
                "Code": code_hint, "Delta": delta_val
            })
            st.rerun()
            
        c_c = str(row['C_Code']) if pd.notna(row['C_Code']) else "N/A"
        p_c = str(row['P_Code']) if pd.notna(row['P_Code']) else "N/A"
        
        b1, b2, b3, b4, _ = st.columns([1, 1, 1, 1, 6]) 
        if b1.button(f"Buy Call"): add("Buy", "Call", row['C_Price'], c_c, row['C_Delta'], trade_qty)
        if b2.button(f"Sell Call"): add("Sell", "Call", row['C_Price'], c_c, row['C_Delta'], trade_qty)
        if b3.button(f"Buy Put"): add("Buy", "Put", row['P_Price'], p_c, row['P_Delta'], trade_qty)
        if b4.button(f"Sell Put"): add("Sell", "Put", row['P_Price'], p_c, row['P_Delta'], trade_qty)

# --- 10. STRATEGY (ALIGNED FOOTER) ---
if st.session_state.legs:
    st.markdown("---")
    st.subheader("Strategy")
    
    # Column Headers
    h_col_spec = [1, 2, 1, 1, 1, 1, 1, 1, 0.5]
    h1, h2, h3, h4, h5, h6, h7, h8, h9 = st.columns(h_col_spec)
    with h1: st.markdown('<div class="trade-header">Qty</div>', unsafe_allow_html=True)
    with h2: st.markdown('<div class="trade-header">Code</div>', unsafe_allow_html=True)
    with h3: st.markdown('<div class="trade-header">Type</div>', unsafe_allow_html=True)
    with h4: st.markdown('<div class="trade-header">Strike</div>', unsafe_allow_html=True)
    with h5: st.markdown('<div class="trade-header">Entry</div>', unsafe_allow_html=True)
    with h6: st.markdown('<div class="trade-header">Theo</div>', unsafe_allow_html=True)
    with h7: st.markdown('<div class="trade-header">Delta</div>', unsafe_allow_html=True)
    with h8: st.markdown('<div class="trade-header">Premium</div>', unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 0 0 10px 0; border-top: 1px solid #334155;'>", unsafe_allow_html=True)

    total_delta = 0
    total_premium = 0
    total_theo = 0
    
    # Rows
    for i, leg in enumerate(st.session_state.legs):
        new_theo, new_delta = calculate_price_and_delta('American', leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'])
        net_delta = leg['Qty'] * new_delta * 100
        premium = -(leg['Qty'] * leg['Entry'] * 100)
        theo_val = leg['Qty'] * new_theo * 100
        
        total_delta += net_delta
        total_premium += premium
        total_theo += theo_val
        
        type_color = "#4ade80" if leg['Type'] == 'Call' else "#f87171"
        type_text = f"<span style='color:{type_color}; font-weight:600'>{leg['Type']}</span>"
        
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(h_col_spec)
        with c1: st.write(f"**{leg['Qty']}**")
        with c2: st.write(f"{leg['Code']}")
        with c3: st.markdown(type_text, unsafe_allow_html=True)
        with c4: st.write(f"{leg['Strike']:.3f}")
        with c5: st.write(f"${leg['Entry']:.3f}")
        with c6: st.write(f"${new_theo:.3f}")
        with c7: st.write(f"{net_delta:.2f}")
        with c8: st.write(f"${premium:.2f}")
        with c9:
            if st.button("✕", key=f"d_{i}"):
                st.session_state.legs.pop(i)
                st.rerun()
        st.markdown("<hr style='margin: 5px 0; border-top: 1px solid #1e293b;'>", unsafe_allow_html=True)

    # --- ALIGNED FOOTER ROW ---
    with st.container():
        # Clean spacing, matching columns
        f1, f2, f3, f4, f5, f6, f7, f8, f9 = st.columns(h_col_spec)
        
        with f2: st.markdown("**TOTAL STRATEGY**")
        with f6: st.markdown(f"**${total_theo:,.2f}**")
        with f7: st.markdown(f"**{total_delta:,.2f}**")
        with f8: 
            p_color = '#4ade80' if total_premium > 0 else '#f87171'
            st.markdown(f"<span style='color:{p_color}; font-weight:bold'>${total_premium:,.2f}</span>", unsafe_allow_html=True)

    # --- MATRIX ---
    st.markdown("---")
    st.subheader("Payoff Matrix")
    m1, m2 = st.columns(2)
    time_step = m1.slider("Step (Days)", 1, 30, 7)
    range_pct = m2.select_slider("Range", options=[0.02, 0.05, 0.10, 0.20], value=0.05, format_func=lambda x: f"{x*100:.0f}%")
    
    with m1:
        c_v1, c_v2, c_v3 = st.columns(3)
        if c_v1.button("-10%"): st.session_state.matrix_vol_mod -= 10
        if c_v2.button("Flat"): st.session_state.matrix_vol_mod = 0
        if c_v3.button("+10%"): st.session_state.matrix_vol_mod += 10
        st.caption(f"Vol Mod: {st.session_state.matrix_vol_mod:+}%")

    spot = st.session_state.spot_price
    prices = np.linspace(spot * (1 - range_pct), spot * (1 + range_pct), 12)
    dates = [0, time_step, time_step*2, time_step*3, time_step*4]
    
    matrix_data = []
    for p in prices:
        row = {"Price": p}
        for d in dates:
            pnl = 0
            for leg in st.session_state.legs:
                sim_vol = max(1.0, leg['Vol'] + st.session_state.matrix_vol_mod)
                rem_days = max(0, leg['Expiry'] - d)
                exit_px, _ = calculate_price_and_delta('American', leg['Type'], p, leg['Strike'], rem_days, sim_vol)
                pnl += (exit_px - leg['Entry']) * leg['Qty'] * 100
            col_name = (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
            if d == 0: col_name = f"Today ({col_name})"
            row[col_name] = pnl
        matrix_data.append(row)
        
    df_mx = pd.DataFrame(matrix_data).set_index("Price")
    st.dataframe(df_mx.style.background_gradient(cmap='RdYlGn', axis=None).format("${:,.0f}"), use_container_width=True, height=600)

    # CHART
    st.markdown("### Payoff Chart")
    chart_prices = np.linspace(spot * (1 - range_pct*1.5), spot * (1 + range_pct*1.5), 100)
    pnl_today = []
    pnl_expiry = []
    for p in chart_prices:
        val_t0 = 0
        val_tF = 0
        for leg in st.session_state.legs:
            price_t0, _ = calculate_price_and_delta('American', leg['Type'], p, leg['Strike'], leg['Expiry'], leg['Vol'])
            val_t0 += (price_t0 - leg['Entry']) * leg['Qty'] * 100
            price_tf = max(0, p - leg['Strike']) if leg['Type'] == 'Call' else max(0, leg['Strike'] - p)
            val_tF += (price_tf - leg['Entry']) * leg['Qty'] * 100
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
