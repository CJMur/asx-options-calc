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
    
    /* HEADER FIX: Use Grid to prevent overlapping */
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
    .header-info { text-align: right; }
    
    /* STATUS BADGE */
    .status-badge {
        font-size: 11px; background-color: #f1f5f9; color: #475569; 
        padding: 2px 6px; border-radius: 4px; border: 1px solid #cbd5e1;
        display: inline-block; margin-top: 4px;
    }
    .status-success { color: #16a34a; background-color: #f0fdf4; border-color: #bbf7d0; }
    
    /* DARK GREEN BUTTON FOR 'LOAD CHAIN' */
    div[data-testid="stButton"] > button {
        background-color: #15803d !important; /* Dark Green */
        color: white !important;
        border: none;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #166534 !important; /* Darker Green on Hover */
        color: white !important;
    }
    
    /* Remove standard table borders for cleaner look */
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
if 'vol_manual' not in st.session_state: st.session_state.vol_manual = 20.0

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

def get_bs_price(kind, spot, strike, time_days, vol_pct, rate_pct=4.0):
    try:
        t = max(0.001, time_days/365.0) # Annualized
        c = mibian.BS([spot, strike, rate_pct, time_days], volatility=vol_pct)
        return c.callPrice if kind == 'Call' else c.putPrice
    except: return 0.0

def get_greeks(kind, spot, strike, time_days, vol_pct, rate_pct=4.0):
    try:
        c = mibian.BS([spot, strike, rate_pct, time_days], volatility=vol_pct)
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
    # IV SLIDER
    st.session_state.vol_manual = st.slider("Implied Volatility %", 10.0, 100.0, st.session_state.vol_manual, 5.0, label_visibility="collapsed")

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

# --- 6. CHAIN DISPLAY ---
df_view = pd.DataFrame()
is_sheet_mode = False
current_exp = None

if st.session_state.data_source == "SHEET":
    # st.info(f"Using Sheet Mode (IV: {st.session_state.vol_manual}%)")
    ref = st.session_state.ref_data
    tkr = st.session_state.ticker.replace(".AX", "")
    subset = ref[ref['Ticker'] == tkr]
    
    if not subset.empty:
        valid_exps = sorted(subset['Expiry'].unique())
        exp_map = {d.strftime("%Y-%m-%d"): d for d in valid_exps}
        current_exp = st.selectbox("Expiry", list(exp_map.keys()))
        target_dt = exp_map[current_exp]
        
        day_chain = subset[subset['Expiry'] == target_dt]
        calls = day_chain[day_chain['Type'] == 'Call'].set_index('Strike')['Code']
        puts = day_chain[day_chain['Type'] == 'Put'].set_index('Strike')['Code']
        
        all_strikes = sorted(list(set(calls.index) | set(puts.index)))
        df_view = pd.DataFrame({'STRIKE': all_strikes})
        df_view['C_Code'] = df_view['STRIKE'].map(calls)
        df_view['P_Code'] = df_view['STRIKE'].map(puts)
        
        # PRICING
        days = (target_dt - datetime.now()).days
        spot = st.session_state.spot_price
        vol = st.session_state.vol_manual
        
        c_prices = [get_bs_price('Call', spot, s, days, vol) for s in df_view['STRIKE']]
        p_prices = [get_bs_price('Put', spot, s, days, vol) for s in df_view['STRIKE']]
            
        df_view['C_Last'] = c_prices
        df_view['P_Last'] = p_prices
        is_sheet_mode = True

# RENDER TABLE
if not df_view.empty and current_exp:
    center = st.session_state.spot_price
    if center > 0:
        df_view = df_view[(df_view['STRIKE'] > center*0.8) & (df_view['STRIKE'] < center*1.2)]
    
    st.markdown(f"**Chain: {current_exp}**")
    cols = {
        "C_Last": st.column_config.NumberColumn("Call", format="$%.2f"),
        "STRIKE": st.column_config.NumberColumn("Strike", format="%.2f"),
        "P_Last": st.column_config.NumberColumn("Put", format="$%.2f"),
        "C_Code": None, "P_Code": None
    }
    
    selection = st.dataframe(
        df_view, column_config=cols, hide_index=True, use_container_width=True,
        on_select="rerun", selection_mode="single-row"
    )
    
    if selection.selection['rows']:
        idx = selection.selection['rows'][0]
        row = df_view.iloc[idx]
        d_obj = datetime.strptime(current_exp, "%Y-%m-%d")
        days = (d_obj - datetime.now()).days
        
        st.info(f"Selected: **${row['STRIKE']} Strike**")
        def add(side, kind, px, code_hint):
            st.session_state.legs.append({
                "Qty": 1 if side=="Buy" else -1, "Type": kind, 
                "Strike": row['STRIKE'], "Expiry": days, 
                "Vol": st.session_state.vol_manual, 
                "Entry": px, "Code": code_hint
            })
            
        c_code = row.get('C_Code', 'ENTER_CODE')
        p_code = row.get('P_Code', 'ENTER_CODE')
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Buy Call"): add("Buy", "Call", row['C_Last'], c_code)
        if b2.button("Sell Call"): add("Sell", "Call", row['C_Last'], c_code)
        if b3.button("Buy Put"): add("Buy", "Put", row['P_Last'], p_code)
        if b4.button("Sell Put"): add("Sell", "Put", row['P_Last'], p_code)

# --- 7. PORTFOLIO & TICKETS ---
if st.session_state.legs:
    st.markdown("---")
    
    # A. TRADE TICKET (COPY PASTE)
    ticket_text = f"TICKET: {st.session_state.ticker} (Spot ${st.session_state.spot_price:.2f})\n"
    ticket_text += "-"*40 + "\n"
    for leg in st.session_state.legs:
        direction = "Buy" if leg['Qty'] > 0 else "Sell"
        qty = abs(leg['Qty'])
        ticket_text += f"{direction} {qty}x {leg['Code']} ({leg['Type']} ${leg['Strike']}) @ ${leg['Entry']:.2f}\n"
    
    c_tick, c_port = st.columns([1, 2])
    with c_tick:
        st.caption("Trade Ticket (Copy)")
        st.code(ticket_text, language="text")
        if st.button("Clear Portfolio"):
            st.session_state.legs = []
            st.rerun()

    # B. COLORED PORTFOLIO DISPLAY
    with c_port:
        st.caption("Active Legs")
        df_port = pd.DataFrame(st.session_state.legs)
        
        # Color Logic Function
        def highlight_type(val):
            color = '#dcfce7' if val == 'Call' else '#fee2e2' # Green vs Red light
            return f'background-color: {color}; color: black; font-weight: bold;'

        # Clean display DF
        disp_port = df_port[['Qty', 'Code', 'Type', 'Strike', 'Entry']].copy()
        
        # Apply Style
        styled_port = disp_port.style.applymap(highlight_type, subset=['Type'])
        styled_port = styled_port.format({"Entry": "${:.2f}", "Strike": "{:.2f}"})
        
        st.dataframe(styled_port, use_container_width=True, hide_index=True)

    # C. EDIT EXPANDER (Hidden by default to keep UI clean)
    with st.expander("Edit Quantities"):
        edited = st.data_editor(df_port, key="editor", num_rows="dynamic")
        st.session_state.legs = edited.to_dict('records')

    # D. CHARTS
    st.markdown("---")
    c_ctrl, c_view = st.columns([1, 3])
    with c_ctrl:
        if st.button("Zoom 5%"): st.session_state.range_pct = 0.05
        if st.button("Zoom 15%"): st.session_state.range_pct = 0.15
        ts = st.slider("Time Step", 0, 60, 15)
    
    with c_view:
        center = st.session_state.spot_price
        pct = st.session_state.range_pct
        prices = np.linspace(center*(1-pct), center*(1+pct), 80)
        p0, pF = [], []
        for p in prices:
            v0, vF = 0, 0
            for leg in st.session_state.legs:
                px = get_bs_price(leg['Type'], p, leg['Strike'], leg['Expiry'], leg['Vol'])
                v0 += (px - leg['Entry']) * leg['Qty'] * 100
                tf = max(0, leg['Expiry'] - ts)
                pxf = get_bs_price(leg['Type'], p, leg['Strike'], tf, leg['Vol'])
                vF += (pxf - leg['Entry']) * leg['Qty'] * 100
            p0.append(v0)
            pF.append(vF)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=p0, name="Today", line=dict(color='#2980b9', width=3)))
        if ts > 0: fig.add_trace(go.Scatter(x=prices, y=pF, name=f"T+{ts}d", line=dict(color='#e67e22', dash='dash')))
        fig.add_vline(x=center, line_dash="dot", annotation_text="Spot")
        fig.update_layout(height=350, template="plotly_white", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
