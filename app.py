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
    
    /* BUTTON STYLING */
    /* Target PRIMARY buttons (Load Chain) -> Dark Green */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #15803d !important; 
        color: white !important;
        border: none;
        font-weight: 600;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #166534 !important;
    }

    /* Target SECONDARY buttons (Clear, Zoom, etc) -> Neutral Tone */
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #e2e8f0 !important;
        border-color: #94a3b8 !important;
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
if 'vol_manual' not in st.session_state: st.session_state.vol_manual = 33.5
if 'matrix_vol_mod' not in st.session_state: st.session_state.matrix_vol_mod = 0.0

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

# --- MATH ENGINE ---
def get_bs_price(kind, spot, strike, time_days, vol_pct, rate_pct=4.0):
    try:
        if time_days <= 0:
            if kind == 'Call': return max(0.0, spot - strike)
            else: return max(0.0, strike - spot)
        
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
# Using columns to align Ticker Input, Vol Slider, and Load Button
c_search, c_vol, c_btn = st.columns([1, 2, 1])

with c_search:
    query = st.text_input("Ticker Symbol", st.session_state.ticker)

with c_vol:
    st.session_state.vol_manual = st.slider("Implied Volatility (IV) %", 10.0, 100.0, st.session_state.vol_manual, 0.5)

with c_btn:
    # Adding a little vertical spacer so the button aligns with the input boxes
    st.write("") 
    st.write("")
    if st.button("Load Chain", type="primary", use_container_width=True) or (query.upper() != st.session_state.ticker):
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
        exp_map = {d.strftime("%Y-%m-%d"): d for d in valid_exps}
        
        current_exp = st.selectbox("Expiry Date", list(exp_map.keys()))
        target_dt = exp_map[current_exp]
        
        day_chain = subset[subset['Expiry'] == target_dt]
        calls = day_chain[day_chain['Type'] == 'Call'].set_index('Strike')['Code']
        puts = day_chain[day_chain['Type'] == 'Put'].set_index('Strike')['Code']
        
        all_strikes = sorted(list(set(calls.index) | set(puts.index)))
        df_view = pd.DataFrame({'STRIKE': all_strikes})
        df_view['C_Code'] = df_view['STRIKE'].map(calls)
        df_view['P_Code'] = df_view['STRIKE'].map(puts)
        
        days = (target_dt - datetime.now()).days
        spot = st.session_state.spot_price
        vol = st.session_state.vol_manual
        
        c_px, c_delta, p_px, p_delta = [], [], [], []
        
        for s in df_view['STRIKE']:
            c_px.append(get_bs_price('Call', spot, s, days, vol))
            c_delta.append(get_greeks('Call', spot, s, days, vol)['delta'])
            p_px.append(get_bs_price('Put', spot, s, days, vol))
            p_delta.append(get_greeks('Put', spot, s, days, vol)['delta'])
            
        df_view['C_Price'] = c_px
        df_view['C_Delta'] = c_delta
        df_view['C_Vol'] = vol
        df_view['P_Price'] = p_px
        df_view['P_Delta'] = p_delta
        df_view['P_Vol'] = vol

# RENDER TABLE
if not df_view.empty and current_exp:
    center = st.session_state.spot_price
    if center > 0:
        df_view = df_view[(df_view['STRIKE'] > center*0.85) & (df_view['STRIKE'] < center*1.15)]
    
    st.markdown(f"**Chain: {current_exp}**")
    
    disp = df_view[['C_Code', 'C_Price', 'C_Vol', 'C_Delta', 'STRIKE', 'P_Price', 'P_Vol', 'P_Delta', 'P_Code']].copy()
    
    selection = st.dataframe(
        disp,
        column_config={
            "C_Code": st.column_config.TextColumn("Call Code"),
            "C_Price": st.column_config.NumberColumn("Theo Price", format="%.3f"),
            "C_Vol": st.column_config.NumberColumn("Vol %", format="%.1f"),
            "C_Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
            "STRIKE": st.column_config.NumberColumn("Strike", format="%.2f"),
            "P_Price": st.column_config.NumberColumn("Theo Price", format="%.3f"),
            "P_Vol": st.column_config.NumberColumn("Vol %", format="%.1f"),
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
        d_obj = datetime.strptime(current_exp, "%Y-%m-%d")
        days = (d_obj - datetime.now()).days
        
        st.info(f"Selected: **${row['STRIKE']} Strike**")
        
        def add(side, kind, px, code_hint):
            st.session_state.legs.append({
                "Qty": 1 if side=="Buy" else -1, "Type": kind, 
                "Strike": row['STRIKE'], "Expiry": days, 
                "Vol": st.session_state.vol_manual, 
                "Entry": px, "Code": code_hint,
                "Remove": False 
            })
            
        c_c = str(row['C_Code']) if pd.notna(row['C_Code']) else "N/A"
        p_c = str(row['P_Code']) if pd.notna(row['P_Code']) else "N/A"
        
        # Action Buttons (Using neutral secondary buttons except where emphasized)
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Buy Call"): add("Buy", "Call", row['C_Price'], c_c)
        if b2.button("Sell Call"): add("Sell", "Call", row['C_Price'], c_c)
        if b3.button("Buy Put"): add("Buy", "Put", row['P_Price'], p_c)
        if b4.button("Sell Put"): add("Sell", "Put", row['P_Price'], p_c)

# --- 7. PORTFOLIO ---
if st.session_state.legs:
    st.markdown("---")
    
    c_tick, c_port = st.columns([1, 2])
    
    with c_tick:
        ticket_text = f"TICKET: {st.session_state.ticker} (Spot ${st.session_state.spot_price:.2f})\n"
        for leg in st.session_state.legs:
            direction = "Buy" if leg['Qty'] > 0 else "Sell"
            qty = abs(leg['Qty'])
            ticket_text += f"{direction} {qty}x {leg['Code']} ({leg['Type']} ${leg['Strike']}) @ ${leg['Entry']:.3f}\n"
        
        st.caption("Trade Ticket (Copy)")
        st.code(ticket_text, language="text")
        
        if st.button("Clear Portfolio", type="secondary"):
            st.session_state.legs = []
            st.rerun()

    with c_port:
        st.caption("Active Legs (Edit Qty or Check 'Remove')")
        
        df_port = pd.DataFrame(st.session_state.legs)
        if 'Remove' not in df_port.columns: df_port['Remove'] = False

        column_config = {
            "Remove": st.column_config.CheckboxColumn("Remove?", default=False),
            "Code": st.column_config.TextColumn("ASX Code", width="medium"),
            "Entry": st.column_config.NumberColumn("Entry", format="$%.3f"),
            "Vol": st.column_config.NumberColumn("Vol %", format="%.1f%%"),
            "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
        }

        edited_df = st.data_editor(
            df_port,
            column_config=column_config,
            use_container_width=True,
            num_rows="dynamic",
            key="portfolio_editor"
        )

        if edited_df['Remove'].any():
            remaining_legs = edited_df[~edited_df['Remove']].drop(columns=['Remove']).to_dict('records')
            st.session_state.legs = remaining_legs
            st.rerun()
        else:
            st.session_state.legs = edited_df.to_dict('records')

    # --- C. PAYOFF MATRIX ---
    st.markdown("### 📊 Payoff Matrix")
    m_c1, m_c2, m_c3 = st.columns(3)
    time_step = m_c1.slider("Time Increment (Days)", 1, 30, 7)
    range_pct = m_c2.select_slider("Price Range", options=[0.02, 0.05, 0.10, 0.20], value=0.05, format_func=lambda x: f"{x*100:.0f}%")
    
    with m_c3:
        st.write("Volatility Modifier")
        c_v1, c_v2, c_v3 = st.columns(3)
        if c_v1.button("-10%"): st.session_state.matrix_vol_mod -= 10
        if c_v2.button("Reset"): st.session_state.matrix_vol_mod = 0
        if c_v3.button("+10%"): st.session_state.matrix_vol_mod += 10
        st.caption(f"Current Mod: {st.session_state.matrix_vol_mod:+}%")

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
                exit_px = get_bs_price(leg['Type'], p, leg['Strike'], rem_days, sim_vol)
                leg_pnl = (exit_px - leg['Entry']) * leg['Qty'] * 100
                pnl += leg_pnl
            
            col_name = (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
            if d == 0: col_name = f"Today ({col_name})"
            row[col_name] = pnl
        matrix_data.append(row)
        
    df_mx = pd.DataFrame(matrix_data).set_index("Price")
    
    def color_pnl(val):
        color = '#dcfce7' if val > 0 else '#fee2e2' if val < 0 else 'white'
        return f'background-color: {color}; color: black'

    st.dataframe(
        df_mx.style.applymap(color_pnl).format("${:,.0f}"),
        use_container_width=True,
        height=450
    )

    # --- D. PAYOFF DIAGRAM ---
    st.markdown("### 📈 Payoff Diagram")
    
    chart_prices = np.linspace(spot * (1 - range_pct*1.5), spot * (1 + range_pct*1.5), 100)
    pnl_today = []
    pnl_expiry = []
    
    for p in chart_prices:
        val_t0 = 0
        val_tF = 0
        for leg in st.session_state.legs:
            # T+0
            sim_vol = leg['Vol'] 
            rem_days = leg['Expiry']
            price_t0 = get_bs_price(leg['Type'], p, leg['Strike'], rem_days, sim_vol)
            val_t0 += (price_t0 - leg['Entry']) * leg['Qty'] * 100
            
            # Expiry
            if leg['Type'] == 'Call': price_tf = max(0, p - leg['Strike'])
            else: price_tf = max(0, leg['Strike'] - p)
            val_tF += (price_tf - leg['Entry']) * leg['Qty'] * 100
            
        pnl_today.append(val_t0)
        pnl_expiry.append(val_tF)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_prices, y=pnl_today, name="Today (T+0)", line=dict(color='#0e1b32', width=3)))
    fig.add_trace(go.Scatter(x=chart_prices, y=pnl_expiry, name="At Expiry", line=dict(color='#15803d', dash='dash')))
    fig.add_vline(x=spot, line_dash="dot", line_color="grey", annotation_text="Spot")
    
    fig.update_layout(
        height=400, 
        template="plotly_white", 
        margin=dict(t=20, b=20, l=40, r=20),
        xaxis_title="Stock Price",
        yaxis_title="Profit / Loss ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
