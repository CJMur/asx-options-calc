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
# 🛑 YOUR GOOGLE SHEET LINK
# ==========================================
SHEET_URL = "https://docs.google.com/spreadsheets/d/1d9FQ5mn--MSNJ_WJkU--IvoSRU0gQBqE0f9s9zEb0Q4/export?format=csv"

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .main-header {
        background-color: #0e1b32; padding: 1.5rem 2rem; color: white;
        border-radius: 8px; display: flex; justify_content: space-between; align_items: center;
        margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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

# --- 3. ROBUST DATA ENGINE ---

@st.cache_data(ttl=600)
def load_reference_data(url):
    try:
        # Load columns A-E only
        df = pd.read_csv(url, usecols=[0, 1, 2, 3, 4], on_bad_lines='skip')
        
        # Standardize Header Names
        df.columns = ['Ticker', 'Expiry', 'Strike', 'Type', 'Code']
        
        # Clean Data Types for Matching
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
        df['Type'] = df['Type'].astype(str).str.title().str.strip() 
        df['Strike'] = df['Strike'].astype(str).str.replace('$', '').astype(float)
        
        # Flexible Date Parsing
        df['Expiry'] = pd.to_datetime(df['Expiry'], dayfirst=True, errors='coerce') 
        
        return df
    except Exception as e:
        return pd.DataFrame()

# Load Data on Startup
st.session_state.ref_data = load_reference_data(SHEET_URL)

def lookup_code(ticker, expiry_str, strike, kind):
    df = st.session_state.ref_data
    if df is None or df.empty: return "ENTER_CODE"

    try:
        # Normalize the Yahoo Date to match Sheet Date
        target_date = pd.to_datetime(expiry_str)
        
        # Filter Logic
        mask = (
            (df['Ticker'] == ticker.replace(".AX","")) &
            (df['Type'] == kind) &
            (np.isclose(df['Strike'], strike, atol=0.01)) & 
            (df['Expiry'] == target_date)
        )
        
        results = df[mask]
        if not results.empty:
            return results.iloc[0]['Code']
        return "ENTER_CODE"
    except:
        return "ERROR"

# --- 4. YAHOO & MATH ENGINE ---
def fetch_yahoo(ticker_input):
    clean_tk = ticker_input.upper().replace(".AX", "").strip()
    sym = f"{clean_tk}.AX"
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="1d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
            return "OK", price, tk if tk.options else None
    except: return "ERROR", 0.0, None
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

# --- 5. UI LAYOUT ---
st.markdown(f"""
<div class="main-header">
    <div style="font-size: 22px; font-weight: 700;">TradersCircle <span style="font-weight: 300; opacity: 0.7;">| PRO</span></div>
    <div style="text-align: right;">
        <div style="font-size: 12px; opacity: 0.8;">{st.session_state.ticker}</div>
        <div style="font-size: 24px; font-weight: 700; color: #4ade80;">${st.session_state.spot_price:.2f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Search
c_search, c_btn = st.columns([3, 1])
with c_search:
    query = st.text_input("Ticker Search", value=st.session_state.ticker, label_visibility="collapsed")
if c_btn.button("Load Chain", type="primary", use_container_width=True) or (query.upper() != st.session_state.ticker):
    st.session_state.ticker = query.upper()
    with st.spinner("Fetching Market Data..."):
        status, px, obj = fetch_yahoo(st.session_state.ticker)
        if status == "OK":
            st.session_state.spot_price = px
            st.session_state.chain_obj = obj
            st.rerun()
        else:
            st.error("Ticker not found.")

# Chain Display
if st.session_state.chain_obj:
    tk = st.session_state.chain_obj
    sel_exp = st.selectbox("Select Expiry", tk.options)
    
    try:
        chain = tk.option_chain(sel_exp)
        df = pd.merge(chain.calls, chain.puts, on="strike", how="outer", suffixes=("_c", "_p")).fillna(0)
        center = st.session_state.spot_price
        df = df[(df['strike'] > center*0.8) & (df['strike'] < center*1.2)]
        
        st.markdown(f"**Chain: {sel_exp}**")
        df_view = pd.DataFrame({
            'STRIKE': df['strike'],
            'C_Last': df['lastPrice_c'], 'C_Vol': df['impliedVolatility_c'],
            'P_Last': df['lastPrice_p'], 'P_Vol': df['impliedVolatility_p']
        })
        
        selection = st.dataframe(
            df_view,
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
            row = df_view.iloc[idx]
            d_obj = datetime.strptime(sel_exp, "%Y-%m-%d")
            days = (d_obj - datetime.now()).days
            
            st.info(f"Selected: **${row['STRIKE']} Strike**")
            
            def add_leg(side, kind, px, iv):
                if iv < 0.01: iv = 0.20
                
                # --- AUTO LOOKUP ---
                code = lookup_code(st.session_state.ticker, sel_exp, row['STRIKE'], kind)
                
                st.session_state.legs.append({
                    "Qty": 1 * (1 if side == "Buy" else -1),
                    "Type": kind, "Strike": row['STRIKE'], "Expiry": days,
                    "Vol": iv * 100, "Entry": px, "Code": code
                })

            c1, c2, c3, c4 = st.columns(4)
            if c1.button("Buy Call"): add_leg("Buy", "Call", row['C_Last'], row['C_Vol'])
            if c2.button("Sell Call"): add_leg("Sell", "Call", row['C_Last'], row['C_Vol'])
            if c3.button("Buy Put"): add_leg("Buy", "Put", row['P_Last'], row['P_Vol'])
            if c4.button("Sell Put"): add_leg("Sell", "Put", row['P_Last'], row['P_Vol'])
            
    except Exception as e: st.error(f"Chain Error: {e}")

# Portfolio
if st.session_state.legs:
    st.markdown("---")
    st.subheader("📋 Active Portfolio")
    
    df_port = pd.DataFrame(st.session_state.legs)
    edited_df = st.data_editor(
        df_port,
        column_config={
            "Code": st.column_config.TextColumn("ASX Code", width="medium", required=True),
            "Qty": st.column_config.NumberColumn("Qty", required=True),
            "Entry": st.column_config.NumberColumn("Entry", format="$%.3f"),
            "Vol": st.column_config.NumberColumn("Vol %", format="%.1f%%"),
            "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
        },
        use_container_width=True, num_rows="dynamic", key="editor"
    )
    st.session_state.legs = edited_df.to_dict('records')
    
    tot_delta = 0
    for leg in st.session_state.legs:
        g = get_greeks(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
        tot_delta += g['delta'] * leg['Qty'] * 100
    st.metric("Net Delta", f"{tot_delta:.1f}")

    st.markdown("---")
    c_ctrl, c_view = st.columns([1, 3])
    with c_ctrl:
        st.caption("Controls")
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
                px = get_bs_price(leg['Type'], p, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
                v0 += (px - leg['Entry']) * leg['Qty'] * 100
                tf = max(0, leg['Expiry'] - ts)
                pxf = get_bs_price(leg['Type'], p, leg['Strike'], tf, leg['Vol'], 4.0)
                vF += (pxf - leg['Entry']) * leg['Qty'] * 100
            p0.append(v0)
            pF.append(vF)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=p0, name="Today", line=dict(color='#2980b9', width=3)))
        if ts > 0: fig.add_trace(go.Scatter(x=prices, y=pF, name=f"T+{ts}d", line=dict(color='#e67e22', dash='dash')))
        fig.add_vline(x=center, line_dash="dot", annotation_text="Spot")
        fig.update_layout(height=350, template="plotly_white", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
