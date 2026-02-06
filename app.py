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
if 'data_source' not in st.session_state: st.session_state.data_source = "None"

# --- 3. DATA ENGINE ---

@st.cache_data(ttl=600)
def load_sheet(raw_url):
    try:
        # Convert to export link
        if "/edit" in raw_url:
            base = raw_url.split("/edit")[0]
            csv_url = f"{base}/export?format=csv&gid=0"
        else:
            csv_url = raw_url

        # READ SPECIFIC COLUMNS: Code(C), Ticker(D), Type(E), Expiry(F), Strike(G)
        df = pd.read_csv(csv_url, usecols=[2, 3, 4, 5, 6], on_bad_lines='skip', dtype=str)
        df.columns = ['Code', 'Ticker', 'Type', 'Expiry', 'Strike']
        
        # Clean
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
        t = max(0.001, time_days/365.0)
        # mibian uses days if < 1? No, it uses annual fractions usually. 
        # Actually mibian BS takes [Underlying, Strike, InterestRate, Days]
        # and returns volatility OR takes volatility and returns price.
        # Let's use standard mibian behavior: BS([Spot, Strike, Rate, Days], Volatility)
        c = mibian.BS([spot, strike, rate_pct, time_days], volatility=vol_pct)
        return c.callPrice if kind == 'Call' else c.putPrice
    except: return 0.0

def get_greeks(kind, spot, strike, time_days, vol_pct, rate_pct=4.0):
    try:
        c = mibian.BS([spot, strike, rate_pct, time_days], volatility=vol_pct)
        g = c.call if kind == 'Call' else c.put
        return {'delta': g.delta, 'gamma': g.gamma, 'theta': g.theta, 'vega': g.vega}
    except: return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

# --- 4. DATA FETCHING ---
def fetch_data(t):
    clean = t.upper().replace(".AX", "").strip()
    sym = f"{clean}.AX"
    
    # 1. Get Price from Yahoo
    spot = 0.0
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="1d")
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
        else:
            return "ERROR", 0.0, None
            
        # 2. Try Yahoo Options
        if tk.options:
            return "YAHOO", spot, tk
        else:
            # 3. Fallback to Sheet
            # We don't return an object, just the signal to use sheet
            return "SHEET", spot, None
            
    except: return "ERROR", 0.0, None

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
        <span class="status-badge {status_cls}">{status_txt}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 6. SEARCH ---
c1, c2 = st.columns([3, 1])
query = c1.text_input("Ticker", st.session_state.ticker, label_visibility="collapsed")

if c2.button("Load Chain", type="primary", use_container_width=True) or (query.upper() != st.session_state.ticker):
    st.session_state.ticker = query.upper()
    with st.spinner("Analyzing Market..."):
        source, px, obj = fetch_data(st.session_state.ticker)
        st.session_state.spot_price = px
        st.session_state.chain_obj = obj
        st.session_state.data_source = source
        
        # Reload sheet to ensure freshness
        data, msg = load_sheet(RAW_SHEET_URL)
        st.session_state.ref_data = data
        st.session_state.sheet_msg = msg
        
        if source == "ERROR":
            st.error("Ticker not found.")
        st.rerun()

# --- 7. CHAIN DISPLAY ---

# Prepare Data for Table
df_view = pd.DataFrame()
current_exp = None
is_sheet_mode = False

# PATH A: YAHOO DATA (Live)
if st.session_state.data_source == "YAHOO" and st.session_state.chain_obj:
    try:
        tk = st.session_state.chain_obj
        exps = tk.options
        current_exp = st.selectbox("Expiry", exps)
        
        chain = tk.option_chain(current_exp)
        df = pd.merge(chain.calls, chain.puts, on="strike", how="outer", suffixes=("_c", "_p")).fillna(0)
        
        # Build View
        df_view['STRIKE'] = df['strike']
        df_view['C_Last'] = df['lastPrice_c']
        df_view['C_Vol'] = df['impliedVolatility_c']
        df_view['P_Last'] = df['lastPrice_p']
        df_view['P_Vol'] = df['impliedVolatility_p']
        
        # Code Lookup Needed
        is_sheet_mode = False
        
    except:
        st.session_state.data_source = "SHEET" # Fallback if yahoo crashes mid-stream

# PATH B: SHEET DATA (Fallback)
if st.session_state.data_source == "SHEET":
    st.warning("⚠️ Live options data unavailable. Showing **Theoretical Pricing** derived from your Sheet.")
    
    # Filter Sheet Data
    ref = st.session_state.ref_data
    tkr = st.session_state.ticker.replace(".AX", "")
    subset = ref[ref['Ticker'] == tkr]
    
    if not subset.empty:
        # Get Expiries
        valid_exps = sorted(subset['Expiry'].unique())
        # Convert to string for selectbox
        exp_map = {d.strftime("%Y-%m-%d"): d for d in valid_exps}
        current_exp = st.selectbox("Expiry", list(exp_map.keys()))
        
        # Filter for Expiry
        target_dt = exp_map[current_exp]
        day_chain = subset[subset['Expiry'] == target_dt]
        
        # Separate Calls/Puts
        calls = day_chain[day_chain['Type'] == 'Call'].set_index('Strike')['Code']
        puts = day_chain[day_chain['Type'] == 'Put'].set_index('Strike')['Code']
        
        # Merge
        all_strikes = sorted(list(set(calls.index) | set(puts.index)))
        df_view = pd.DataFrame({'STRIKE': all_strikes})
        
        # Map Codes
        df_view['C_Code'] = df_view['STRIKE'].map(calls)
        df_view['P_Code'] = df_view['STRIKE'].map(puts)
        
        # Calculate Theo Prices
        days = (target_dt - datetime.now()).days
        spot = st.session_state.spot_price
        
        # Vectorized or Loop calculation
        # Simple loop for safety
        c_prices, p_prices = [], []
        for s in df_view['STRIKE']:
            c_prices.append(get_bs_price('Call', spot, s, days, 20.0)) # Default 20% Vol
            p_prices.append(get_bs_price('Put', spot, s, days, 20.0))
            
        df_view['C_Last'] = c_prices
        df_view['P_Last'] = p_prices
        df_view['C_Vol'] = 0.20 # Dummy
        df_view['P_Vol'] = 0.20 # Dummy
        
        is_sheet_mode = True
    else:
        st.error(f"No options found in sheet for {tkr}")

# RENDER TABLE
if not df_view.empty and current_exp:
    # Filter Near Money
    center = st.session_state.spot_price
    if center > 0:
        df_view = df_view[(df_view['STRIKE'] > center*0.8) & (df_view['STRIKE'] < center*1.2)]
    
    st.markdown(f"**Chain: {current_exp}**")
    
    # Configure Columns
    cols = {
        "C_Last": st.column_config.NumberColumn("Call $", format="$%.2f"),
        "STRIKE": st.column_config.NumberColumn("Strike", format="%.2f"),
        "P_Last": st.column_config.NumberColumn("Put $", format="$%.2f"),
        "C_Vol": None, "P_Vol": None, "C_Code": None, "P_Code": None
    }
    
    selection = st.dataframe(
        df_view,
        column_config=cols,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    if selection.selection['rows']:
        idx = selection.selection['rows'][0]
        row = df_view.iloc[idx]
        d_obj = datetime.strptime(current_exp, "%Y-%m-%d")
        days = (d_obj - datetime.now()).days
        
        st.info(f"Selected: **${row['STRIKE']} Strike**")
        
        def add(side, kind, px, iv, code_hint=None):
            if iv < 0.01: iv = 0.20
            
            # Code Logic
            if is_sheet_mode:
                final_code = code_hint
            else:
                # Yahoo Mode -> Lookup
                # We need to re-implement lookup for Yahoo mode
                # But for now let's rely on Sheet mode being the primary Fix
                final_code = "LOOKUP_NEEDED"
            
            st.session_state.legs.append({
                "Qty": 1 if side=="Buy" else -1, "Type": kind, 
                "Strike": row['STRIKE'], "Expiry": days, "Vol": iv*100, 
                "Entry": px, "Code": final_code
            })
            
        # Get Codes from row if in sheet mode
        c_code = row.get('C_Code', 'ENTER_CODE')
        p_code = row.get('P_Code', 'ENTER_CODE')
        
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Buy Call"): add("Buy", "Call", row['C_Last'], row['C_Vol'], c_code)
        if b2.button("Sell Call"): add("Sell", "Call", row['C_Last'], row['C_Vol'], c_code)
        if b3.button("Buy Put"): add("Buy", "Put", row['P_Last'], row['P_Vol'], p_code)
        if b4.button("Sell Put"): add("Sell", "Put", row['P_Last'], row['P_Vol'], p_code)

# --- 8. MANUAL FALLBACK ---
if df_view.empty:
    with st.expander("🛠 Manual Leg Builder", expanded=True):
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

# --- 9. PORTFOLIO & CHARTS ---
if st.session_state.legs:
    st.markdown("---")
    st.subheader("📋 Active Portfolio")
    
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
