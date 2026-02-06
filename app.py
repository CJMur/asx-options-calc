import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import requests
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="TradersCircle Options")

# Custom CSS for Corporate Branding
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    
    /* Corporate Navy Header */
    .main-header {
        background-color: #0e1b32;
        padding: 1.5rem 2rem;
        color: white;
        border-radius: 8px;
        display: flex; justify_content: space-between; align_items: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Compact Table Styling */
    div[data-testid="stDataFrame"] { font-size: 13px; }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] { font-size: 1.1rem; color: #0e1b32; }
    
    /* Expander Styling */
    .streamlit-expanderHeader { font-weight: bold; color: #0e1b32; }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "BHP"
if 'spot_price' not in st.session_state: st.session_state.spot_price = 45.00
if 'range_pct' not in st.session_state: st.session_state.range_pct = 0.05
if 'vol_mod' not in st.session_state: st.session_state.vol_mod = 0.0
# Cache for API results
if 'chain_cache' not in st.session_state: st.session_state.chain_cache = None

# --- 3. MATH ENGINE ---
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

# --- 4. BRANDED HEADER ---
st.markdown(f"""
<div class="main-header">
    <div style="font-size: 22px; font-weight: 700;">
        TradersCircle <span style="font-weight: 300; opacity: 0.7;">| PRO</span>
    </div>
    <div style="text-align: right;">
        <div style="font-size: 12px; opacity: 0.8;">{st.session_state.ticker} (ASX)</div>
        <div style="font-size: 24px; font-weight: 700; color: #4ade80;">${st.session_state.spot_price:.2f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. UNIFIED WORKFLOW ---

# A. SEARCH BAR
c_search, c_btn, c_man = st.columns([3, 1, 1])
with c_search:
    search_input = st.text_input("Ticker Search", value=st.session_state.ticker, label_visibility="collapsed", placeholder="e.g. BHP, CBA, XJO")

# B. API LOGIC
api_data = None
fetch_trigger = c_btn.button("Load Chain", type="primary", use_container_width=True)

if fetch_trigger or (search_input != st.session_state.ticker):
    st.session_state.ticker = search_input.upper()
    
    # API KEY HANDLING
    try:
        API_KEY = st.secrets["EODHD_KEY"]
    except:
        API_KEY = "698428937d4174.67123401" # Fallback if secrets.toml is missing

    # 1. Get Spot Price
    try:
        t_sym = f"{st.session_state.ticker}.AU"
        url_spot = f"https://eodhd.com/api/real-time/{t_sym}?api_token={API_KEY}&fmt=json"
        r_spot = requests.get(url_spot).json()
        if r_spot and 'close' in r_spot:
            st.session_state.spot_price = float(r_spot['close'])
        
        # 2. Get Option Chain
        url_opt = f"https://eodhd.com/api/options/{t_sym}?api_token={API_KEY}&fmt=json"
        r_opt = requests.get(url_opt).json()
        
        if r_opt and 'data' in r_opt:
            st.session_state.chain_cache = r_opt['data']
        else:
            st.session_state.chain_cache = None
            st.warning("No option data found. Try a major stock.")
            
    except Exception as e:
        st.error(f"Connection Error: {e}")

# C. CHAIN DISPLAY
if st.session_state.chain_cache:
    data = st.session_state.chain_cache
    df_all = pd.DataFrame(data)
    
    # Expiry Selector
    exps = sorted(list(set(df_all['expirationDate'])))
    c_exp, c_pad = st.columns([2, 4])
    sel_exp = c_exp.selectbox("Select Expiry", exps)
    
    # Filter Chain
    chain = df_all[df_all['expirationDate'] == sel_exp].copy()
    
    # Split & Merge for Butterfly View
    calls = chain[chain['type'] == 'CALL'].set_index('strikePrice')
    puts = chain[chain['type'] == 'PUT'].set_index('strikePrice')
    
    # Create Display DF
    df_view = pd.DataFrame()
    df_view['STRIKE'] = chain['strikePrice'].unique()
    df_view = df_view.sort_values('STRIKE').set_index('STRIKE')
    
    # Map Official Codes & Prices
    df_view['C_Code'] = calls['contractName'] # Official Code
    df_view['C_Last'] = calls['lastPrice']
    df_view['C_Vol'] = calls['impliedVolatility']
    
    df_view['P_Last'] = puts['lastPrice']
    df_view['P_Code'] = puts['contractName'] # Official Code
    df_view['P_Vol'] = puts['impliedVolatility']
    
    df_view = df_view.fillna(0).reset_index()
    
    # Filter Near Money (Spot +/- 20%)
    center = st.session_state.spot_price
    df_view = df_view[(df_view['STRIKE'] > center*0.8) & (df_view['STRIKE'] < center*1.2)]
    
    # RENDER BUTTERFLY TABLE
    st.markdown(f"**Option Chain: {sel_exp}**")
    
    selection = st.dataframe(
        df_view,
        column_config={
            "C_Code": st.column_config.TextColumn("Call Code", width="small"),
            "C_Last": st.column_config.NumberColumn("Price", format="$%.2f"),
            "C_Vol": None,
            "STRIKE": st.column_config.NumberColumn("Strike", format="%.2f"),
            "P_Last": st.column_config.NumberColumn("Price", format="$%.2f"),
            "P_Code": st.column_config.TextColumn("Put Code", width="small"),
            "P_Vol": None
        },
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # ADD TO PORTFOLIO LOGIC
    if selection.selection['rows']:
        idx = selection.selection['rows'][0]
        row = df_view.iloc[idx]
        
        # Calc Days
        d_obj = datetime.strptime(sel_exp, "%Y-%m-%d")
        days = (d_obj - datetime.now()).days
        
        st.info(f"Selected: **{row['STRIKE']} Strike**")
        
        c1, c2, c3, c4 = st.columns(4)
        
        def add_leg(side, kind, code, price, iv):
            if iv < 0.01: iv = 0.20 # Fallback Vol
            
            st.session_state.legs.append({
                "Qty": 1 * (1 if side == "Buy" else -1),
                "Type": kind,
                "Strike": row['STRIKE'],
                "Expiry": days,
                "Vol": iv * 100,
                "Entry": price,
                "Code": code
            })
        
        if c1.button(f"Buy Call ({row['C_Code']})"):
            add_leg("Buy", "Call", row['C_Code'], row['C_Last'], row['C_Vol'])
            st.rerun()
            
        if c2.button(f"Sell Call"):
            add_leg("Sell", "Call", row['C_Code'], row['C_Last'], row['C_Vol'])
            st.rerun()
            
        if c3.button(f"Buy Put ({row['P_Code']})"):
            add_leg("Buy", "Put", row['P_Code'], row['P_Last'], row['P_Vol'])
            st.rerun()
            
        if c4.button(f"Sell Put"):
            add_leg("Sell", "Put", row['P_Code'], row['P_Last'], row['P_Vol'])
            st.rerun()

# D. MANUAL OVERRIDE
with st.expander("🛠 Manual / Custom Leg Builder"):
    with st.form("manual"):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        qty = c1.number_input("Qty", 1)
        kind = c2.selectbox("Type", ["Call", "Put", "Stock"])
        strike = c3.number_input("Strike", value=round(st.session_state.spot_price, 2))
        expiry = c4.number_input("Days", 45)
        vol = c5.number_input("Vol %", 20.0)
        cost = c6.number_input("Price", 1.00)
        
        if st.form_submit_button("Add Custom Leg"):
            st.session_state.legs.append({
                "Qty": qty, "Type": kind, "Strike": strike, 
                "Expiry": expiry, "Vol": vol, "Entry": cost, "Code": "CUSTOM"
            })
            st.rerun()

# --- 6. PORTFOLIO & ANALYSIS ---
if st.session_state.legs:
    st.markdown("---")
    
    # PORTFOLIO TABLE
    p_rows = []
    t_delta = 0
    
    for leg in st.session_state.legs:
        g = get_greeks(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
        d_net = g['delta'] * leg['Qty'] * 100
        t_delta += d_net
        
        p_rows.append({
            "Code": leg.get('Code', 'Custom'),
            "Qty": leg['Qty'],
            "Strike": leg['Strike'],
            "Expiry": f"{leg['Expiry']}d",
            "Entry": f"${leg['Entry']:.2f}",
            "Delta": f"{g['delta']:.3f}"
        })
        
    st.subheader("📋 Active Portfolio")
    st.dataframe(pd.DataFrame(p_rows), hide_index=True, use_container_width=True)
    
    c_m1, c_m2 = st.columns([1, 6])
    c_m1.metric("Net Delta", f"{t_delta:.1f}")
    if c_m2.button("Clear All"):
        st.session_state.legs = []
        st.rerun()

    # --- SCENARIOS ---
    st.markdown("---")
    st.subheader("📊 Scenario Analysis")
    
    # Controls
    c_ctrl1, c_ctrl2 = st.columns([1, 2])
    with c_ctrl1:
        st.caption("Price Range")
        rng_cols = st.columns(4)
        if rng_cols[0].button("1%"): st.session_state.range_pct = 0.01
        if rng_cols[1].button("5%"): st.session_state.range_pct = 0.05
        if rng_cols[2].button("10%"): st.session_state.range_pct = 0.10
        
    with c_ctrl2:
        st.caption("Time Increment (Days)")
        time_step = st.slider("days", 0, 60, 15, label_visibility="collapsed")
        
    # Charts & Matrix
    c_viz, c_mx = st.columns([2, 1])
    
    # Data Prep
    center = st.session_state.spot_price
    pct = st.session_state.range_pct
    prices = np.linspace(center*(1-pct), center*(1+pct), 100)
    
    pnl0, pnlF = [], []
    
    for p in prices:
        v0, vF = 0, 0
        for leg in st.session_state.legs:
            # T0
            px = get_bs_price(leg['Type'], p, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
            v0 += (px - leg['Entry']) * leg['Qty'] * 100
            # TF
            tf = max(0, leg['Expiry'] - time_step)
            pxf = get_bs_price(leg['Type'], p, leg['Strike'], tf, leg['Vol'], 4.0)
            vF += (pxf - leg['Entry']) * leg['Qty'] * 100
        pnl0.append(v0)
        pnlF.append(vF)
        
    with c_viz:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=pnl0, name="Today", line=dict(color='#2980b9', width=3)))
        if time_step > 0:
            fig.add_trace(go.Scatter(x=prices, y=pnlF, name=f"T+{time_step}d", line=dict(color='#e67e22', dash='dash')))
        fig.add_vline(x=center, line_dash="dot", annotation_text="Spot")
        fig.update_layout(height=400, template="plotly_white", margin=dict(t=20, b=20, l=40, r=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    with c_mx:
        # Matrix with Dates
        dates = [0, time_step, time_step*2]
        today = datetime.now()
        headers = [(today + timedelta(days=d)).strftime("%d %b") for d in dates]
        
        m_prices = np.linspace(center*(1-pct), center*(1+pct), 6)
        m_data = []
        for p in m_prices:
            row = {"Price": f"${p:.2f}"}
            for i, d in enumerate(dates):
                val = 0
                for leg in st.session_state.legs:
                    tf = max(0, leg['Expiry'] - d)
                    px = get_bs_price(leg['Type'], p, leg['Strike'], tf, leg['Vol'], 4.0)
                    val += (px - leg['Entry']) * leg['Qty'] * 100
                row[headers[i]] = val
            m_data.append(row)
            
        st.dataframe(pd.DataFrame(m_data).set_index("Price").style.background_gradient(cmap="RdYlGn", axis=None).format("${:,.0f}"), use_container_width=True, height=400)
