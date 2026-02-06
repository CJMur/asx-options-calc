import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import requests
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="TradersCircle Options")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    
    /* Branding */
    .main-header {
        background-color: #0e1b32;
        padding: 1.5rem 2rem;
        color: white;
        border-radius: 8px;
        display: flex; justify_content: space-between; align_items: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Code Badge */
    .asx-code {
        font-family: 'Roboto Mono', monospace;
        background-color: #e2e8f0;
        color: #0f172a;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        border: 1px solid #cbd5e1;
    }
    
    /* Table Font */
    div[data-testid="stDataFrame"] { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "BHP"
if 'spot_price' not in st.session_state: st.session_state.spot_price = 0.0
if 'chain_data' not in st.session_state: st.session_state.chain_data = None
if 'range_pct' not in st.session_state: st.session_state.range_pct = 0.05

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

# --- 4. ASX DIRECT DATA ENGINE ---
def fetch_asx_official(ticker):
    # This hits the endpoint used by the ASX website itself
    clean_tk = ticker.replace(".AX","").upper()
    url = f"https://www.asx.com.au/asx/1/company/{clean_tk}/options?count=2000"
    
    # Headers are required to mimic a real browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200:
            return "ERROR", 0.0, None
            
        data = r.json()
        if not data: return "EMPTY", 0.0, None

        # 1. Extract Spot Price (From the first option record usually, or separate call)
        # ASX options JSON doesn't always have underlying spot. We fetch spot from Yahoo for reliability.
        try:
            import yfinance as yf
            spot = yf.Ticker(f"{clean_tk}.AX").history(period="1d")['Close'].iloc[-1]
        except:
            spot = 0.0 # Fallback

        # 2. Process Options
        # ASX JSON format: {'code': 'BHP', 'options': [{'code': 'BHPS49', 'expiryDate': '2026-02-19', 'strikePrice': 50.0, 'lastPrice': 1.45, 'type': 'Put'}]}
        # Actually it returns a list of dicts directly
        
        df = pd.DataFrame(data)
        # Rename columns to match our app logic
        # Expected cols: code (the option code), expiryDate, strikePrice, lastPrice, type (Put/Call)
        
        # Filter out weird stuff
        df = df[df['expiryDate'].notna()]
        
        return "OK", spot, df
        
    except Exception as e:
        return "ERROR", 0.0, str(e)

# --- 5. UI: HEADER ---
st.markdown(f"""
<div class="main-header">
    <div style="font-size: 22px; font-weight: 700;">
        TradersCircle <span style="font-weight: 300; opacity: 0.7;">| PRO</span>
    </div>
    <div style="text-align: right;">
        <div style="font-size: 12px; opacity: 0.8;">{st.session_state.ticker}</div>
        <div style="font-size: 24px; font-weight: 700; color: #4ade80;">
            ${st.session_state.spot_price:.2f}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 6. WORKFLOW ---
c_search, c_btn = st.columns([3, 1])
with c_search:
    query = st.text_input("Ticker Search", value=st.session_state.ticker, label_visibility="collapsed")

if c_btn.button("Load Chain", type="primary", use_container_width=True) or (query.upper() != st.session_state.ticker):
    st.session_state.ticker = query.upper()
    
    with st.spinner("Connecting to ASX.com.au..."):
        status, spot, df_chain = fetch_asx_official(st.session_state.ticker)
        
        if status == "OK":
            st.session_state.spot_price = spot
            st.session_state.chain_data = df_chain
        elif status == "EMPTY":
            st.warning("No options found on ASX website.")
        else:
            st.error("Could not connect to ASX. They may be blocking the request.")

# --- 7. CHAIN DISPLAY (ASX OFFICIAL) ---
if st.session_state.chain_data is not None and not st.session_state.chain_data.empty:
    df = st.session_state.chain_data
    
    # 1. Expiry Selector
    # ASX format is ISO '2026-02-19T00:00:00+1100' or similar. We need to clean it.
    # Often it is just '2026-02-19'. Let's handle both.
    df['expiry_clean'] = df['expiryDate'].apply(lambda x: x.split("T")[0])
    exps = sorted(list(set(df['expiry_clean'])))
    
    sel_exp = st.selectbox("Select Expiry", exps)
    
    # 2. Filter Chain
    subset = df[df['expiry_clean'] == sel_exp].copy()
    
    # 3. Pivot for Butterfly View
    # We need to separate Calls and Puts
    # ASX Type is usually 'Call' or 'Put'
    calls = subset[subset['type'] == 'Call'].set_index('strikePrice')
    puts = subset[subset['type'] == 'Put'].set_index('strikePrice')
    
    # Merge
    df_view = pd.DataFrame()
    all_strikes = sorted(list(set(calls.index) | set(puts.index)))
    df_view['STRIKE'] = all_strikes
    df_view = df_view.set_index('STRIKE')
    
    # Map Columns
    df_view['C_Code'] = calls['code']   # <--- THE HOLY GRAIL (e.g. BHPS49)
    df_view['C_Last'] = calls['lastPrice']
    df_view['C_Bid'] = calls['bidPrice']
    df_view['C_Ask'] = calls['offerPrice']
    
    df_view['P_Code'] = puts['code']    # <--- THE HOLY GRAIL
    df_view['P_Last'] = puts['lastPrice']
    df_view['P_Bid'] = puts['bidPrice']
    df_view['P_Ask'] = puts['offerPrice']
    
    df_view = df_view.fillna(0).reset_index()
    
    # Filter Near Spot
    center = st.session_state.spot_price
    if center > 0:
        df_view = df_view[(df_view['STRIKE'] > center*0.8) & (df_view['STRIKE'] < center*1.2)]
    
    # RENDER TABLE
    st.markdown(f"**Official ASX Chain: {sel_exp}**")
    
    selection = st.dataframe(
        df_view,
        column_config={
            "C_Code": st.column_config.TextColumn("Call Code", width="small"), # Visible Code!
            "C_Last": st.column_config.NumberColumn("Last", format="$%.2f"),
            "C_Bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
            "C_Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
            "STRIKE": st.column_config.NumberColumn("Strike", format="%.2f"),
            "P_Last": st.column_config.NumberColumn("Last", format="$%.2f"),
            "P_Bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
            "P_Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
            "P_Code": st.column_config.TextColumn("Put Code", width="small"), # Visible Code!
        },
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # SELECTION LOGIC
    if selection.selection['rows']:
        idx = selection.selection['rows'][0]
        row = df_view.iloc[idx]
        
        # Calc Days
        d_obj = datetime.strptime(sel_exp, "%Y-%m-%d")
        days = (d_obj - datetime.now()).days
        
        st.divider()
        st.markdown(f"**Selected: ${row['STRIKE']} Strike**")
        
        c1, c2, c3, c4 = st.columns(4)
        
        # Helper function
        def add(side, kind, code, px):
            # If price is 0 (no trade), use Ask or fallback
            entry = px if px > 0 else 0.01 
            st.session_state.legs.append({
                "Qty": 1 * (1 if side == "Buy" else -1),
                "Type": kind, "Strike": row['STRIKE'], "Expiry": days,
                "Vol": 20.0, "Entry": entry, "Code": code # Store the official code
            })
            
        if c1.button(f"Buy Call ({row['C_Code']})"): add("Buy", "Call", row['C_Code'], row['C_Ask'])
        if c2.button(f"Sell Call ({row['C_Code']})"): add("Sell", "Call", row['C_Code'], row['C_Bid'])
        if c3.button(f"Buy Put ({row['P_Code']})"): add("Buy", "Put", row['P_Code'], row['P_Ask'])
        if c4.button(f"Sell Put ({row['P_Code']})"): add("Sell", "Put", row['P_Code'], row['P_Bid'])


# --- 8. MANUAL OVERRIDE ---
with st.expander("🛠 Manual Entry (If ASX is blocked)", expanded=False):
    with st.form("manual"):
        c1, c2, c3, c4, c5 = st.columns(5)
        code = c1.text_input("ASX Code (e.g. BHPS49)")
        q = c2.number_input("Qty", 1)
        k = c3.selectbox("Type", ["Call", "Put"])
        s = c4.number_input("Strike", value=round(st.session_state.spot_price, 2))
        e = c5.number_input("Days", 45)
        
        if st.form_submit_button("Add Code Manually"):
            st.session_state.legs.append({
                "Qty": q, "Type": k, "Strike": s, "Expiry": e, "Vol": 20.0, "Entry": 1.0, "Code": code
            })
            st.rerun()

# --- 9. PORTFOLIO & CHARTS ---
if st.session_state.legs:
    st.markdown("---")
    rows = []
    tot_delta = 0
    for leg in st.session_state.legs:
        g = get_greeks(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
        dn = g['delta'] * leg['Qty'] * 100
        tot_delta += dn
        
        # Format as Badge
        code_disp = leg.get('Code', 'CUSTOM')
        
        rows.append({
            "ASX Code": code_disp, 
            "Qty": leg['Qty'], 
            "Strike": leg['Strike'], 
            "Expiry": leg['Expiry'], 
            "Entry": f"${leg['Entry']:.3f}", 
            "Delta": f"{g['delta']:.3f}"
        })
        
    st.subheader("📋 Active Portfolio")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    
    col_m, col_c = st.columns([1, 6])
    col_m.metric("Net Delta", f"{tot_delta:.1f}")
    if col_c.button("Clear All"):
        st.session_state.legs = []
        st.rerun()

    # CHARTS (Same as before)
    st.markdown("---")
    c_ctrl, c_view = st.columns([1, 3])
    with c_ctrl:
        st.caption("Price Range")
        rc = st.columns(3)
        if rc[0].button("2%"): st.session_state.range_pct = 0.02
        if rc[1].button("5%"): st.session_state.range_pct = 0.05
        if rc[2].button("10%"): st.session_state.range_pct = 0.10
        st.caption("Time Step")
        ts = st.slider("Days", 0, 60, 15)
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
