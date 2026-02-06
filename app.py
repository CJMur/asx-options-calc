import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="TradersCircle Options")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    
    /* Corporate Header */
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
    .trade-code {
        font-family: monospace;
        background-color: #f1f5f9;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #cbd5e1;
        font-weight: bold;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "BHP"
if 'spot_price' not in st.session_state: st.session_state.spot_price = 0.0
if 'range_pct' not in st.session_state: st.session_state.range_pct = 0.05
if 'chain_obj' not in st.session_state: st.session_state.chain_obj = None

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

# --- 4. DATA ENGINE (Yahoo Native) ---
def fetch_yahoo(ticker_input):
    clean_tk = ticker_input.upper().replace(".AX", "").strip()
    sym = f"{clean_tk}.AX"
    
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="1d")
        
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
            if tk.options:
                return "OK", price, tk
            else:
                return "NO_OPTS", price, None
                
    except Exception as e:
        return "ERROR", 0.0, None
    
    return "ERROR", 0.0, None

def clean_desc(date_str, strike, kind):
    # Generates a clean description like "19Feb $50.00 Call"
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        fmt_date = dt.strftime("%d%b")
        return f"{fmt_date} ${strike:.2f} {kind}"
    except:
        return f"{date_str} {strike} {kind}"

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
    with st.spinner("Fetching Yahoo Data..."):
        status, px, obj = fetch_yahoo(st.session_state.ticker)
        
        if status == "OK":
            st.session_state.spot_price = px
            st.session_state.chain_obj = obj
            st.rerun()
        elif status == "NO_OPTS":
            st.session_state.spot_price = px
            st.session_state.chain_obj = None
            st.warning(f"Found price for {query}, but no options data available.")
        else:
            st.error("Ticker not found.")

# --- 7. CHAIN DISPLAY (Yahoo) ---
if st.session_state.chain_obj:
    tk = st.session_state.chain_obj
    exps = tk.options
    
    sel_exp = st.selectbox("Select Expiry", exps)
    
    try:
        chain = tk.option_chain(sel_exp)
        calls = chain.calls
        puts = chain.puts
        
        # Merge for Butterfly
        df = pd.merge(calls, puts, on="strike", how="outer", suffixes=("_c", "_p"))
        df = df.fillna(0)
        
        # Filter Spot +/- 20%
        center = st.session_state.spot_price
        df = df[(df['strike'] > center*0.8) & (df['strike'] < center*1.2)]
        
        # Build View DF
        df_view = pd.DataFrame()
        df_view['STRIKE'] = df['strike']
        
        # Data Columns
        df_view['C_Last'] = df['lastPrice_c']
        df_view['C_Vol'] = df['impliedVolatility_c']
        df_view['P_Last'] = df['lastPrice_p']
        df_view['P_Vol'] = df['impliedVolatility_p']

        # Descriptions
        df_view['C_Desc'] = df.apply(lambda x: clean_desc(sel_exp, x['strike'], "Call"), axis=1)
        df_view['P_Desc'] = df.apply(lambda x: clean_desc(sel_exp, x['strike'], "Put"), axis=1)

        st.markdown(f"**Chain: {sel_exp}**")
        selection = st.dataframe(
            df_view,
            column_config={
                "C_Desc": st.column_config.TextColumn("Call", width="medium"),
                "C_Last": st.column_config.NumberColumn("Price", format="$%.2f"),
                "STRIKE": st.column_config.NumberColumn("Strike", format="%.2f"),
                "P_Last": st.column_config.NumberColumn("Price", format="$%.2f"),
                "P_Desc": st.column_config.TextColumn("Put", width="medium"),
                "C_Vol": None, "P_Vol": None 
            },
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )

        if selection.selection['rows']:
            idx = selection.selection['rows'][0]
            row = df_view.iloc[idx]
            
            d_obj = datetime.strptime(sel_exp, "%Y-%m-%d")
            days = (d_obj - datetime.now()).days
            
            st.info(f"Selected: **${row['STRIKE']} Strike**")
            
            # Helper to Add with Placeholder Code
            def add_leg(side, kind, desc, px, iv):
                if iv < 0.01: iv = 0.20
                st.session_state.legs.append({
                    "Qty": 1 * (1 if side == "Buy" else -1),
                    "Type": kind, "Strike": row['STRIKE'], "Expiry": days,
                    "Vol": iv * 100, "Entry": px, "Code": "ENTER_CODE" # Placeholder
                })

            c1, c2, c3, c4 = st.columns(4)
            if c1.button("Buy Call"): add_leg("Buy", "Call", row['C_Desc'], row['C_Last'], row['C_Vol'])
            if c2.button("Sell Call"): add_leg("Sell", "Call", row['C_Desc'], row['C_Last'], row['C_Vol'])
            if c3.button("Buy Put"): add_leg("Buy", "Put", row['P_Desc'], row['P_Last'], row['P_Vol'])
            if c4.button("Sell Put"): add_leg("Sell", "Put", row['P_Desc'], row['P_Last'], row['P_Vol'])
            
    except Exception as e:
        st.error("No option data available for this expiry.")

# --- 8. MANUAL OVERRIDE ---
with st.expander("🛠 Manual / Custom Leg Builder"):
    with st.form("man"):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        code_in = c1.text_input("Code", "BHPS49")
        q = c2.number_input("Qty", 1)
        k = c3.selectbox("Type", ["Call", "Put"])
        s = c4.number_input("Strike", value=round(st.session_state.spot_price, 2))
        e = c5.number_input("Days", 45)
        v = c6.number_input("Vol", 20.0)
        if st.form_submit_button("Add Custom"):
            px = get_bs_price(k, st.session_state.spot_price, s, e, v, 4.0)
            st.session_state.legs.append({"Qty": q, "Type": k, "Strike": s, "Expiry": e, "Vol": v, "Entry": px, "Code": code_in})
            st.rerun()

# --- 9. PORTFOLIO & EDITABLE TABLE ---
if st.session_state.legs:
    st.markdown("---")
    st.subheader("📋 Active Portfolio (Edit Codes Here)")
    
    # PREPARE DATA FOR EDITOR
    # We convert the list of dicts to a DataFrame
    df_port = pd.DataFrame(st.session_state.legs)
    
    # We want users to be able to EDIT the "Code" column
    edited_df = st.data_editor(
        df_port,
        column_config={
            "Code": st.column_config.TextColumn("ASX Code (Edit Me)", width="medium", required=True),
            "Qty": st.column_config.NumberColumn("Qty", required=True),
            "Entry": st.column_config.NumberColumn("Entry Price", format="$%.3f"),
            "Vol": st.column_config.NumberColumn("Vol %", format="%.1f%%"),
            "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
        },
        use_container_width=True,
        num_rows="dynamic",
        key="editor"
    )
    
    # Sync edits back to session state logic (for Charts)
    # We recalculate Greeks based on the EDITED values
    # (e.g. if user changes Qty or Vol in the table, charts update!)
    
    updated_legs = edited_df.to_dict('records')
    st.session_state.legs = updated_legs # Update state
    
    # Calculate Portfolio Greeks
    tot_delta = 0
    for leg in updated_legs:
        g = get_greeks(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
        tot_delta += g['delta'] * leg['Qty'] * 100
    
    c_m, c_c = st.columns([1, 6])
    c_m.metric("Net Delta", f"{tot_delta:.1f}")
    
    # CHARTS
    st.markdown("---")
    st.subheader("📊 Scenario Analysis")
    
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
            for leg in updated_legs:
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
        
        # Simple Matrix (No Matplotlib dependency)
        st.write("**Quick Matrix**")
        mx_data = []
        chk_prices = [center*0.95, center, center*1.05]
        for cp in chk_prices:
            r = {"Spot": f"${cp:.2f}"}
            for d in [0, ts]:
                val = 0
                for leg in updated_legs:
                    tf = max(0, leg['Expiry'] - d)
                    px = get_bs_price(leg['Type'], cp, leg['Strike'], tf, leg['Vol'], 4.0)
                    val += (px - leg['Entry']) * leg['Qty'] * 100
                r[f"T+{d}"] = f"${val:.0f}"
            mx_data.append(r)
        st.dataframe(pd.DataFrame(mx_data), hide_index=True)
