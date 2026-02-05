import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="ASX Options Manager")

# Custom CSS for a "Pro" Financial Look
st.markdown("""
<style>
    /* Remove default top padding */
    .block-container { padding-top: 1rem; }
    
    /* TradeFloor-style Navy Header */
    .main-header {
        background-color: #0e1b32;
        padding: 1rem 2rem;
        color: white;
        border-radius: 8px;
        display: flex;
        justify_content: space-between;
        align_items: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Card Styling for Sections */
    .stCard {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
        color: #0e1b32;
    }
    
    /* Table Headers */
    thead tr th:first-child { display:none }
    tbody th { display:none }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "BHP.AX"
if 'spot_price' not in st.session_state: st.session_state.spot_price = 45.00
if 'vol_modifier' not in st.session_state: st.session_state.vol_modifier = 0.0

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

# --- 4. HEADER UI ---
st.markdown(f"""
<div class="main-header">
    <div style="font-size: 24px; font-weight: 600; letter-spacing: 1px;">
        TradersCircle <span style="font-weight: 300; opacity: 0.8;">| PORTFOLIO</span>
    </div>
    <div style="background: rgba(255,255,255,0.1); padding: 5px 15px; border-radius: 20px; font-size: 14px;">
        ● Live Price: <b>${st.session_state.spot_price:.2f}</b>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. STRATEGY INPUTS (The "Trade Ticket") ---
with st.container():
    st.markdown("### 🛠 Strategy Builder")
    
    # Top Row: Search & Spot
    c_search, c_spot = st.columns([3, 1])
    new_ticker = c_search.text_input("Underlying Asset", value=st.session_state.ticker, label_visibility="collapsed", placeholder="Search Ticker (e.g. CBA.AX)")
    
    # Auto-fetch price logic
    if new_ticker != st.session_state.ticker:
        st.session_state.ticker = new_ticker
        try:
            tick = yf.Ticker(new_ticker)
            hist = tick.history(period="1d")
            if not hist.empty:
                st.session_state.spot_price = float(hist['Close'].iloc[-1])
        except: pass

    # Input Bar (Horizontal Layout like screenshot)
    with st.form("add_leg_form", clear_on_submit=False):
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1.5, 1.5, 2, 2, 2, 1])
        qty = c1.number_input("Qty", min_value=1, value=1)
        action = c2.selectbox("Action", ["Buy", "Sell"])
        l_type = c3.selectbox("Type", ["Call", "Put"])
        strike = c4.number_input("Strike ($)", value=round(st.session_state.spot_price, 0))
        expiry = c5.number_input("Days to Exp", value=45)
        vol = c6.number_input("Vol (%)", value=20.0)
        
        submitted = c7.form_submit_button("➕ Add", type="primary")
        
        if submitted:
            theo = get_bs_price(l_type, st.session_state.spot_price, strike, expiry, vol, 4.0)
            st.session_state.legs.append({
                "Qty": qty * (1 if action == "Buy" else -1),
                "Type": l_type,
                "Strike": strike,
                "Expiry": expiry,
                "Vol": vol,
                "Entry": theo
            })
            st.rerun()

# --- 6. POSITION TABLE ---
if st.session_state.legs:
    st.markdown("### 📋 Active Legs")
    
    # Calculate Live Values for Table
    table_rows = []
    portfolio_delta = 0
    
    for i, leg in enumerate(st.session_state.legs):
        # Calculate Greeks
        greeks = get_greeks(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
        
        # Position Delta
        pos_delta = greeks['delta'] * leg['Qty'] * 100
        portfolio_delta += pos_delta
        
        table_rows.append({
            "Side": "LONG" if leg['Qty'] > 0 else "SHORT",
            "Qty": abs(leg['Qty']),
            "Type": leg['Type'],
            "Strike": leg['Strike'],
            "Expiry": f"{leg['Expiry']} Days",
            "Entry Price": leg['Entry'],
            "Live Delta": f"{greeks['delta']:.3f}",
            "Net Delta": f"{pos_delta:.1f}"
        })
    
    # Render Table
    df = pd.DataFrame(table_rows)
    st.dataframe(
        df, 
        use_container_width=True,
        column_config={
            "Entry Price": st.column_config.NumberColumn(format="$%.2f"),
            "Strike": st.column_config.NumberColumn(format="$%.2f"),
        },
        hide_index=True
    )
    
    # Portfolio Summary Footer
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Net Delta", f"{portfolio_delta:.2f}")
    if st.button("Clear All Legs"):
        st.session_state.legs = []
        st.rerun()

# --- 7. ANALYSIS ENGINE ---
if st.session_state.legs:
    st.markdown("---")
    st.markdown("### 📈 Scenario Analysis")
    
    left_panel, right_panel = st.columns([1, 3])
    
    with left_panel:
        st.info("Market Shocks")
        st.session_state.vol_modifier = st.slider("Volatility Shock", -50, 50, 0, format="%d%%")
        time_slide = st.slider("Time Jump (Days)", 0, 90, 0)
        range_pct = st.slider("Price Range", 5, 40, 15, format="+/-%d%%")

    with right_panel:
        # Chart Logic
        center = st.session_state.spot_price
        prices = np.linspace(center * (1 - range_pct/100), center * (1 + range_pct/100), 50)
        
        pnl_now = []
        pnl_future = []
        
        for p in prices:
            val_now = 0
            val_fut = 0
            for leg in st.session_state.legs:
                # Modifiers
                sim_vol = leg['Vol'] + st.session_state.vol_modifier
                
                # T+0 Value
                px_0 = get_bs_price(leg['Type'], p, leg['Strike'], leg['Expiry'], sim_vol, 4.0)
                val_now += (px_0 - leg['Entry']) * leg['Qty'] * 100
                
                # T+Future Value
                t_fut = max(0, leg['Expiry'] - time_slide)
                px_f = get_bs_price(leg['Type'], p, leg['Strike'], t_fut, sim_vol, 4.0)
                val_fut += (px_f - leg['Entry']) * leg['Qty'] * 100
            
            pnl_now.append(val_now)
            pnl_future.append(val_fut)
            
        # Plotly Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=pnl_now, name="Today (T+0)", line=dict(color='#00C805', width=3)))
        if time_slide > 0:
            fig.add_trace(go.Scatter(x=prices, y=pnl_future, name=f"Future (T+{time_slide})", line=dict(color='#FF4B4B', dash='dash')))
            
        fig.add_vline(x=center, line_dash="dot", line_color="gray", annotation_text="Spot")
        fig.add_hline(y=0, line_color="white", opacity=0.2)
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(t=30, b=20, l=20, r=20),
            xaxis_title="Underlying Price",
            yaxis_title="Profit / Loss ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 8. DATA MATRIX (The Error Fix) ---
    st.markdown("#### 🔢 Payoff Matrix")
    
    # Calculate Grid
    grid_prices = np.linspace(center * 0.95, center * 1.05, 7)
    grid_dates = [0, 15, 30]
    
    matrix_data = []
    for p in grid_prices:
        row = {"Price": p}  # Keep as float for now
        for d in grid_dates:
            val = 0
            for leg in st.session_state.legs:
                t = max(0, leg['Expiry'] - d)
                px = get_bs_price(leg['Type'], p, leg['Strike'], t, leg['Vol'] + st.session_state.vol_modifier, 4.0)
                val += (px - leg['Entry']) * leg['Qty'] * 100
            row[f"T+{d}"] = val
        matrix_data.append(row)
    
    df_mx = pd.DataFrame(matrix_data).set_index("Price")
    
    # Apply Styling (Requires matplotlib in requirements.txt)
    st.dataframe(
        df_mx.style.background_gradient(cmap="RdYlGn", axis=None)
             .format("${:,.0f}"),
        use_container_width=True
    )
