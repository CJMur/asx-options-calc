import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="ASX Options Manager")

# Custom CSS for "TradersCircle" Professional Look
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    
    /* Navy Header matching your screenshot */
    .main-header {
        background-color: #0e1b32;
        padding: 1.5rem 2rem;
        color: white;
        border-radius: 8px;
        display: flex; 
        justify_content: space-between; 
        align_items: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1b32;
        color: white;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] { font-size: 1.1rem; color: #0e1b32; }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE MANAGEMENT ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "BHP.AX"
if 'spot_price' not in st.session_state: st.session_state.spot_price = 45.00

# --- 3. MATH ENGINE (Black-Scholes) ---
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

# --- 4. HEADER SECTION ---
st.markdown(f"""
<div class="main-header">
    <div style="font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">
        TradersCircle <span style="font-weight: 300; opacity: 0.7; font-size: 18px;">| PORTFOLIO MANAGER</span>
    </div>
    <div style="text-align: right;">
        <div style="font-size: 12px; opacity: 0.8; letter-spacing: 1px;">UNDERLYING ASSET</div>
        <div style="font-size: 24px; font-weight: 700;">
            {st.session_state.ticker.replace('.AX','')} <span style="color: #4ade80;">${st.session_state.spot_price:.2f}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. STRATEGY BUILDER ---
st.markdown("### 🛠 Strategy Builder")

# Tabs to separate Manual Entry (Reliable) from Chain Search (Experimental)
tab_manual, tab_search = st.tabs(["✍️ Manual Entry (Recommended)", "🔍 Chain Search (Live)"])

# === TAB 1: MANUAL BUILDER ===
with tab_manual:
    st.markdown("Use this for **Price Overrides** or when Yahoo Finance data is unavailable.")
    
    # Price Update Row
    c_tick, c_btn = st.columns([3, 1])
    with c_tick:
        new_ticker = st.text_input("Ticker Symbol", value=st.session_state.ticker, label_visibility="collapsed")
    with c_btn:
        if st.button("🔄 Update Spot Price", use_container_width=True):
            sym = f"{new_ticker}.AX" if not new_ticker.endswith(".AX") else new_ticker
            st.session_state.ticker = sym
            try:
                hist = yf.Ticker(sym).history(period="1d")
                if not hist.empty:
                    st.session_state.spot_price = float(hist['Close'].iloc[-1])
                    st.success(f"Updated: ${st.session_state.spot_price:.2f}")
            except:
                st.error("Ticker not found.")
            st.rerun()

    # The "Trade Ticket"
    with st.container():
        st.markdown("#### Add Position")
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1.2, 1.2, 1.5, 1.5, 1.5, 1])
        
        qty = c1.number_input("Qty", value=1, step=1)
        side = c2.selectbox("Side", ["Buy", "Sell"])
        kind = c3.selectbox("Type", ["Call", "Put"])
        strike = c4.number_input("Strike", value=round(st.session_state.spot_price, 1))
        expiry = c5.number_input("Days", value=45)
        vol = c6.number_input("Vol %", value=20.0)
        
        # Add Button Logic
        if c7.button("➕ Add", type="primary", use_container_width=True):
            # Auto-calc premium
            entry_px = get_bs_price(kind, st.session_state.spot_price, strike, expiry, vol, 4.0)
            
            st.session_state.legs.append({
                "Qty": qty * (1 if side == "Buy" else -1),
                "Type": kind,
                "Strike": strike,
                "Expiry": expiry,
                "Vol": vol,
                "Entry": entry_px
            })
            st.rerun()

# === TAB 2: CHAIN SEARCH ===
with tab_search:
    st.info("⚠️ Note: Calls Yahoo Finance. If no data appears, switch to the 'Manual Entry' tab.")
    
    search_tk = st.text_input("Search Option Chain (e.g. BHP)", value="BHP").upper()
    
    if st.button("Load Chain"):
        sym = f"{search_tk}.AX" if not search_tk.endswith(".AX") else search_tk
        try:
            tk = yf.Ticker(sym)
            dates = tk.options
            
            if not dates:
                st.warning(f"No option data found for {sym}. Try Manual Entry.")
            else:
                st.success(f"Found Expiries: {dates[:3]}...")
                # Fetch first expiry for demo
                chain = tk.option_chain(dates[0])
                
                # Merge Calls and Puts for "Butterfly" View
                calls = chain.calls[['contractSymbol', 'strike', 'lastPrice', 'impliedVolatility']]
                calls.columns = ['C_Code', 'Strike', 'C_Last', 'C_IV']
                
                puts = chain.puts[['contractSymbol', 'strike', 'lastPrice', 'impliedVolatility']]
                puts.columns = ['P_Code', 'Strike', 'P_Last', 'P_IV']
                
                df = pd.merge(calls, puts, on="Strike", how="outer").fillna(0).sort_values("Strike")
                
                # Filter near spot
                center = st.session_state.spot_price
                df = df[(df['Strike'] > center*0.8) & (df['Strike'] < center*1.2)]
                
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.caption("To trade these, enter the Strike/Price in the Manual Tab.")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- 6. PORTFOLIO & ANALYSIS (Only shows if legs exist) ---
if st.session_state.legs:
    st.markdown("---")
    st.markdown("### 📋 Active Portfolio")
    
    # Portfolio Table
    rows = []
    tot_delta = 0
    
    for i, leg in enumerate(st.session_state.legs):
        g = get_greeks(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
        d_net = g['delta'] * leg['Qty'] * 100
        tot_delta += d_net
        
        rows.append({
            "Side": "LONG" if leg['Qty'] > 0 else "SHORT",
            "Qty": abs(leg['Qty']),
            "Type": leg['Type'],
            "Strike": leg['Strike'],
            "Expiry": f"{leg['Expiry']}d",
            "Entry Price": f"${leg['Entry']:.2f}",
            "Live Delta": f"{g['delta']:.3f}",
            "Net Delta": f"{d_net:.1f}"
        })
    
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    c_d, c_c = st.columns([1, 6])
    c_d.metric("Portfolio Delta", f"{tot_delta:.1f}")
    if c_c.button("Clear All"):
        st.session_state.legs = []
        st.rerun()

    # --- 7. CHARTS ---
    st.markdown("### 📉 Payoff Scenarios")
    
    col_chart, col_mx = st.columns([2, 1])
    
    with col_chart:
        # P&L Chart
        center = st.session_state.spot_price
        prices = np.linspace(center*0.85, center*1.15, 50)
        pnl = []
        
        for p in prices:
            val = 0
            for leg in st.session_state.legs:
                curr = get_bs_price(leg['Type'], p, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
                val += (curr - leg['Entry']) * leg['Qty'] * 100
            pnl.append(val)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=pnl, fill='tozeroy', line=dict(color='#2ecc71', width=2), name="P&L"))
        fig.add_vline(x=center, line_dash="dash", line_color="white", annotation_text="Spot")
        fig.add_hline(y=0, line_color="gray", opacity=0.5)
        
        fig.update_layout(
            template="plotly_dark", 
            height=400, 
            title="Profit/Loss at Expiry",
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_mx:
        st.markdown("**Payoff Matrix**")
        # Simplified Matrix
        mx_rows = []
        # 5 price points
        test_pts = np.linspace(center*0.9, center*1.1, 5)
        
        for p in test_pts:
            val = 0
            for leg in st.session_state.legs:
                curr = get_bs_price(leg['Type'], p, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
                val += (curr - leg['Entry']) * leg['Qty'] * 100
            mx_rows.append({"Spot": f"${p:.2f}", "P&L": val})
            
        df_mx = pd.DataFrame(mx_rows)
        # Gradient Coloring
        st.dataframe(
            df_mx.style.background_gradient(subset=["P&L"], cmap="RdYlGn").format({"P&L": "${:.2f}"}),
            use_container_width=True,
            hide_index=True
        )

else:
    st.info("👆 Use the **Manual Entry** tab above to add your first position.")
