import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="ASX Options Manager")

# Custom CSS to replicate the "TradeFloor" Dark Blue Header and clean look
st.markdown("""
<style>
    /* Main Header Bar */
    .main-header {
        background-color: #0e1b32; /* Dark Blue */
        padding: 15px;
        color: white;
        border-radius: 5px;
        margin-bottom: 20px;
        display: flex;
        justify_content: space-between;
        align_items: center;
    }
    .nav-link { margin-right: 20px; font-weight: bold; color: #a0c4ff; }
    .metric-box {
        background-color: #f8f9fa;
        border-left: 5px solid #0e1b32;
        padding: 10px;
        margin: 5px 0;
    }
    /* Button Styling to match screenshot pills */
    div.stButton > button {
        width: 100%;
        border-radius: 20px;
        font-size: 12px;
    }
    /* Compact tables */
    .dataframe { font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE MANAGEMENT ---
# We store the "Strategy" (list of legs) in memory so it doesn't vanish on reload
if 'legs' not in st.session_state:
    st.session_state.legs = [] 
if 'ticker' not in st.session_state:
    st.session_state.ticker = "BHP.AX"
if 'spot_price' not in st.session_state:
    st.session_state.spot_price = 45.00
if 'vol_modifier' not in st.session_state:
    st.session_state.vol_modifier = 0.0 # 0% change default

# --- 3. HELPER FUNCTIONS (THE MATH ENGINE) ---

def get_bs_price(kind, spot, strike, time_days, vol_pct, rate_pct):
    """Calculates Black-Scholes price safely."""
    try:
        if time_days <= 0: time_days = 0.001 # Avoid division by zero
        c = mibian.BS([spot, strike, rate_pct, time_days], volatility=vol_pct)
        return c.callPrice if kind == 'Call' else c.putPrice
    except:
        return 0.0

def get_greeks(kind, spot, strike, time_days, vol_pct, rate_pct):
    try:
        if time_days <= 0: time_days = 0.001
        c = mibian.BS([spot, strike, rate_pct, time_days], volatility=vol_pct)
        g = c.call if kind == 'Call' else c.put
        return {'delta': g.delta, 'gamma': g.gamma, 'theta': g.theta, 'vega': g.vega}
    except:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

# --- 4. UI: HEADER BAR ---
st.markdown(f"""
<div class="main-header">
    <div>
        <span style="font-size: 20px; font-weight: bold;">TradersCircle</span>
        <span style="margin-left: 30px;">PORTFOLIO</span>
        <span style="margin-left: 20px;">PERFORMANCE</span>
    </div>
    <div>
        <span style="background-color: #2c3e50; padding: 5px 15px; border-radius: 15px;">
            🔥 Price Override: ${st.session_state.spot_price:.2f}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. STRATEGY BUILDER SECTION ---

col_search, col_act = st.columns([3, 1])

with col_search:
    new_ticker = st.text_input("Enter Position (Code or Search)...", value=st.session_state.ticker)
    if new_ticker != st.session_state.ticker:
        # Fetch new price if ticker changes
        st.session_state.ticker = new_ticker
        try:
            tick = yf.Ticker(new_ticker)
            hist = tick.history(period="1d")
            if not hist.empty:
                st.session_state.spot_price = float(hist['Close'].iloc[-1])
        except:
            pass

# Add Leg Form (Hidden in an expander to keep UI clean, or inline)
with st.expander("➕ Configure New Leg", expanded=True):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    qty = c1.number_input("Qty", value=1, step=1)
    l_type = c2.selectbox("Type", ["Call", "Put", "Stock"])
    action = c3.selectbox("Side", ["Buy", "Sell"])
    strike = c4.number_input("Strike", value=round(st.session_state.spot_price, 0))
    expiry_days = c5.number_input("Days to Exp", value=45)
    vol = c6.number_input("Implied Vol (%)", value=20.0)
    
    if st.button("Create Position"):
        # Calculate Initial Theo
        theo = get_bs_price(l_type, st.session_state.spot_price, strike, expiry_days, vol, 4.0)
        
        new_leg = {
            "Quantity": qty * (1 if action == "Buy" else -1),
            "Security": st.session_state.ticker,
            "Type": l_type,
            "Strike": strike,
            "Expiry": expiry_days,
            "Vol": vol,
            "Entry Price": theo, # Assuming we enter at current theo
            "Side": action
        }
        st.session_state.legs.append(new_leg)
        st.rerun()

# --- 6. STRATEGY TABLE (REPLICATING SCREENSHOT) ---
if st.session_state.legs:
    st.subheader(f"Strategy: {len(st.session_state.legs)} Legs")
    
    # Process Data for Table
    table_data = []
    total_delta = 0
    total_margin = 0
    
    for leg in st.session_state.legs:
        # Recalculate live Greeks
        greeks = get_greeks(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
        theo = get_bs_price(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
        
        net_delta = greeks['delta'] * leg['Quantity'] * 100 # Standard contract size 100
        total_delta += net_delta
        
        row = {
            "Qty": leg['Quantity'],
            "Security": leg['Security'],
            "Strike": leg['Strike'],
            "Expiry (Days)": leg['Expiry'],
            "Type": leg['Type'],
            "Theo": f"{theo:.3f}",
            "Delta": f"{greeks['delta']:.3f}",
            "Net Delta": f"{net_delta:.1f}",
            "Entry": f"{leg['Entry Price']:.3f}"
        }
        table_data.append(row)

    # Display Table
    df_legs = pd.DataFrame(table_data)
    st.dataframe(df_legs, use_container_width=True, hide_index=True)
    
    # Totals Row (Manual imitation)
    t1, t2, t3 = st.columns(3)
    t1.metric("Net Portfolio Delta", f"{total_delta:.2f}")
    t2.metric("Est. Margin Impact", "$0.00") # Placeholder for SPAN logic
    if st.button("Clear Strategy", type="primary"):
        st.session_state.legs = []
        st.rerun()

else:
    st.info("No active legs. Add a position above to begin.")


# --- 7. PAYOFF MATRIX & ANALYSIS SECTION ---
st.markdown("---")
st.subheader("📊 Payoff Matrix & Scenarios")

col_controls, col_viz = st.columns([1, 2])

# --- CONTROLS (Left Side) ---
with col_controls:
    st.markdown("#### Volatility Modifier")
    # Using columns for the pill buttons
    v1, v2, v3, v4 = st.columns(4)
    if v1.button("+5%"): st.session_state.vol_modifier += 5
    if v2.button("+10%"): st.session_state.vol_modifier += 10
    if v3.button("-5%"): st.session_state.vol_modifier -= 5
    if v4.button("-10%"): st.session_state.vol_modifier -= 10
    
    st.markdown(f"**Current Adjustment: {st.session_state.vol_modifier}%**")
    
    st.markdown("#### Time Increment")
    time_slide = st.slider("Days Forward", 0, 90, 0)
    
    st.markdown("#### Price Range")
    price_range_pct = st.slider("Range +/- %", 5, 50, 20)

# --- VISUALIZATION (Right Side) ---
with col_viz:
    if st.session_state.legs:
        # 1. Generate Price Range
        lower = st.session_state.spot_price * (1 - (price_range_pct/100))
        upper = st.session_state.spot_price * (1 + (price_range_pct/100))
        prices = np.linspace(lower, upper, 40)
        
        # 2. Calculate P&L Curves (Current vs Time Forward)
        fig = go.Figure()
        
        # Curve 1: T+0 (Today)
        pnl_today = []
        pnl_future = []
        
        for p in prices:
            # Scenario A: Today
            val_today = 0
            val_future = 0
            
            for leg in st.session_state.legs:
                # Modifiers
                mod_vol = leg['Vol'] + st.session_state.vol_modifier
                
                # Valuation Today
                px_now = get_bs_price(leg['Type'], p, leg['Strike'], leg['Expiry'], mod_vol, 4.0)
                val_today += (px_now - leg['Entry Price']) * leg['Quantity'] * 100
                
                # Valuation at T+Slide
                t_future = max(0, leg['Expiry'] - time_slide)
                px_fut = get_bs_price(leg['Type'], p, leg['Strike'], t_future, mod_vol, 4.0)
                val_future += (px_fut - leg['Entry Price']) * leg['Quantity'] * 100
            
            pnl_today.append(val_today)
            pnl_future.append(val_future)

        # Add Traces
        fig.add_trace(go.Scatter(x=prices, y=pnl_today, mode='lines', name='T+0 (Today)', line=dict(color='blue')))
        if time_slide > 0:
            fig.add_trace(go.Scatter(x=prices, y=pnl_future, mode='lines', name=f'T+{time_slide} Days', line=dict(color='orange', dash='dash')))
        
        # Zero Line & Spot Line
        fig.add_hline(y=0, line_color="white", opacity=0.3)
        fig.add_vline(x=st.session_state.spot_price, line_color="gray", line_dash="dot", annotation_text="Spot")
        
        fig.update_layout(
            title="Payoff by Price & Time",
            xaxis_title="Stock Price",
            yaxis_title="P&L ($)",
            template="plotly_dark",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- THE MATRIX (Bottom Dataframe) ---
        # Generate a small grid of P&L for specific Dates vs Prices
        st.markdown("#### Data Matrix")
        matrix_dates = [0, 15, 30, 45] # Days forward
        matrix_prices = np.linspace(st.session_state.spot_price*0.9, st.session_state.spot_price*1.1, 7)
        
        matrix_data = []
        for mp in matrix_prices:
            row = {"Price": f"${mp:.2f}"}
            for d in matrix_dates:
                val = 0
                for leg in st.session_state.legs:
                    t_rem = max(0, leg['Expiry'] - d)
                    px = get_bs_price(leg['Type'], mp, leg['Strike'], t_rem, leg['Vol'] + st.session_state.vol_modifier, 4.0)
                    val += (px - leg['Entry Price']) * leg['Quantity'] * 100
                row[f"T+{d}"] = val
            matrix_data.append(row)
            
        df_matrix = pd.DataFrame(matrix_data).set_index("Price")
        # Style grid with color
        st.dataframe(df_matrix.style.background_gradient(cmap="RdYlGn", axis=None).format("${:.0f}"), use_container_width=True)