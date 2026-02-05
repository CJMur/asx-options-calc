import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="ASX Options Manager")

# Custom CSS for "TradeFloor" Look
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    
    /* Navy Header */
    .main-header {
        background-color: #0e1b32;
        padding: 1rem 2rem;
        color: white;
        border-radius: 8px;
        display: flex;
        justify_content: space-between;
        align_items: center;
        margin-bottom: 1rem;
    }
    
    /* Option Chain Table Styling */
    div[data-testid="stDataFrame"] {
        font-size: 12px;
    }
    
    /* Highlight "In The Money" */
    .itm-cell { background-color: #e6fffa; color: #004d40; }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "BHP.AX"
if 'spot_price' not in st.session_state: st.session_state.spot_price = 0.0
if 'vol_modifier' not in st.session_state: st.session_state.vol_modifier = 0.0
if 'chain_data' not in st.session_state: st.session_state.chain_data = None

# --- 3. MATH HELPER ---
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

# --- 4. DATA FETCHING (NEW) ---
def fetch_chain(ticker_symbol):
    """Fetches option chain from Yahoo Finance and merges Calls/Puts"""
    try:
        tk = yf.Ticker(ticker_symbol)
        
        # Get Spot
        hist = tk.history(period="1d")
        if not hist.empty:
            st.session_state.spot_price = float(hist['Close'].iloc[-1])
            
        # Get Expiries
        exps = tk.options
        if not exps: return None, []
        
        return tk, exps
    except Exception as e:
        return None, []

# --- 5. HEADER UI ---
st.markdown(f"""
<div class="main-header">
    <div style="font-size: 24px; font-weight: 600;">
        TradersCircle <span style="font-weight: 300; opacity: 0.8;">| PORTFOLIO</span>
    </div>
    <div style="text-align: right;">
        <div style="font-size: 12px; opacity: 0.8;">{st.session_state.ticker.replace('.AX','')}</div>
        <div style="font-size: 20px; font-weight: bold;">${st.session_state.spot_price:.2f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 6. STRATEGY BUILDER (SEARCH & CHAIN) ---
with st.container():
    st.markdown("### 🔍 Option Chain Search")
    
    # 1. Search Bar
    c_input, c_fetch = st.columns([4, 1])
    user_ticker = c_input.text_input("Search Ticker", value="BHP", label_visibility="collapsed").upper()
    
    # Handle ASX Suffix
    search_symbol = f"{user_ticker}.AX" if not user_ticker.endswith(".AX") else user_ticker
    
    if search_symbol != st.session_state.ticker or st.session_state.spot_price == 0:
        st.session_state.ticker = search_symbol
        tk_obj, dates = fetch_chain(search_symbol)
        st.session_state.tk_obj = tk_obj
        st.session_state.dates = dates
    else:
        # Restore from state if exists
        tk_obj = getattr(st.session_state, 'tk_obj', None)
        dates = getattr(st.session_state, 'dates', [])

    if dates:
        # 2. Date Selector (Tabs)
        # We limit to first 5 dates for UI cleanliness
        tabs = st.tabs([d for d in dates[:6]])
        
        for i, date_str in enumerate(dates[:6]):
            with tabs[i]:
                # Fetch Chain for this date
                try:
                    chain = tk_obj.option_chain(date_str)
                    calls = chain.calls
                    puts = chain.puts
                    
                    # Merge on Strike to create "Butterfly"
                    df_calls = calls[['contractSymbol', 'strike', 'lastPrice', 'impliedVolatility', 'inTheMoney']].copy()
                    df_calls.columns = ['C_Code', 'Strike', 'C_Last', 'C_IV', 'C_ITM']
                    
                    df_puts = puts[['contractSymbol', 'strike', 'lastPrice', 'impliedVolatility', 'inTheMoney']].copy()
                    df_puts.columns = ['P_Code', 'Strike', 'P_Last', 'P_IV', 'P_ITM']
                    
                    # Merge
                    df_chain = pd.merge(df_calls, df_puts, on="Strike", how="outer").sort_values("Strike")
                    df_chain = df_chain.fillna(0)
                    
                    # Filter near money (optional, keeps table short)
                    center_idx = (df_chain['Strike'] - st.session_state.spot_price).abs().idxmin()
                    start = max(0, center_idx - 8)
                    end = min(len(df_chain), center_idx + 9)
                    df_view = df_chain.iloc[start:end].reset_index(drop=True)

                    # --- RENDER TABLE ---
                    # We use Streamlit's styling to highlight ITM green
                    st.caption(f"Expiry: {date_str} | Showing strikes near ${st.session_state.spot_price:.2f}")
                    
                    def color_coding(val):
                        # Simple alternating or conditional logic could go here
                        return ''

                    # Selectable Grid
                    # We configure columns to look like the screenshot
                    event = st.dataframe(
                        df_view,
                        column_config={
                            "C_ITM": st.column_config.CheckboxColumn("ITM", width="small"),
                            "C_Last": st.column_config.NumberColumn("Call Price", format="$%.2f"),
                            "C_IV": st.column_config.NumberColumn("Vol", format="%.1f%%"),
                            "Strike": st.column_config.NumberColumn("Strike", format="$%d"),
                            "P_Last": st.column_config.NumberColumn("Put Price", format="$%.2f"),
                            "P_IV": st.column_config.NumberColumn("Vol", format="%.1f%%"),
                            "P_ITM": st.column_config.CheckboxColumn("ITM", width="small"),
                            "C_Code": None, "P_Code": None # Hide codes
                        },
                        use_container_width=True,
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="single-row"
                    )
                    
                    # 3. Add to Strategy Logic
                    if len(event.selection['rows']) > 0:
                        idx = event.selection['rows'][0]
                        selected_row = df_view.iloc[idx]
                        
                        st.divider()
                        st.markdown(f"**Selected Strike: ${selected_row['Strike']:.2f}**")
                        
                        c_col, p_col = st.columns(2)
                        
                        # CALL SIDE BUTTONS
                        with c_col:
                            st.info(f"CALL (Price: ${selected_row['C_Last']:.2f})")
                            c1, c2 = st.columns(2)
                            if c1.button(f"Buy Call ${selected_row['Strike']}", key=f"bc_{i}"):
                                days = (datetime.strptime(date_str, "%Y-%m-%d") - datetime.now()).days
                                st.session_state.legs.append({
                                    "Qty": 1, "Type": "Call", "Strike": selected_row['Strike'],
                                    "Expiry": days, "Vol": selected_row['C_IV']*100, "Entry": selected_row['C_Last']
                                })
                                st.rerun()
                            if c2.button(f"Sell Call ${selected_row['Strike']}", key=f"sc_{i}"):
                                days = (datetime.strptime(date_str, "%Y-%m-%d") - datetime.now()).days
                                st.session_state.legs.append({
                                    "Qty": -1, "Type": "Call", "Strike": selected_row['Strike'],
                                    "Expiry": days, "Vol": selected_row['C_IV']*100, "Entry": selected_row['C_Last']
                                })
                                st.rerun()

                        # PUT SIDE BUTTONS
                        with p_col:
                            st.info(f"PUT (Price: ${selected_row['P_Last']:.2f})")
                            p1, p2 = st.columns(2)
                            if p1.button(f"Buy Put ${selected_row['Strike']}", key=f"bp_{i}"):
                                days = (datetime.strptime(date_str, "%Y-%m-%d") - datetime.now()).days
                                st.session_state.legs.append({
                                    "Qty": 1, "Type": "Put", "Strike": selected_row['Strike'],
                                    "Expiry": days, "Vol": selected_row['P_IV']*100, "Entry": selected_row['P_Last']
                                })
                                st.rerun()
                            if p2.button(f"Sell Put ${selected_row['Strike']}", key=f"sp_{i}"):
                                days = (datetime.strptime(date_str, "%Y-%m-%d") - datetime.now()).days
                                st.session_state.legs.append({
                                    "Qty": -1, "Type": "Put", "Strike": selected_row['Strike'],
                                    "Expiry": days, "Vol": selected_row['P_IV']*100, "Entry": selected_row['P_Last']
                                })
                                st.rerun()

                except Exception as e:
                    st.error(f"Could not load chain for {date_str}. Yahoo Finance data may be missing.")
    else:
        st.warning("No option data found for this ticker. Try 'BHP', 'CBA', or 'XJO'.")

# --- 7. ACTIVE PORTFOLIO TABLE ---
if st.session_state.legs:
    st.markdown("### 📋 Portfolio")
    
    # Process Portfolio Rows
    rows = []
    tot_delta = 0
    
    for idx, leg in enumerate(st.session_state.legs):
        # Live Greeks
        g = get_greeks(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
        d_pos = g['delta'] * leg['Qty'] * 100
        tot_delta += d_pos
        
        rows.append({
            "Qty": leg['Qty'],
            "Type": leg['Type'],
            "Strike": leg['Strike'],
            "Expiry": f"{leg['Expiry']}d",
            "Vol": f"{leg['Vol']:.1f}%",
            "Entry": leg['Entry'],
            "Delta": f"{g['delta']:.3f}",
            "Net Delta": f"{d_pos:.1f}"
        })
        
    df_port = pd.DataFrame(rows)
    st.dataframe(df_port, use_container_width=True, hide_index=True)
    
    col_d, col_clr = st.columns([1, 6])
    col_d.metric("Net Delta", f"{tot_delta:.1f}")
    if col_clr.button("Clear Portfolio"):
        st.session_state.legs = []
        st.rerun()

# --- 8. VISUALIZATION & MATRIX ---
if st.session_state.legs:
    st.markdown("---")
    st.markdown("### 📉 Payoff Analysis")
    
    # Simple Controls
    vol_shift = st.slider("Volatility Shock (%)", -50, 50, 0)
    
    # Chart
    center = st.session_state.spot_price
    rng = np.linspace(center*0.8, center*1.2, 100)
    pnl = []
    
    for p in rng:
        val = 0
        for leg in st.session_state.legs:
            # Pricing now vs entry
            curr = get_bs_price(leg['Type'], p, leg['Strike'], leg['Expiry'], leg['Vol']+vol_shift, 4.0)
            val += (curr - leg['Entry']) * leg['Qty'] * 100
        pnl.append(val)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rng, y=pnl, fill='tozeroy', name='P&L', line=dict(color='#00C805')))
    fig.add_vline(x=center, line_dash='dash', annotation_text="Spot")
    fig.update_layout(template="plotly_dark", height=400, title=f"Projected P&L (Vol {vol_shift:+}%)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Matrix (Fixed with matplotlib)
    st.markdown("#### Scenario Matrix")
    matrix_res = []
    test_prices = np.linspace(center*0.9, center*1.1, 7)
    dates = [0, 15, 30]
    
    for tp in test_prices:
        row = {"Price": tp}
        for d in dates:
            v = 0
            for leg in st.session_state.legs:
                rem_t = max(0, leg['Expiry'] - d)
                px = get_bs_price(leg['Type'], tp, leg['Strike'], rem_t, leg['Vol']+vol_shift, 4.0)
                v += (px - leg['Entry']) * leg['Qty'] * 100
            row[f"T+{d}"] = v
        matrix_res.append(row)
        
    df_mx = pd.DataFrame(matrix_res).set_index("Price")
    st.dataframe(df_mx.style.background_gradient(cmap="RdYlGn", axis=None).format("${:,.0f}"), use_container_width=True)
