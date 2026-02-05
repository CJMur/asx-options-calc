import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="ASX Options Manager")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    
    /* TradeFloor Navy Header */
    .main-header {
        background-color: #0e1b32;
        padding: 1.5rem 2rem;
        color: white;
        border-radius: 8px;
        display: flex; justify_content: space-between; align_items: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Pill Button Styling */
    .stButton > button {
        border-radius: 20px;
        font-size: 12px;
        border: 1px solid #ddd;
    }
    
    /* Option Chain specific styling */
    .chain-header {
        font-weight: bold; 
        text-align: center; 
        background-color: #f0f2f6; 
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "BHP.AX"
if 'spot_price' not in st.session_state: st.session_state.spot_price = 45.00
if 'range_pct' not in st.session_state: st.session_state.range_pct = 0.05
if 'vol_mod' not in st.session_state: st.session_state.vol_mod = 0.0

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

# --- 4. HEADER ---
st.markdown(f"""
<div class="main-header">
    <div style="font-size: 22px; font-weight: 700;">
        TradersCircle <span style="font-weight: 300; opacity: 0.7;">| PORTFOLIO</span>
    </div>
    <div style="text-align: right;">
        <div style="font-size: 12px; opacity: 0.8;">{st.session_state.ticker.replace('.AX','')}</div>
        <div style="font-size: 24px; font-weight: 700; color: #4ade80;">${st.session_state.spot_price:.2f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. STRATEGY BUILDER (The Core Update) ---
st.markdown("### 🛠 Strategy Builder")

tab_manual, tab_chain = st.tabs(["✍️ Manual Entry", "🦋 Option Chain (Butterfly)"])

# === TAB 1: MANUAL (Fallback) ===
with tab_manual:
    with st.container():
        c_tick, c_btn = st.columns([3, 1])
        new_ticker = c_tick.text_input("Underlying Ticker", value=st.session_state.ticker)
        if c_btn.button("Update Price"):
            sym = f"{new_ticker}.AX" if not new_ticker.endswith(".AX") else new_ticker
            st.session_state.ticker = sym
            try:
                hist = yf.Ticker(sym).history(period="1d")
                if not hist.empty:
                    st.session_state.spot_price = float(hist['Close'].iloc[-1])
            except: pass
            st.rerun()

        with st.form("manual"):
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1.5, 1.5, 1.5, 1])
            qty = c1.number_input("Qty", 1)
            side = c2.selectbox("Side", ["Buy", "Sell"])
            kind = c3.selectbox("Type", ["Call", "Put"])
            strike = c4.number_input("Strike", value=round(st.session_state.spot_price, 2))
            expiry = c5.number_input("Days", 45)
            vol = c6.number_input("Vol %", 20.0)
            if c7.form_submit_button("➕ Add"):
                ep = get_bs_price(kind, st.session_state.spot_price, strike, expiry, vol, 4.0)
                st.session_state.legs.append({
                    "Qty": qty * (1 if side == "Buy" else -1),
                    "Type": kind, "Strike": strike, "Expiry": expiry, "Vol": vol, "Entry": ep
                })
                st.rerun()

# === TAB 2: OPTION CHAIN (The "Vital" Part) ===
with tab_chain:
    st.info("Fetching live chain from Yahoo Finance. Note: Codes (e.g. BHP2602...) differ from ASX 6-letter codes, but pricing/strikes are accurate.")
    
    # 1. Search & Load Expiries
    col_s1, col_s2 = st.columns([3, 1])
    search_tk = col_s1.text_input("Search Ticker", value=st.session_state.ticker.replace(".AX","")).upper()
    
    if col_s2.button("Load Chain"):
        st.session_state.ticker = f"{search_tk}.AX"
        st.rerun()

    # Fetch Data
    try:
        tk = yf.Ticker(st.session_state.ticker)
        expiries = tk.options
        
        if not expiries:
            st.warning("No option data found. Try a major stock (BHP, CBA, XJO).")
        else:
            # 2. Expiry Selector (Tabs)
            # We limit to first 6 dates to keep UI clean
            sel_exp = st.selectbox("Select Expiry", expiries)
            
            # 3. Build Butterfly Table
            chain = tk.option_chain(sel_exp)
            calls = chain.calls
            puts = chain.puts
            
            # Merge on Strike
            # We keep 'contractSymbol' to show we have data, even if format differs
            df = pd.merge(calls, puts, on="strike", how="outer", suffixes=("_c", "_p"))
            df = df.fillna(0)
            
            # Filter near money (Spot +/- 20%)
            center = st.session_state.spot_price
            df = df[(df['strike'] > center * 0.8) & (df['strike'] < center * 1.2)]
            
            # Create Clean Display DataFrame
            # Structure: [Call Vol | Call Price | STRIKE | Put Price | Put Vol]
            display_df = pd.DataFrame()
            display_df['C_Vol'] = df['impliedVolatility_c'].map('{:.1%}'.format)
            display_df['C_Last'] = df['lastPrice_c']
            display_df['STRIKE'] = df['strike']
            display_df['P_Last'] = df['lastPrice_p']
            display_df['P_Vol'] = df['impliedVolatility_p'].map('{:.1%}'.format)
            display_df['C_Code'] = df['contractSymbol_c'] # Hidden ref
            display_df['P_Code'] = df['contractSymbol_p'] # Hidden ref
            
            # Render Interactive Table
            st.markdown(f"**Chain: {sel_exp}** (Spot: {center:.2f})")
            
            # We use st.dataframe with selection enabled
            selection = st.dataframe(
                display_df,
                column_config={
                    "C_Last": st.column_config.NumberColumn("Call $", format="$%.2f"),
                    "P_Last": st.column_config.NumberColumn("Put $", format="$%.2f"),
                    "STRIKE": st.column_config.NumberColumn("Strike", format="%.2f"),
                    "C_Code": None, "P_Code": None # Hide ugly codes
                },
                hide_index=True,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            # 4. Handle Selection (Click to Add)
            if selection.selection['rows']:
                idx = selection.selection['rows'][0]
                row = df.iloc[idx] # Get raw data row
                
                # Calculate Days to Expiry
                exp_date = datetime.strptime(sel_exp, "%Y-%m-%d")
                days_diff = (exp_date - datetime.now()).days
                
                st.divider()
                st.markdown(f"**Selected: Strike ${row['strike']:.2f}**")
                
                c_buy, c_sell = st.columns(2)
                
                # Action Buttons
                with c_buy:
                    if st.button(f"Buy Call @ {row['lastPrice_c']:.2f}"):
                        st.session_state.legs.append({
                            "Qty": 1, "Type": "Call", "Strike": row['strike'], 
                            "Expiry": days_diff, "Vol": row['impliedVolatility_c']*100, "Entry": row['lastPrice_c']
                        })
                        st.rerun()
                    if st.button(f"Buy Put @ {row['lastPrice_p']:.2f}"):
                        st.session_state.legs.append({
                            "Qty": 1, "Type": "Put", "Strike": row['strike'], 
                            "Expiry": days_diff, "Vol": row['impliedVolatility_p']*100, "Entry": row['lastPrice_p']
                        })
                        st.rerun()

                with c_sell:
                    if st.button(f"Sell Call @ {row['lastPrice_c']:.2f}"):
                        st.session_state.legs.append({
                            "Qty": -1, "Type": "Call", "Strike": row['strike'], 
                            "Expiry": days_diff, "Vol": row['impliedVolatility_c']*100, "Entry": row['lastPrice_c']
                        })
                        st.rerun()
                    if st.button(f"Sell Put @ {row['lastPrice_p']:.2f}"):
                        st.session_state.legs.append({
                            "Qty": -1, "Type": "Put", "Strike": row['strike'], 
                            "Expiry": days_diff, "Vol": row['impliedVolatility_p']*100, "Entry": row['lastPrice_p']
                        })
                        st.rerun()

    except Exception as e:
        st.error(f"Chain Error: {e}")

# --- 6. PORTFOLIO & SCENARIOS ---
if st.session_state.legs:
    st.markdown("---")
    
    # Portfolio Table
    rows = []
    tot_delta = 0
    for leg in st.session_state.legs:
        g = get_greeks(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], leg['Vol'], 4.0)
        net_d = g['delta'] * leg['Qty'] * 100
        tot_delta += net_d
        rows.append({
            "Qty": leg['Qty'], "Type": leg['Type'], "Strike": leg['Strike'], 
            "Expiry": f"{leg['Expiry']}d", "Entry": f"${leg['Entry']:.2f}", 
            "Delta": f"{g['delta']:.3f}", "Net Delta": f"{net_d:.1f}"
        })
    
    st.markdown("### 📋 Portfolio")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    c_met, c_clr = st.columns([1, 6])
    c_met.metric("Total Delta", f"{tot_delta:.1f}")
    if c_clr.button("Clear All"):
        st.session_state.legs = []
        st.rerun()

    # --- 7. SCENARIO ANALYSIS (FIXED DATES & PILLS) ---
    st.markdown("---")
    st.subheader("📊 Payoff Matrix & Scenarios")

    # Controls
    c_p1, c_p2, c_p3 = st.columns([1, 2, 1])
    
    # Price Increments (Pills)
    with c_p1:
        st.caption("Price Range")
        cols = st.columns(4)
        if cols[0].button("1%"): st.session_state.range_pct = 0.01
        if cols[1].button("2%"): st.session_state.range_pct = 0.02
        if cols[2].button("5%"): st.session_state.range_pct = 0.05
        if cols[3].button("10%"): st.session_state.range_pct = 0.10
    
    # Time Slider
    with c_p2:
        st.caption("Time Increment (Days)")
        time_step = st.slider("Days", 0, 60, 15, label_visibility="collapsed")
    
    # Vol Pills
    with c_p3:
        st.caption(f"Vol Adj: {st.session_state.vol_mod}%")
        vc = st.columns(3)
        if vc[0].button("-5%"): st.session_state.vol_mod -= 5
        if vc[1].button("Reset"): st.session_state.vol_mod = 0
        if vc[2].button("+5%"): st.session_state.vol_mod += 5

    # Charts & Matrix
    c_viz, c_mat = st.columns([2, 1])
    
    with c_viz:
        # P&L Calculation
        center = st.session_state.spot_price
        pct = st.session_state.range_pct
        prices = np.linspace(center*(1-pct), center*(1+pct), 100)
        
        # Scenarios: Today vs Future
        pnl_now, pnl_fut = [], []
        for p in prices:
            v0, vf = 0, 0
            for leg in st.session_state.legs:
                # Today
                px = get_bs_price(leg['Type'], p, leg['Strike'], leg['Expiry'], leg['Vol']+st.session_state.vol_mod, 4.0)
                v0 += (px - leg['Entry']) * leg['Qty'] * 100
                # Future
                tf = max(0, leg['Expiry'] - time_step)
                pxf = get_bs_price(leg['Type'], p, leg['Strike'], tf, leg['Vol']+st.session_state.vol_mod, 4.0)
                vf += (pxf - leg['Entry']) * leg['Qty'] * 100
            pnl_now.append(v0)
            pnl_fut.append(vf)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=pnl_now, name="Today", line=dict(color='#2980b9', width=3)))
        if time_step > 0:
            fig.add_trace(go.Scatter(x=prices, y=pnl_fut, name=f"T+{time_step}d", line=dict(color='#e67e22', dash='dash')))
        fig.add_vline(x=center, line_dash="dot", annotation_text="Spot")
        fig.update_layout(height=400, template="plotly_white", margin=dict(t=20, b=20, l=40, r=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with c_mat:
        # FIXED MATRIX (Real Dates)
        st.markdown("**Payoff Matrix**")
        
        # Calculate Future Dates
        today = datetime.now()
        dates_fut = [0, time_step, time_step*2]
        date_labels = [(today + timedelta(days=d)).strftime("%d %b") for d in dates_fut]
        
        # Matrix Rows (5 price points)
        m_prices = np.linspace(center*(1-pct), center*(1+pct), 6)
        m_data = []
        
        for p in m_prices:
            row = {"Price": f"${p:.2f}"}
            for i, d in enumerate(dates_fut):
                val = 0
                for leg in st.session_state.legs:
                    tf = max(0, leg['Expiry'] - d)
                    px = get_bs_price(leg['Type'], p, leg['Strike'], tf, leg['Vol']+st.session_state.vol_mod, 4.0)
                    val += (px - leg['Entry']) * leg['Qty'] * 100
                
                # Use Real Date as Header
                row[date_labels[i]] = val
            m_data.append(row)
            
        df_m = pd.DataFrame(m_data).set_index("Price")
        st.dataframe(df_m.style.background_gradient(cmap="RdYlGn", axis=None).format("${:,.0f}"), use_container_width=True, height=400)
