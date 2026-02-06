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

# --- CSS STYLING (Simplified & Robust) ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    
    /* Clean Header Box */
    .header-box {
        padding: 1.5rem;
        background-color: #0e1b32;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .header-title { font-size: 24px; font-weight: 700; margin: 0; }
    .header-sub { font-size: 14px; opacity: 0.8; margin: 0; }
    
    /* Native Button Overrides */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #15803d !important;
        border: none;
    }
    
    /* Status Badge */
    .status-tag {
        background-color: rgba(255,255,255,0.15);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-family: monospace;
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
if 'sheet_msg' not in st.session_state: st.session_state.sheet_msg = "Initializing..."
if 'data_source' not in st.session_state: st.session_state.data_source = "None"
if 'vol_manual' not in st.session_state: st.session_state.vol_manual = 33.5
if 'matrix_vol_mod' not in st.session_state: st.session_state.matrix_vol_mod = 0.0
if 'manual_spot' not in st.session_state: st.session_state.manual_spot = False

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=600)
def load_sheet(raw_url):
    try:
        if "/edit" in raw_url:
            base = raw_url.split("/edit")[0]
            csv_url = f"{base}/export?format=csv&gid=0"
        else:
            csv_url = raw_url

        df = pd.read_csv(csv_url, usecols=[2, 3, 4, 5, 6], on_bad_lines='skip', dtype=str)
        df.columns = ['Code', 'Ticker', 'Type', 'Expiry', 'Strike']
        
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

# --- MATH ENGINE (Robust Integer Days) ---
def get_bs_price(kind, spot, strike, time_days, vol_pct, rate_pct=4.0):
    try:
        # 1. Intrinsic Check (Expired)
        if time_days <= 0:
            return max(0.0, spot - strike) if kind == 'Call' else max(0.0, strike - spot)
        
        # 2. Minimum Time Safety (1 Day)
        # Mibian works best with integer days or safe floats. 
        # We ensure it's never 0 to avoid ZeroDivision errors deep in the library.
        safe_days = max(1.0, float(time_days))
        
        c = mibian.BS([spot, strike, rate_pct, safe_days], volatility=vol_pct)
        return c.callPrice if kind == 'Call' else c.putPrice
    except:
        return 0.0

def get_greeks(kind, spot, strike, time_days, vol_pct, rate_pct=4.0):
    try:
        if time_days <= 0: return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        safe_days = max(1.0, float(time_days))
        c = mibian.BS([spot, strike, rate_pct, safe_days], volatility=vol_pct)
        g = c.call if kind == 'Call' else c.put
        return {'delta': g.delta, 'gamma': g.gamma, 'theta': g.theta, 'vega': g.vega}
    except: return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

def solve_iv(price, spot, strike, time_days, rate_pct=4.0):
    try:
        if time_days <= 0: return None
        safe_days = max(1.0, float(time_days))
        c = mibian.BS([spot, strike, rate_pct, safe_days], callPrice=price)
        return c.impliedVolatility
    except: return None

def fetch_data(t):
    if st.session_state.manual_spot:
        return "MANUAL", st.session_state.spot_price, None

    clean = t.upper().replace(".AX", "").strip()
    sym = f"{clean}.AX"
    spot = 0.0
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="1d")
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
        else:
            return "ERROR", 0.0, None
        
        if tk.options: return "YAHOO", spot, tk
        else: return "SHEET", spot, None
    except: return "ERROR", 0.0, None

# --- 4. UI HEADER ---
status_parts = st.session_state.sheet_msg.split("|")
status_txt = status_parts[1] if len(status_parts) > 1 else status_parts[0]

# Use Streamlit Native Layout for robustness
with st.container():
    st.markdown(f"""
    <div class="header-box">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="header-title">TradersCircle <span style="font-weight: 300;">PRO</span></div>
                <div class="header-sub">Option Strategy Builder</div>
            </div>
            <div style="text-align: right;">
                <div class="header-title" style="color: #4ade80;">${st.session_state.spot_price:.2f}</div>
                <div class="header-sub">{st.session_state.ticker}</div>
                <span class="status-tag">{status_txt}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 5. CONTROLS ---
c1, c2, c3, c4 = st.columns([1, 1, 2, 1], gap="medium")

with c1:
    query = st.text_input("Ticker", st.session_state.ticker)

with c2:
    new_spot = st.number_input(
        "Spot Price ($)", 
        value=float(st.session_state.spot_price), 
        format="%.2f",
        step=0.01
    )
    if new_spot != st.session_state.spot_price:
        st.session_state.spot_price = new_spot
        st.session_state.manual_spot = True

with c3:
    st.session_state.vol_manual = st.slider("Implied Volatility (IV) %", 10.0, 100.0, st.session_state.vol_manual, 0.5)

with c4:
    st.write("") 
    st.write("")
    if st.button("Load Chain", type="primary", use_container_width=True) or (query.upper() != st.session_state.ticker):
        if query.upper() != st.session_state.ticker:
            st.session_state.manual_spot = False
        st.session_state.ticker = query.upper()
        
        with st.spinner("Fetching Data..."):
            source, px, obj = fetch_data(st.session_state.ticker)
            if not st.session_state.manual_spot:
                st.session_state.spot_price = px
            st.session_state.chain_obj = obj
            st.session_state.data_source = source
            data, msg = load_sheet(RAW_SHEET_URL)
            st.session_state.ref_data = data
            st.session_state.sheet_msg = msg
            st.rerun()

# --- 6. IV CALIBRATOR ---
with st.expander("🎯 IV Calibrator (Reverse Engineer)"):
    cc1, cc2, cc3, cc4 = st.columns(4)
    cal_strike = cc1.number_input("ATM Strike", value=round(st.session_state.spot_price,0))
    cal_days = cc2.number_input("Days to Exp", value=30)
    cal_price = cc3.number_input("Call Price", value=1.00)
    
    cc4.write("")
    cc4.write("")
    if cc4.button("Apply"):
        new_iv = solve_iv(cal_price, st.session_state.spot_price, cal_strike, cal_days)
        if new_iv:
            st.session_state.vol_manual = new_iv
            st.success(f"IV Updated: {new_iv:.2f}%")
            st.rerun()
        else:
            st.error("Calc Failed")

# --- 7. TABLE ---
df_view = pd.DataFrame()
current_exp = None

if st.session_state.ref_data is not None:
    ref = st.session_state.ref_data
    tkr = st.session_state.ticker.replace(".AX", "")
    subset = ref[ref['Ticker'] == tkr]
    
    # Filter Future Expiries Only
    today = datetime.now().replace(hour=0, minute=0, second=0)
    subset = subset[subset['Expiry'] >= today]
    
    if not subset.empty:
        valid_exps = sorted(subset['Expiry'].unique())
        exp_map = {d.strftime("%Y-%m-%d"): d for d in valid_exps}
        
        current_exp = st.selectbox("Expiry", list(exp_map.keys()))
        target_dt = exp_map[current_exp]
        
        # Simple Day Difference (Integer)
        days_diff = (target_dt - today).days
        
        day_chain = subset[subset['Expiry'] == target_dt]
        calls = day_chain[day_chain['Type'] == 'Call'].set_index('Strike')['Code']
        puts = day_chain[day_chain['Type'] == 'Put'].set_index('Strike')['Code']
        
        all_strikes = sorted(list(set(calls.index) | set(puts.index)))
        df_view = pd.DataFrame({'STRIKE': all_strikes})
        df_view['C_Code'] = df_view['STRIKE'].map(calls)
        df_view['P_Code'] = df_view['STRIKE'].map(puts)
        
        spot = st.session_state.spot_price
        vol = st.session_state.vol_manual
        
        # Calculations
        c_px = [get_bs_price('Call', spot, s, days_diff, vol) for s in df_view['STRIKE']]
        c_delta = [get_greeks('Call', spot, s, days_diff, vol)['delta'] for s in df_view['STRIKE']]
        p_px = [get_bs_price('Put', spot, s, days_diff, vol) for s in df_view['STRIKE']]
        p_delta = [get_greeks('Put', spot, s, days_diff, vol)['delta'] for s in df_view['STRIKE']]
            
        df_view['C_Price'] = c_px
        df_view['C_Delta'] = c_delta
        df_view['C_Vol'] = vol
        df_view['P_Price'] = p_px
        df_view['P_Delta'] = p_delta
        df_view['P_Vol'] = vol

if not df_view.empty and current_exp:
    center = st.session_state.spot_price
    if center > 0:
        df_view = df_view[(df_view['STRIKE'] > center*0.85) & (df_view['STRIKE'] < center*1.15)]
    
    st.markdown(f"**Chain: {current_exp}**")
    
    disp = df_view[['C_Code', 'C_Price', 'C_Vol', 'C_Delta', 'STRIKE', 'P_Price', 'P_Vol', 'P_Delta', 'P_Code']].copy()
    
    selection = st.dataframe(
        disp,
        column_config={
            "C_Code": st.column_config.TextColumn("Call Code"),
            "C_Price": st.column_config.NumberColumn("Theo", format="%.3f"),
            "C_Vol": st.column_config.NumberColumn("Vol", format="%.1f"),
            "C_Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
            "STRIKE": st.column_config.NumberColumn("Strike", format="%.2f"),
            "P_Price": st.column_config.NumberColumn("Theo", format="%.3f"),
            "P_Vol": st.column_config.NumberColumn("Vol", format="%.1f"),
            "P_Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
            "P_Code": st.column_config.TextColumn("Put Code"),
        },
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    if selection.selection['rows']:
        idx = selection.selection['rows'][0]
        row = disp.iloc[idx]
        # Days recalc for consistent logic
        days_diff = (datetime.strptime(current_exp, "%Y-%m-%d") - datetime.now()).days
        
        def add(side, kind, px, code_hint):
            st.session_state.legs.append({
                "Qty": 1 if side=="Buy" else -1, "Type": kind, 
                "Strike": row['STRIKE'], "Expiry": days_diff, 
                "Vol": st.session_state.vol_manual, 
                "Entry": px, "Code": code_hint,
                "Remove": False 
            })
            
        c_c = str(row['C_Code']) if pd.notna(row['C_Code']) else "N/A"
        p_c = str(row['P_Code']) if pd.notna(row['P_Code']) else "N/A"
        
        b1, b2, b3, b4 = st.columns(4)
        if b1.button(f"Buy Call {row['STRIKE']}"): add("Buy", "Call", row['C_Price'], c_c)
        if b2.button(f"Sell Call"): add("Sell", "Call", row['C_Price'], c_c)
        if b3.button(f"Buy Put {row['STRIKE']}"): add("Buy", "Put", row['P_Price'], p_c)
        if b4.button(f"Sell Put"): add("Sell", "Put", row['P_Price'], p_c)

# --- 8. PORTFOLIO & MATRIX ---
if st.session_state.legs:
    st.markdown("---")
    c_tick, c_port = st.columns([1, 2], gap="medium")
    
    with c_tick:
        st.subheader("Ticket")
        ticket_text = f"TICKET: {st.session_state.ticker} (Spot ${st.session_state.spot_price:.2f})\n"
        for leg in st.session_state.legs:
            direction = "Buy" if leg['Qty'] > 0 else "Sell"
            qty = abs(leg['Qty'])
            ticket_text += f"{direction} {qty}x {leg['Code']} ({leg['Type']} ${leg['Strike']}) @ ${leg['Entry']:.3f}\n"
        st.text_area("Copy", ticket_text, height=100)
        if st.button("Clear Portfolio", type="secondary"):
            st.session_state.legs = []
            st.rerun()

    with c_port:
        st.subheader("Legs")
        df_port = pd.DataFrame(st.session_state.legs)
        if 'Remove' not in df_port.columns: df_port['Remove'] = False
        
        edited_df = st.data_editor(
            df_port,
            column_config={
                "Remove": st.column_config.CheckboxColumn("Del", default=False, width="small"),
                "Code": st.column_config.TextColumn("Code"),
                "Entry": st.column_config.NumberColumn("Entry", format="$%.3f"),
                "Vol": st.column_config.NumberColumn("Vol", format="%.1f%%"),
                "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
            },
            use_container_width=True,
            num_rows="dynamic",
            key="port_edit"
        )
        
        if edited_df['Remove'].any():
            st.session_state.legs = edited_df[~edited_df['Remove']].drop(columns=['Remove']).to_dict('records')
            st.rerun()
        else:
            st.session_state.legs = edited_df.to_dict('records')

    # MATRIX
    st.markdown("---")
    st.subheader("Payoff Matrix")
    m1, m2 = st.columns(2)
    time_step = m1.slider("Step (Days)", 1, 30, 7)
    range_pct = m2.select_slider("Range", options=[0.02, 0.05, 0.10, 0.20], value=0.05, format_func=lambda x: f"{x*100:.0f}%")
    
    spot = st.session_state.spot_price
    prices = np.linspace(spot * (1 - range_pct), spot * (1 + range_pct), 12)
    dates = [0, time_step, time_step*2, time_step*3, time_step*4]
    
    matrix_data = []
    for p in prices:
        row = {"Price": p}
        for d in dates:
            pnl = 0
            for leg in st.session_state.legs:
                rem_days = max(0, leg['Expiry'] - d)
                exit_px = get_bs_price(leg['Type'], p, leg['Strike'], rem_days, leg['Vol'])
                pnl += (exit_px - leg['Entry']) * leg['Qty'] * 100
            
            col_name = (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
            if d == 0: col_name = f"Today ({col_name})"
            row[col_name] = pnl
        matrix_data.append(row)
        
    df_mx = pd.DataFrame(matrix_data).set_index("Price")
    def color_pnl(val):
        color = '#dcfce7' if val > 0 else '#fee2e2' if val < 0 else 'white'
        return f'background-color: {color}; color: black'
    st.dataframe(df_mx.style.applymap(color_pnl).format("${:,.0f}"), use_container_width=True)

    # CHART
    st.markdown("### Chart")
    chart_prices = np.linspace(spot * (1 - range_pct*1.5), spot * (1 + range_pct*1.5), 100)
    pnl_today = []
    pnl_expiry = []
    
    for p in chart_prices:
        val_t0 = 0
        val_tF = 0
        for leg in st.session_state.legs:
            # T+0
            price_t0 = get_bs_price(leg['Type'], p, leg['Strike'], leg['Expiry'], leg['Vol'])
            val_t0 += (price_t0 - leg['Entry']) * leg['Qty'] * 100
            # Expiry
            if leg['Type'] == 'Call': price_tf = max(0, p - leg['Strike'])
            else: price_tf = max(0, leg['Strike'] - p)
            val_tF += (price_tf - leg['Entry']) * leg['Qty'] * 100
        pnl_today.append(val_t0)
        pnl_expiry.append(val_tF)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_prices, y=pnl_today, name="Today", line=dict(color='#0e1b32', width=3)))
    fig.add_trace(go.Scatter(x=chart_prices, y=pnl_expiry, name="Expiry", line=dict(color='#15803d', dash='dash')))
    fig.add_vline(x=spot, line_dash="dot", line_color="grey")
    fig.update_layout(height=400, template="plotly_white", margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
