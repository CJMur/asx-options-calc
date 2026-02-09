# ==========================================
# TradersCircle Options Calculator
# VERSION: 5.6 (Clean Footer - Blank Zeroes)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import math

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="TradersCircle Options v5.6")
RAW_SHEET_URL = "https://docs.google.com/spreadsheets/d/1d9FQ5mn--MSNJ_WJkU--IvoSRU0gQBqE0f9s9zEb0Q4/edit?usp=sharing"

# --- CSS STYLING ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; padding-bottom: 5rem !important; }
    
    .header-box {
        padding: 1.5rem; background-color: #0e1b32; border-radius: 10px; color: white;
        margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title { font-size: 24px; font-weight: 700; margin: 0; }
    .header-sub { font-size: 14px; opacity: 0.8; margin: 0; }
    
    .status-tag {
        background-color: rgba(255,255,255,0.15); padding: 4px 10px; border-radius: 4px;
        font-size: 12px; font-family: monospace;
    }
    
    /* Button Overrides */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #15803d !important; border: none;
    }
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #f8fafc !important; color: #334155 !important; border: 1px solid #cbd5e1;
    }
    
    .stDataFrame { border: none !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "" 
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

# --- 4. MATH ENGINE ---
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes(S, K, T, r, sigma, option_type='Call'):
    try:
        if T <= 0: return max(0, S - K) if option_type == 'Call' else max(0, K - S)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == 'Call':
            return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    except: return 0.0

def calculate_delta(S, K, T, r, sigma, option_type='Call'):
    try:
        if T <= 0: return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return norm_cdf(d1) if option_type == 'Call' else norm_cdf(d1) - 1
    except: return 0.0

def get_bs_price(kind, spot, strike, time_days, vol_pct, rate_pct=4.0):
    T = max(0.001, time_days / 365.0)
    v = vol_pct / 100.0
    r = rate_pct / 100.0
    return black_scholes(float(spot), float(strike), T, r, v, kind)

def get_greeks(kind, spot, strike, time_days, vol_pct, rate_pct=4.0):
    T = max(0.001, time_days / 365.0)
    v = vol_pct / 100.0
    r = rate_pct / 100.0
    delta = calculate_delta(float(spot), float(strike), T, r, v, kind)
    return {'delta': delta}

# --- 5. SMART VOLATILITY ---
def calculate_smart_volatility(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        hist = tk.history(period="3mo")
        if hist.empty: return None
        
        hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
        std_vol = hist['Log_Ret'].std() * np.sqrt(252) * 100
        
        hl_ratio_sq = np.log(hist['High'] / hist['Low']) ** 2
        parkinson_vol = np.sqrt(1 / (4 * np.log(2)) * hl_ratio_sq.mean()) * np.sqrt(252) * 100
        
        final_vol = max(std_vol, parkinson_vol)
        return round(final_vol, 1)
    except: return None

def fetch_data(t):
    clean = t.upper().replace(".AX", "").strip()
    sym = f"{clean}.AX"
    
    auto_vol = calculate_smart_volatility(sym)
    if auto_vol and auto_vol > 0:
        st.session_state.vol_manual = auto_vol
        st.toast(f"✅ Volatility Set: {auto_vol}%")

    if st.session_state.manual_spot:
        return "MANUAL", st.session_state.spot_price, None

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

# --- 6. HEADER ---
status_parts = st.session_state.sheet_msg.split("|")
status_txt = status_parts[1] if len(status_parts) > 1 else status_parts[0]

with st.container():
    st.markdown(f"""
    <div class="header-box">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="header-title">TradersCircle <span style="font-weight: 300;">PRO</span></div>
                <div class="header-sub">Option Strategy Builder v5.6</div>
            </div>
            <div style="text-align: right;">
                <div class="header-title" style="color: #4ade80;">${st.session_state.spot_price:.2f}</div>
                <div class="header-sub">{st.session_state.ticker if st.session_state.ticker else "---"}</div>
                <span class="status-tag">{status_txt}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 7. CONTROLS ---
c1, c2, c3, c4 = st.columns([1, 1, 2, 1], gap="medium")
with c1: 
    query = st.text_input("Ticker", value=st.session_state.ticker, placeholder="Enter Stock Code")

with c2:
    new_spot = st.number_input("Spot Price ($)", value=float(st.session_state.spot_price), format="%.2f", step=0.01)
    if new_spot != st.session_state.spot_price:
        st.session_state.spot_price = new_spot
        st.session_state.manual_spot = True
with c3:
    st.session_state.vol_manual = st.slider("Volatility Estimate %", 10.0, 100.0, st.session_state.vol_manual, 0.5)
with c4:
    st.write("") 
    st.write("")
    if st.button("Load Chain", type="primary", use_container_width=True) or (query and query.upper() != st.session_state.ticker):
        if not query:
            st.warning("Please enter a ticker symbol.")
        else:
            if query.upper() != st.session_state.ticker: st.session_state.manual_spot = False
            st.session_state.ticker = query.upper()
            with st.spinner("Fetching Data..."):
                source, px, obj = fetch_data(st.session_state.ticker)
                if not st.session_state.manual_spot: st.session_state.spot_price = px
                st.session_state.chain_obj = obj
                st.session_state.data_source = source
                data, msg = load_sheet(RAW_SHEET_URL)
                st.session_state.ref_data = data
                st.session_state.sheet_msg = msg
                st.rerun()

# --- 9. TABLE DISPLAY ---
df_view = pd.DataFrame()
current_exp = None

if st.session_state.ref_data is not None and st.session_state.ticker:
    ref = st.session_state.ref_data
    tkr = st.session_state.ticker.replace(".AX", "")
    subset = ref[ref['Ticker'] == tkr]
    
    today = datetime.now().replace(hour=0, minute=0, second=0)
    subset = subset[subset['Expiry'] >= today]
    
    if not subset.empty:
        valid_exps = sorted(subset['Expiry'].unique())
        exp_map = {d.strftime("%Y-%m-%d"): d for d in valid_exps}
        
        current_exp = st.selectbox("Expiry", list(exp_map.keys()), index=None, placeholder="Select Expiry")
        
        if current_exp:
            target_dt = exp_map[current_exp]
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
            
            # Batch Calculation
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
        df_view['Diff'] = abs(df_view['STRIKE'] - center)
        atm_idx = df_view['Diff'].idxmin()
        start_idx = max(0, atm_idx - 12)
        end_idx = min(len(df_view), atm_idx + 13)
        df_view = df_view.iloc[start_idx:end_idx].drop(columns=['Diff'])
    
    st.markdown(f"**Chain: {current_exp}**")
    
    disp = df_view[['C_Code', 'C_Price', 'C_Vol', 'C_Delta', 'STRIKE', 'P_Price', 'P_Vol', 'P_Delta', 'P_Code']].copy()
    
    selection = st.dataframe(
        disp,
        column_config={
            "C_Code": st.column_config.TextColumn("Call Code"),
            "C_Price": st.column_config.NumberColumn("Fair Value", format="%.3f"),
            "C_Vol": st.column_config.NumberColumn("Hist Vol", format="%.1f"),
            "C_Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
            "STRIKE": st.column_config.NumberColumn("Strike", format="%.2f"),
            "P_Price": st.column_config.NumberColumn("Fair Value", format="%.3f"),
            "P_Vol": st.column_config.NumberColumn("Hist Vol", format="%.1f"),
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
        days_diff = (datetime.strptime(current_exp, "%Y-%m-%d") - datetime.now()).days
        
        q_col, _ = st.columns([1, 5])
        trade_qty = q_col.number_input("Trade Quantity", min_value=1, value=1, step=1)
        
        def add(side, kind, px, code_hint, delta_val, qty_val):
            final_qty = qty_val if side == "Buy" else -qty_val
            
            st.session_state.legs.append({
                "Qty": final_qty, 
                "Type": kind, 
                "Strike": row['STRIKE'], 
                "Expiry": days_diff, 
                "Vol": st.session_state.vol_manual, 
                "Entry": px, 
                "Code": code_hint,
                "Delta": delta_val,
                "Remove": False 
            })
            st.rerun()
            
        c_c = str(row['C_Code']) if pd.notna(row['C_Code']) else "N/A"
        p_c = str(row['P_Code']) if pd.notna(row['P_Code']) else "N/A"
        
        b1, b2, b3, b4, _ = st.columns([1, 1, 1, 1, 6]) 
        if b1.button(f"Buy Call"): add("Buy", "Call", row['C_Price'], c_c, row['C_Delta'], trade_qty)
        if b2.button(f"Sell Call"): add("Sell", "Call", row['C_Price'], c_c, row['C_Delta'], trade_qty)
        if b3.button(f"Buy Put"): add("Buy", "Put", row['P_Price'], p_c, row['P_Delta'], trade_qty)
        if b4.button(f"Sell Put"): add("Sell", "Put", row['P_Price'], p_c, row['P_Delta'], trade_qty)

# --- 10. PORTFOLIO & MATRIX ---
if st.session_state.legs:
    st.markdown("---")
    c_tick, c_port = st.columns([1, 2], gap="medium")
    with c_tick:
        st.subheader("Ticket")
        ticket_text = f"TICKET: {st.session_state.ticker} (Spot ${st.session_state.spot_price:.2f})\n"
        for leg in st.session_state.legs:
            ticket_text += f"{'Buy' if leg['Qty']>0 else 'Sell'} {abs(leg['Qty'])}x {leg['Code']} ({leg['Type']} ${leg['Strike']}) @ ${leg['Entry']:.3f}\n"
        st.text_area("Copy", ticket_text, height=100)
        if st.button("Clear Portfolio", type="secondary"):
            st.session_state.legs = []
            st.rerun()

    with c_port:
        st.subheader("Legs (Select row to delete)")
        
        # --- RECALC ---
        live_legs = []
        for leg in st.session_state.legs:
            new_theo = get_bs_price(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], st.session_state.vol_manual)
            new_greeks = get_greeks(leg['Type'], st.session_state.spot_price, leg['Strike'], leg['Expiry'], st.session_state.vol_manual)
            
            l = leg.copy()
            l['Theo (Unit)'] = new_theo
            l['Delta'] = new_greeks['delta']
            l['Net Delta'] = leg['Qty'] * new_greeks['delta'] * 100
            l['Premium'] = -(leg['Qty'] * leg['Entry'] * 100)
            live_legs.append(l)
            
        df_port = pd.DataFrame(live_legs)
        
        # --- TOTALS ROW ---
        if not df_port.empty:
            total_delta = df_port['Net Delta'].sum()
            total_premium = df_port['Premium'].sum()
            total_theo = (df_port['Qty'] * df_port['Theo (Unit)'] * 100).sum()
            
            total_row = pd.DataFrame([{
                'Qty': None, # Blank
                'Type': '', 
                'Strike': None, # Blank
                'Expiry': None, 
                'Entry': None, # Blank
                'Code': 'TOTAL', 
                'Theo (Unit)': total_theo, 
                'Net Delta': total_delta,
                'Premium': total_premium
            }])
            
            df_display = pd.concat([df_port, total_row], ignore_index=True)
        else:
            df_display = df_port

        # --- DISPLAY WITH DARK MODE COMPATIBLE STYLE ---
        column_config = {
            "Code": st.column_config.TextColumn("Code"),
            "Entry": st.column_config.NumberColumn("Entry", format="$%.3f"),
            "Theo (Unit)": st.column_config.NumberColumn("Theo Value", format="$%.2f"),
            "Net Delta": st.column_config.NumberColumn("Net Delta", format="%.2f"),
            "Premium": st.column_config.NumberColumn("Premium (Cash)", format="$%.2f"),
            "Qty": st.column_config.NumberColumn("Qty", format="%d"),
        }
        
        cols = ['Qty', 'Code', 'Type', 'Strike', 'Entry', 'Theo (Unit)', 'Net Delta', 'Premium']
        
        def highlight_total(row):
            if row['Code'] == 'TOTAL':
                # No background color = transparent (adapts to theme)
                # Just bold font and top border
                return ['font-weight: bold; border-top: 2px solid #888888;'] * len(row)
            return [''] * len(row)

        event = st.dataframe(
            df_display[cols].style.apply(highlight_total, axis=1).format({
                'Entry': '${:,.3f}', 'Theo (Unit)': '${:,.2f}', 
                'Net Delta': '{:,.2f}', 'Premium': '${:,.2f}'
            }),
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row"
        )
        
        if event.selection.rows:
            selected_indices = event.selection.rows
            valid_deletions = [i for i in selected_indices if i < len(st.session_state.legs)]
            if valid_deletions:
                current_legs = st.session_state.legs
                st.session_state.legs = [leg for i, leg in enumerate(current_legs) if i not in valid_deletions]
                st.rerun()

    # MATRIX
    st.markdown("---")
    st.subheader("Payoff Matrix")
    m1, m2 = st.columns(2)
    time_step = m1.slider("Step (Days)", 1, 30, 7)
    range_pct = m2.select_slider("Range", options=[0.02, 0.05, 0.10, 0.20], value=0.05, format_func=lambda x: f"{x*100:.0f}%")
    
    with m1:
        c_v1, c_v2, c_v3 = st.columns(3)
        if c_v1.button("-10%"): st.session_state.matrix_vol_mod -= 10
        if c_v2.button("Flat"): st.session_state.matrix_vol_mod = 0
        if c_v3.button("+10%"): st.session_state.matrix_vol_mod += 10
        st.caption(f"Vol Mod: {st.session_state.matrix_vol_mod:+}%")

    spot = st.session_state.spot_price
    prices = np.linspace(spot * (1 - range_pct), spot * (1 + range_pct), 12)
    dates = [0, time_step, time_step*2, time_step*3, time_step*4]
    
    matrix_data = []
    for p in prices:
        row = {"Price": p}
        for d in dates:
            pnl = 0
            for leg in st.session_state.legs:
                sim_vol = max(1.0, leg['Vol'] + st.session_state.matrix_vol_mod)
                rem_days = max(0, leg['Expiry'] - d)
                exit_px = get_bs_price(leg['Type'], p, leg['Strike'], rem_days, sim_vol)
                pnl += (exit_px - leg['Entry']) * leg['Qty'] * 100
            col_name = (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
            if d == 0: col_name = f"Today ({col_name})"
            row[col_name] = pnl
        matrix_data.append(row)
        
    df_mx = pd.DataFrame(matrix_data).set_index("Price")
    st.dataframe(df_mx.style.background_gradient(cmap='RdYlGn', axis=None).format("${:,.0f}"), use_container_width=True)

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
            price_tf = max(0, p - leg['Strike']) if leg['Type'] == 'Call' else max(0, leg['Strike'] - p)
            val_tF += (price_tf - leg['Entry']) * leg['Qty'] * 100
        pnl_today.append(val_t0)
        pnl_expiry.append(val_tF)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_prices, y=pnl_today, name="Today", line=dict(color='#0e1b32', width=3)))
    fig.add_trace(go.Scatter(x=chart_prices, y=pnl_expiry, name="Expiry", line=dict(color='#15803d', dash='dash')))
    fig.add_vline(x=spot, line_dash="dot", line_color="grey")
    fig.update_layout(height=400, template="plotly_white", margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
