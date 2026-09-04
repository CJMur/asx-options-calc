# ==========================================
# TradersCircle Options Calculator
# VERSION: 1.6.15 (ASX Dictionary Update)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, time, timedelta
import pytz
import math
import uuid
import requests
import re
import json
import base64
import io
import copy

# --- 1. CONFIGURATION & THEME ---
st.set_page_config(layout="wide", page_title="TradersCircle Options")

# REPLACE THIS WITH YOUR WORDPRESS DOMAIN:
WP_PORTFOLIO_API_URL = "https://portal.traderscircle.com.au/wp-json/tc-options/v1/portfolio"

# Direct GitHub Raw CDN URLs
OPTIONS_SHEET_URL = "https://raw.githubusercontent.com/CJMur/tc-options-data/main/options_data.parquet"
FWD_CURVE_URL = "https://raw.githubusercontent.com/CJMur/tc-options-data/main/fwd_curve.parquet"

# --- TOP ASX NAMES DICTIONARY ---
ASX_NAMES = {
    "A2M": "The a2 Milk Company", "AGL": "AGL Energy", "ALD": "Ampol Limited", "ALL": "Aristocrat Leisure",
    "AMC": "Amcor", "AMP": "AMP Limited", "ANN": "Ansell Limited", "ANZ": "ANZ Group Holdings",
    "APA": "APA Group", "ASX": "ASX Limited", "AZJ": "Aurizon Holdings", "BEN": "Bendigo and Adelaide Bank",
    "BGL": "Bellevue Gold", "BHP": "BHP Group", "BOQ": "Bank of Queensland", "BPT": "Beach Energy",
    "BSL": "BlueScope Steel", "BXB": "Brambles Limited", "CAR": "CAR Group Limited", "CBA": "Commonwealth Bank",
    "CGF": "Challenger Limited", "CMM": "Capricorn Metals Ltd", "COH": "Cochlear Limited", "COL": "Coles Group",
    "CPU": "Computershare", "CSL": "CSL Limited", "CTD": "Corporate Travel Management", "CWY": "Cleanaway Waste Management",
    "DMP": "Domino's Pizza Enterprises", "DNL": "Dyno Nobel Limited", "DRO": "DroneShield", "DXS": "Dexus",
    "EDV": "Endeavour Group", "EVN": "Evolution Mining", "FLT": "Flight Centre", "FMG": "Fortescue Metals",
    "FPH": "Fisher & Paykel Healthcare", "GMD": "Genesis Minerals Limited", "GMG": "Goodman Group",
    "GOLD": "Global X Physical Gold", "GPT": "GPT Group", "GYG": "Guzman y Gomez", "HUB": "HUB24",
    "HVN": "Harvey Norman", "IAG": "Insurance Australia", "IEL": "IDP Education", "IGO": "IGO Limited",
    "ILU": "Iluka Resources", "IVV": "iShares S&P 500 ETF", "JBH": "JB Hi-Fi", "JDO": "Judo Capital Holdings Limited",
    "JHX": "James Hardie", "LLC": "Lendlease Group", "LOV": "Lovisa Holdings", "LTR": "Liontown Limited",
    "LYC": "Lynas Rare Earths", "MGR": "Mirvac Group", "MIN": "Mineral Resources", "MPL": "Medibank Private",
    "MQG": "Macquarie Group", "MSB": "Mesoblast", "MTS": "Metcash", "NAB": "National Australia Bank",
    "NDQ": "BetaShares NASDAQ 100 ETF", "NEC": "Nine Entertainment Co.", "NHC": "New Hope Corporation",
    "NIC": "Nickel Industries Limited", "NST": "Northern Star Resources", "NWL": "Netwealth Group",
    "NXT": "NextDC Limited", "ORG": "Origin Energy", "ORI": "Orica Limited", "PDN": "Paladin Energy",
    "PLS": "Pilbara Minerals", "PME": "Pro Medicus Limited", "PNI": "Pinnacle Investment", "PRU": "Perseus Mining",
    "QAN": "Qantas Airways", "QBE": "QBE Insurance", "QUB": "Qube Holdings", "REA": "REA Group Ltd",
    "REH": "Reece Limited", "RHC": "Ramsay Health Care", "RIO": "Rio Tinto", "RMS": "Ramelius Resources Limited",
    "RRL": "Regis Resources", "S32": "South32 Limited", "SCG": "Scentre Group", "SDF": "Steadfast Group",
    "SEK": "Seek Limited", "SFR": "Sandfire Resources", "SGH": "Seven Group Holdings", "SGM": "Sims Limited",
    "SGP": "Stockland", "SHL": "Sonic Healthcare", "SIG": "Sigma Healthcare", "SOL": "Washington H. Soul Pattinson",
    "STO": "Santos Limited", "STW": "SPDR S&P/ASX 200 Fund", "SUN": "Suncorp Group", "TAH": "Tabcorp Holdings",
    "TCL": "Transurban Group", "TLC": "The Lottery Corporation", "TLS": "Telstra Group", "TLX": "Telix Pharmaceuticals",
    "TNE": "Technology One", "TPG": "TPG Telecom", "TWE": "Treasury Wine Estates", "VAU": "Vaneck Gold Bullion",
    "WBC": "Westpac Banking Corp", "WDS": "Woodside Energy Group", "WES": "Wesfarmers Limited",
    "WHC": "Whitehaven Coal", "WOR": "Worley Limited", "WOW": "Woolworths Group", "WTC": "WiseTech Global",
    "XJO": "S&P/ASX 200 Index", "XRO": "Xero Limited", "YAL": "Yancoal Australia", "ZIP": "Zip Co Limited"
}

# --- CSS STYLING ---
st.markdown("""
<style>
    div[class^="viewerBadge_container"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}

    .block-container { padding-top: 2rem !important; padding-bottom: 5rem !important; }
    
    .header-box {
        padding: 1.5rem; background-color: #0e1b32; border-radius: 10px; color: white;
        margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-bottom: 4px solid #1DBFD2;
    }
    .header-title { font-size: 24px; font-weight: 700; margin: 0; }
    .header-sub { font-size: 14px; opacity: 0.8; margin: 0; }
    .status-tag {
        background-color: rgba(255,255,255,0.15); padding: 4px 10px; border-radius: 4px;
        font-size: 12px; font-family: monospace;
    }
    
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #1DBFD2 !important; border: none; color: white !important; font-weight: bold;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #16aebf !important;
    }
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #f8fafc !important; color: #334155 !important; border: 1px solid #cbd5e1; font-weight: bold;
    }
    
    div[data-testid="stButton"] button[kind="tertiary"] {
        height: 39px !important;
        min-height: 39px !important;
        padding: 0 !important;
        margin-top: 1px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94a3b8 !important;
    }
    div[data-testid="stButton"] button[kind="tertiary"]:hover {
        color: #f87171 !important;
        background-color: rgba(248, 113, 113, 0.1) !important;
        border-color: transparent !important;
    }
    
    div[data-baseweb="slider"] > div > div > div { background-color: #0050FF !important; }
    div[role="slider"] { background-color: #0050FF !important; border: none !important; box-shadow: none !important; }
    div[data-testid="stSlider"] svg path { fill: #0050FF !important; stroke: #0050FF !important; }
    div[data-testid="stSlider"] p { color: var(--text-color) !important; }
    input[type=range] { accent-color: #0050FF !important; }
    
    [data-testid="stDataFrame"] [aria-selected="true"] > div {
        background-color: rgba(29, 191, 210, 0.4) !important;
        color: white !important;
    }
    
    .stDataFrame { border: none !important; }
    .trade-header {
        font-weight: 700; color: #94a3b8; font-size: 12px; text-transform: uppercase;
        margin-bottom: 5px; cursor: help; user-select: none;
    }
    
    .strategy-text { 
        user-select: none; 
        display: flex; 
        align-items: center; 
        height: 39px; 
        padding: 0 10px; 
        border-radius: 6px; 
        width: 100%;
        margin-top: 1px; 
        font-size: 14.5px;
    }
    
    div.row-widget.stRadio > div { flex-direction: row; align-items: center; }

    div[data-testid="stNumberInputStepUp"], 
    div[data-testid="stNumberInputStepDown"] {
        border: none !important;
        background-color: transparent !important;
    }
    div[data-baseweb="input"] {
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- TIMEZONE UTILITY ---
def get_sydney_time():
    return datetime.now(pytz.timezone('Australia/Sydney')).replace(tzinfo=None)

def format_date_ui(d_str):
    """Formats YYYY-MM-DD to DD-MMM-YYYY for UI display."""
    try:
        return datetime.strptime(d_str, "%Y-%m-%d").strftime("%d-%b-%Y")
    except:
        return d_str

# --- WORDPRESS REST API HELPER FUNCTIONS ---
def wp_fetch_portfolio(uid):
    """Fetches user portfolio from WordPress database."""
    if not uid: return []
    try:
        res = requests.get(f"{WP_PORTFOLIO_API_URL}?uid={uid}", timeout=5)
        res.raise_for_status() 
        return res.json().get('portfolio', [])
    except Exception as e:
        st.error(f"Failed to load from database: {e}")
    return []

def wp_save_portfolio(uid, portfolio):
    """Saves user portfolio to WordPress database silently."""
    if not uid: return
    try:
        payload = {"uid": uid, "portfolio": portfolio}
        res = requests.post(WP_PORTFOLIO_API_URL, json=payload, timeout=5)
        res.raise_for_status() 
    except Exception as e:
        st.error(f"Failed to save to database: {e}")

# --- 2. SESSION STATE & USER INITIALIZATION ---
if 'options_loaded' not in st.session_state: st.session_state.options_loaded = False
if 'portfolio' not in st.session_state: st.session_state.portfolio = []
if 'portfolio_last_refresh' not in st.session_state: st.session_state.portfolio_last_refresh = None
if 'last_upload_hash' not in st.session_state: st.session_state.last_upload_hash = None
if 'wp_loaded' not in st.session_state: st.session_state.wp_loaded = False
if 'trigger_db_save' not in st.session_state: st.session_state.trigger_db_save = False
if 'open_strat_id' not in st.session_state: st.session_state.open_strat_id = None
if 'is_monday_data' not in st.session_state: st.session_state.is_monday_data = False

# Detect WordPress User Token from URL
wp_uid = st.query_params.get("uid", None)

# Boot Load from WordPress Database
if wp_uid and not st.session_state.wp_loaded:
    st.session_state.portfolio = wp_fetch_portfolio(wp_uid)
    st.session_state.wp_loaded = True

# Navigation binding
if 'nav_view' not in st.session_state:
    st.session_state.nav_view = "🧮 Strategy Builder"

if 'fetch_time' not in st.session_state: st.session_state.fetch_time = get_sydney_time()
if 'url_loaded' not in st.session_state:
    st.session_state.url_loaded = True
    if "s" in st.query_params:
        try:
            encoded_state = st.query_params["s"]
            decoded_json = base64.urlsafe_b64decode(encoded_state.encode()).decode()
            payload = json.loads(decoded_json)
            st.session_state.ticker = payload.get("t", "XJO")
            if not st.session_state.ticker: st.session_state.ticker = "XJO"
            st.session_state.spot_price = payload.get("p", 0.0)
            st.session_state.manual_spot = payload.get("m", False)
            st.session_state.legs = payload.get("l", [])
            st.session_state.options_loaded = True
        except: pass

if 'legs' not in st.session_state: st.session_state.legs = [] 
if 'ticker' not in st.session_state: st.session_state.ticker = "XJO" 
if 'spot_price' not in st.session_state: st.session_state.spot_price = 0.0
if 'chain_obj' not in st.session_state: st.session_state.chain_obj = None
if 'ref_data' not in st.session_state: st.session_state.ref_data = None
if 'fwd_spreads' not in st.session_state: st.session_state.fwd_spreads = {}
if 'data_date' not in st.session_state: st.session_state.data_date = "Unknown"
if 'sheet_msg' not in st.session_state: st.session_state.sheet_msg = "Initializing..."
if 'manual_spot' not in st.session_state: st.session_state.manual_spot = False
if 'is_market_open' not in st.session_state: st.session_state.is_market_open = True
if 'div_info' not in st.session_state: st.session_state.div_info = None
if 'matrix_vol_mod' not in st.session_state: st.session_state.matrix_vol_mod = 0.0
if 'editor_reset' not in st.session_state: st.session_state.editor_reset = 0 

if 'preselect_code' not in st.session_state: st.session_state.preselect_code = None
if 'preselect_expiry' not in st.session_state: st.session_state.preselect_expiry = None
if 'preselect_strike' not in st.session_state: st.session_state.preselect_strike = None

TOOLTIPS = {
    "Theo": "The theoretical fair value of the option calculated using the Black-Scholes or Bjerksund-Stensland pricing model.",
    "IV": "Implied Volatility: The market's forecast of a likely movement in the security's price.",
    "Delta": "Fractional delta (per-contract directional exposure between -1.0 and +1.0).",
    "Strike": "The set price at which the option contract can be exercised.",
    "Code": "The unique ASX exchange ticker symbol for this specific option contract.",
    "Premium": "The total cost or credit for the trade. Calculated as Price × Quantity × Contract Multiplier.",
    "Margin": "The estimated portfolio collateral required to hold this specific strategy.",
    "Expected Margin": "The estimated portfolio collateral required to hold this specific strategy."
}

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=86400)
def fetch_rba_cash_rate():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get("https://www.rba.gov.au/", headers=headers, timeout=5)
        match = re.search(r'Cash rate target.*?(\d+\.\d+)\s*%', res.text, re.IGNORECASE | re.DOTALL)
        if match: return float(match.group(1))
        return 3.85
    except: return 3.85

global_rba_rate = fetch_rba_cash_rate()

def load_databases(opts_url, fwd_url, cb="default"):
    try:
        live_fwd_url = f"{fwd_url}?cb={cb}"
        live_opts_url = f"{opts_url}?cb={cb}"
        
        fwd_df = pd.DataFrame()
        spreads_dict = {}
        
        try:
            res_fwd = requests.get(live_fwd_url, timeout=10)
            if res_fwd.status_code == 200 and res_fwd.content.startswith(b'PAR1'):
                fwd_df = pd.read_parquet(io.BytesIO(res_fwd.content), engine='pyarrow')
        except: pass

        if not fwd_df.empty:
            fwd_df.columns = [str(c).strip() for c in fwd_df.columns]
            spread_col, fwd_date_col = None, None
            for col in fwd_df.columns:
                clean_c = str(col).lower()
                if clean_c == 'spread': spread_col = col
                elif clean_c in ['fwd expiry', 'expiry', 'xjo forward yield', 'forward yield']: fwd_date_col = col

            if spread_col and fwd_date_col:
                fwd_dates = pd.to_datetime(fwd_df[fwd_date_col], dayfirst=True, errors='coerce', format='mixed')
                fwd_vals = pd.to_numeric(fwd_df[spread_col].astype(str).str.replace(',', ''), errors='coerce')
                valid_mask = fwd_dates.notna() & fwd_vals.notna()
                for d, v in zip(fwd_dates[valid_mask], fwd_vals[valid_mask]):
                    spreads_dict[d.strftime("%Y-%m-%d")] = v

        df = pd.DataFrame()
        try:
            res_opts = requests.get(live_opts_url, timeout=15)
            if res_opts.status_code == 200:
                if res_opts.content.startswith(b'PAR1'):
                    df = pd.read_parquet(io.BytesIO(res_opts.content), engine='pyarrow')
                else:
                    return pd.DataFrame(), "error|GitHub blocked the request (Sent HTML instead of Parquet). Wait 1 minute and retry.", spreads_dict, "Unknown"
            else:
                return pd.DataFrame(), f"error|GitHub returned HTTP {res_opts.status_code}.", spreads_dict, "Unknown"
        except Exception as e:
            return pd.DataFrame(), f"error|Download failed: {str(e)}", spreads_dict, "Unknown"

        if df.empty:
            return pd.DataFrame(), "error|The Parquet file was downloaded but it is empty.", spreads_dict, "Unknown"

        df.columns = [str(c).strip() for c in df.columns]

        db_date = "Unknown"
        for col in df.columns:
            if str(col).lower() in ['busdate', 'bus date', 'date', 'businessdate']:
                valid_dates = df[col].dropna()
                if not valid_dates.empty:
                    db_date = str(valid_dates.iloc[0]).split(' ')[0].strip()
                break

        header_map = {
            'ASXCode': 'Code', 'Underlying': 'Ticker', 'OptType': 'Type', 
            'ExpDate': 'Expiry', 'Strike': 'Strike', 'Volatility': 'Vol', 
            'Settlement': 'Settlement', 'Style': 'Style', 'Lookup Key': 'LookupKey'
        }
        df = df.rename(columns=header_map)
        
        required = ['Code', 'Ticker', 'Strike', 'Expiry']
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            return pd.DataFrame(), f"error|Missing required columns: {missing_cols}.", spreads_dict, db_date

        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip().replace('NAN', np.nan).replace('', np.nan)
        df['Code'] = df['Code'].astype(str).str.upper().str.strip().replace('NAN', np.nan).replace('', np.nan)
        df = df.drop_duplicates(subset=['Code'], keep='last')

        if 'Type' in df.columns:
            raw_type = df['Type'].astype(str).str.strip().str.upper()
            df['Type'] = np.where(raw_type.str.startswith('C'), 'Call', 'Put')
        else: df['Type'] = 'Call'
            
        if 'Style' in df.columns:
            raw_style = df['Style'].astype(str).str.strip().str.upper()
            df['Style'] = np.where(raw_style.str.startswith('E'), 'European', 'American')
        else: df['Style'] = 'American'
            
        df['Strike'] = pd.to_numeric(df['Strike'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').round(3)
        df['Expiry'] = pd.to_datetime(df['Expiry'], dayfirst=True, errors='coerce', format='mixed').dt.normalize()
        
        if 'Vol' in df.columns:
            df['Vol'] = pd.to_numeric(df['Vol'].astype(str).str.replace('%', ''), errors='coerce')
            mask = df['Vol'] <= 1.0 
            df.loc[mask, 'Vol'] = df.loc[mask, 'Vol'] * 100
        else: df['Vol'] = 30.0
            
        df['Settlement'] = pd.to_numeric(df['Settlement'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce') if 'Settlement' in df.columns else 0.0

        scen_cols = [c for c in df.columns if 'Scenario' in str(c)]
        if scen_cols:
            for col in scen_cols:
                clean_str = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                df[col] = pd.to_numeric(clean_str, errors='coerce').fillna(0.0)
            df['UnitMargin'] = df[scen_cols].min(axis=1, skipna=True).fillna(0.0)
        else: df['UnitMargin'] = 0.0

        df = df.dropna(subset=['Code', 'Ticker', 'Strike', 'Expiry'])
        return df, f"success|{len(df)} Codes Loaded", spreads_dict, db_date
    except Exception as e:
        return pd.DataFrame(), f"error|Pipeline failed: {str(e)[:50]}", {}, "Error"

if st.session_state.ref_data is None:
    data, msg, extracted_spreads, d_date = load_databases(OPTIONS_SHEET_URL, FWD_CURVE_URL, "default")
    st.session_state.ref_data = data
    st.session_state.sheet_msg = msg
    st.session_state.fwd_spreads = extracted_spreads
    st.session_state.data_date = d_date
    try: 
        if d_date != "Unknown":
            st.session_state.is_monday_data = (pd.to_datetime(d_date).weekday() == 0)
    except: pass

# --- 4. MATH ENGINE ---
def norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_76_futures_model(S, F, K, T, r, v, kind):
    if T <= 0.0:
        price = max(0.0, S - K) if kind == 'Call' else max(0.0, K - S)
        delta = 1.0 if (kind == 'Call' and S > K) else (-1.0 if (kind == 'Put' and S < K) else 0.0)
        return price, delta

    d1 = (math.log(F / K) + 0.5 * v**2 * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)

    if kind == 'Call':
        price = math.exp(-r * T) * (F * norm_cdf(d1) - K * norm_cdf(d2))
        delta = (F / S) * math.exp(-r * T) * norm_cdf(d1)
    else:
        price = math.exp(-r * T) * (K * norm_cdf(-d2) - F * norm_cdf(-d1))
        delta = (F / S) * math.exp(-r * T) * (norm_cdf(d1) - 1.0)
    return price, delta

def black_scholes_european(S, K, T, r, sigma, option_type, q=0.0):
    if T <= 0: return max(0, S - K) if option_type == 'Call' else max(0, K - S)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'Call':
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)

def american_binomial_pricer(S, K, T, r, sigma, option_type, q=0.0, steps=100):
    if T <= 0:
        px = max(0, S - K) if option_type == 'Call' else max(0, K - S)
        delta = 1.0 if (option_type == 'Call' and S > K) else (-1.0 if (option_type == 'Put' and S < K) else 0.0)
        return px, delta
    
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    asset_prices = S * (u ** np.arange(steps, -1, -1)) * (d ** np.arange(0, steps + 1))
    
    values = np.maximum(0, asset_prices - K) if option_type == 'Call' else np.maximum(0, K - asset_prices)
    discount = math.exp(-r * dt)
    v_u, v_d = 0.0, 0.0
    
    for j in range(steps - 1, -1, -1):
        asset_prices = S * (u ** np.arange(j, -1, -1)) * (d ** np.arange(0, j + 1))
        values = discount * (p * values[:-1] + (1 - p) * values[1:])
        values = np.maximum(values, asset_prices - K) if option_type == 'Call' else np.maximum(values, K - asset_prices)
        if j == 1:
            v_u = values[0]
            v_d = values[1]
            
    delta = (v_u - v_d) / (S * u - S * d)
    return values[0], delta

def calculate_price_and_delta(ticker_symbol, style, kind, simulated_spot, strike, time_days, vol_pct, expiry_str_key, eval_date=None):
    if eval_date is None:
        eval_date = st.session_state.get('fetch_time', get_sydney_time())
        
    if simulated_spot <= 0 or strike <= 0: return 0.0, 0.0
    
    # --- STRICT EXPIRY KILL SWITCH ---
    try:
        exp_exact_dt = datetime.strptime(expiry_str_key, "%Y-%m-%d").replace(hour=16, minute=0)
        if eval_date >= exp_exact_dt:
            S, K = float(simulated_spot), float(strike)
            price = max(0.0, S - K) if kind == 'Call' else max(0.0, K - S)
            delta = 1.0 if (kind == 'Call' and S > K) else (-1.0 if (kind == 'Put' and S < K) else 0.0)
            return price, delta
    except:
        pass
        
    # --- STANDARD TIME FALLBACK ---
    if time_days <= 0.0:
        S, K = float(simulated_spot), float(strike)
        price = max(0.0, S - K) if kind == 'Call' else max(0.0, K - S)
        delta = 1.0 if (kind == 'Call' and S > K) else (-1.0 if (kind == 'Put' and S < K) else 0.0)
        return price, delta
    
    # --- THE MONDAY RULE ---
    # If the data file is dated Monday, it contains Friday's IV. Add 2.8 days to offset the weekend decay.
    if st.session_state.get('is_monday_data', False):
        time_days += 2.8
        
    r = global_rba_rate / 100.0
    q = 0.0
    is_xjo = (ticker_symbol == 'XJO')
    
    try:
        S, K = float(simulated_spot), float(strike)
        v = max(0.0001, vol_pct / 100.0)
        T = time_days / 365.0
        
        if is_xjo:
            style = 'EUROPEAN' 
            if expiry_str_key in st.session_state.fwd_spreads:
                basis_offset = st.session_state.fwd_spreads[expiry_str_key]
                return black_76_futures_model(S, S + basis_offset, K, T, r, v, kind)
            else: q = 0.04
        elif st.session_state.div_info:
            d_info = st.session_state.div_info
            if d_info['amount'] > 0 and d_info['date']:
                days_to_div = (d_info['date'] - eval_date).days
                if 0 <= days_to_div < time_days:
                    div_pv = d_info['amount'] * math.exp(-r * (days_to_div / 365.0))
                    S = max(0.01, S - div_pv)
        
        if style.upper() == 'EUROPEAN':
            price = black_scholes_european(S, K, T, r, v, kind, q)
            d1 = (math.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * math.sqrt(T))
            delta = math.exp(-q * T) * norm_cdf(d1) if kind == 'Call' else math.exp(-q * T) * (norm_cdf(d1) - 1)
        else:
            price, delta = american_binomial_pricer(S, K, T, r, v, kind, q, steps=100)
            
        return price, delta
    except: return 0.0, 0.0

def check_market_hours():
    now = get_sydney_time()
    return False if now.weekday() >= 5 else time(10, 0) <= now.time() <= time(16, 10)

st.session_state.is_market_open = check_market_hours()

def fetch_data(t):
    clean = t.upper().replace(".AX", "").strip()
    if clean == 'XJOW': clean = 'XJO'
    sym = "^AXJO" if clean == "XJO" else f"{clean}.AX"
    div_info, spot = None, 0.0
    
    if st.session_state.manual_spot:
        spot = st.session_state.spot_price
        try:
            info = yf.Ticker(sym).info
            if 'exDividendDate' in info and info['exDividendDate']:
                ex_ts = info['exDividendDate']
                if isinstance(ex_ts, (int, float)):
                    ex_date = datetime.fromtimestamp(ex_ts)
                    amt = info.get('lastDividendValue', info.get('dividendRate', 0)/2)
                    if ex_date > get_sydney_time(): div_info = {'amount': amt, 'date': ex_date}
        except: pass
        return "MANUAL", spot, div_info

    try:
        tk = yf.Ticker(sym)
        info = tk.info
        spot = float(info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose', 0.0))))
        if spot == 0.0:
            hist = tk.history(period="1d")
            if not hist.empty: spot = float(hist['Close'].iloc[-1])
            
        if 'exDividendDate' in info and info['exDividendDate']:
            ex_ts = info['exDividendDate']
            if isinstance(ex_ts, (int, float)):
                ex_date = datetime.fromtimestamp(ex_ts)
                if ex_date > get_sydney_time():
                    amt = info.get('lastDividendValue', 0)
                    if amt == 0: amt = info.get('dividendRate', 0) / 2
                    div_info = {'amount': amt, 'date': ex_date}

        if div_info is None and clean != 'XJO':
            try:
                divs = tk.dividends
                if not divs.empty:
                    now = get_sydney_time()
                    projected_divs = []
                    for d_date, amt in divs.items():
                        if amt > 0:
                            clean_date = pd.to_datetime(d_date).replace(tzinfo=None)
                            proj_date = clean_date + timedelta(days=364) 
                            if proj_date > now: projected_divs.append({'amount': float(amt), 'date': proj_date})
                    if projected_divs:
                        projected_divs.sort(key=lambda x: x['date'])
                        div_info = projected_divs[0]
            except: pass
        return "YAHOO", spot, div_info
    except: return "ERROR", 0.0, None

# --- HEADER ---
mkt_status = "🟢 OPEN" if st.session_state.is_market_open else "🔴 CLOSED"
date_status = f"📊 Data: {st.session_state.data_date}"
div_display_txt = f"💰 Auto Div: ${st.session_state.div_info['amount']:.2f} | {date_status}" if (st.session_state.div_info and st.session_state.ticker != 'XJO') else f"{date_status}"

st.markdown(f"""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="header-title">TradersCircle Options Calculator</div>
            <div class="header-sub">Option Strategy Builder v1.6.14</div>
        </div>
        <div style="text-align: right;">
            <div class="header-title" style="color: #4ade80;">${st.session_state.spot_price:.2f}</div>
            <div class="header-sub">{st.session_state.ticker if st.session_state.ticker else "---"}</div>
            <span class="status-tag">{mkt_status} | {div_display_txt}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if isinstance(st.session_state.sheet_msg, str) and st.session_state.sheet_msg.startswith("error|"):
    st.error(f"**Data Engine Warning:** {st.session_state.sheet_msg.split('|')[1]}")

# ==========================================
# 🗂️ PYTHON NAVIGATION ROUTER
# ==========================================

current_view = st.radio("Navigation", ["🧮 Strategy Builder", "💼 Portfolio Tracker"], horizontal=True, label_visibility="collapsed", key="nav_view")
st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

if current_view == "🧮 Strategy Builder":
    builder_needs_rerun = False
    tickers_list = sorted(st.session_state.ref_data['Ticker'].dropna().unique().tolist()) if (st.session_state.ref_data is not None and not st.session_state.ref_data.empty) else []

    c1, c2, c3, c4 = st.columns([1.4, 0.8, 0.7, 1.6], gap="medium")
    with c1: 
        asset_options = [f"{t} - {ASX_NAMES[t]}" if t in ASX_NAMES else t for t in tickers_list] or ["XJO - S&P/ASX 200 Index"]
        default_idx = tickers_list.index(st.session_state.ticker) if st.session_state.ticker in tickers_list else 0
        asset_sel = st.selectbox("Search Underlying Asset:", options=asset_options, index=default_idx)

    with c2: code_sel = st.text_input("Or Search Specific Code:", value=st.session_state.preselect_code if st.session_state.preselect_code else "")

    with c3:
        if st.session_state.ticker:
            new_spot = st.number_input("Spot Price ($)", value=float(st.session_state.spot_price), format="%.2f", step=0.01)
            if new_spot != st.session_state.spot_price:
                st.session_state.spot_price = new_spot
                st.session_state.manual_spot = True

    with c4:
        st.write(""); st.write("")
        bc1, bc2 = st.columns([2.5, 1.2]) 
        with bc2:
            if st.button("🔄 RESTART", use_container_width=True):
                st.query_params.clear() 
                saved_db, saved_fwd, saved_date = st.session_state.get('ref_data'), st.session_state.get('fwd_spreads'), st.session_state.get('data_date')
                saved_port, saved_refresh, saved_hash = st.session_state.get('portfolio'), st.session_state.get('portfolio_last_refresh'), st.session_state.get('last_upload_hash')
                st.session_state.clear() 
                st.session_state.ref_data, st.session_state.fwd_spreads, st.session_state.data_date = saved_db, saved_fwd, saved_date
                st.session_state.portfolio, st.session_state.portfolio_last_refresh, st.session_state.last_upload_hash = saved_port, saved_refresh, saved_hash
                st.session_state.options_loaded, st.session_state.wp_loaded = False, True
                st.rerun()

        with bc1: do_load = st.button("🔍 LOAD OPTIONS", type="primary", use_container_width=True)

    query = code_sel.strip() if code_sel.strip() else asset_sel.split(' - ')[0]
    trigger_search = do_load or (query and query.upper() != (st.session_state.preselect_code if st.session_state.preselect_code else st.session_state.ticker))

    if trigger_search:
        if not query: st.warning("Please select an asset or enter an option code.")
        else:
            query_upper = query.upper().strip()
            ref = st.session_state.ref_data
            ticker_to_fetch = query_upper
            
            if ref is not None and not ref.empty:
                match = ref[ref['Code'] == query_upper]
                if not match.empty:
                    ticker_to_fetch = str(match.iloc[0]['Ticker']).strip()
                    if ticker_to_fetch == 'XJOW': ticker_to_fetch = 'XJO'
                    st.session_state.preselect_expiry = match.iloc[0]['Expiry'].strftime("%Y-%m-%d")
                    st.session_state.preselect_strike = float(match.iloc[0]['Strike'])
                    st.session_state.preselect_code = query_upper
                else:
                    tickers = ref['Ticker'].unique()
                    possible_matches = [t for t in tickers if query_upper.startswith(t)]
                    if possible_matches:
                        best_match = max(possible_matches, key=len)
                        if len(query_upper) > len(best_match):
                            ticker_to_fetch = best_match
                            if ticker_to_fetch == 'XJOW': ticker_to_fetch = 'XJO'
                            st.session_state.preselect_code = query_upper
                            st.session_state.preselect_expiry = st.session_state.preselect_strike = None
                    else:
                        if ticker_to_fetch == 'XJOW': ticker_to_fetch = 'XJO'
                        st.session_state.preselect_expiry = st.session_state.preselect_strike = st.session_state.preselect_code = None

            if st.session_state.ticker != ticker_to_fetch:
                st.session_state.legs = []
                st.session_state.editor_reset += 1
            st.session_state.ticker = ticker_to_fetch

            with st.spinner("Fetching Fresh Market Data..."):
                source, px, div_data = fetch_data(st.session_state.ticker)
                if px > 0:
                    st.session_state.spot_price = px
                    st.session_state.manual_spot = False
                st.session_state.div_info, st.session_state.data_source = div_data, source
                st.session_state.fetch_time = get_sydney_time()
                data, msg, ext_spreads, d_date = load_databases(OPTIONS_SHEET_URL, FWD_CURVE_URL, str(uuid.uuid4())[:8])
                st.session_state.ref_data, st.session_state.sheet_msg, st.session_state.fwd_spreads, st.session_state.data_date = data, msg, ext_spreads, d_date
                try: 
                    if d_date != "Unknown":
                        st.session_state.is_monday_data = (pd.to_datetime(d_date).weekday() == 0)
                except: pass
                st.session_state.is_market_open, st.session_state.options_loaded = check_market_hours(), True
                st.rerun()

    if st.session_state.options_loaded or st.session_state.legs:
        df_view = pd.DataFrame()
        current_exp = None
        subset = pd.DataFrame()

        if st.session_state.ref_data is not None and not st.session_state.ref_data.empty and st.session_state.ticker:
            ref = st.session_state.ref_data
            tkr = st.session_state.ticker.replace(".AX", "")
            subset = ref[ref['Ticker'].isin(['XJO', 'XJOW'])] if tkr == 'XJO' else ref[ref['Ticker'] == tkr]
            subset = subset[subset['Expiry'] >= get_sydney_time().replace(hour=0, minute=0, second=0, microsecond=0)]
            
            if not subset.empty:
                valid_exps = sorted(subset['Expiry'].unique())
                exp_map = {d.strftime("%Y-%m-%d"): d for d in valid_exps}
                exp_list = list(exp_map.keys())
                default_idx = exp_list.index(st.session_state.preselect_expiry) if st.session_state.preselect_expiry in exp_list else None
                
                exp_col1, exp_col2 = st.columns([1, 2])
                with exp_col1: current_exp = st.selectbox("Expiry", exp_list, index=default_idx, placeholder="Select Expiry", format_func=format_date_ui)
                with exp_col2:
                    st.write("<div style='height: 29px;'></div>", unsafe_allow_html=True) 
                    view_mode = st.radio("Strikes View", options=["Standard View (30 Strikes)", "All Strikes"], horizontal=True, label_visibility="collapsed")
                
                if current_exp:
                    target_dt = exp_map[current_exp].replace(hour=16, minute=0)
                    days_diff_exact = (target_dt - st.session_state.get('fetch_time', get_sydney_time())).total_seconds() / 86400.0
                    day_chain = subset[subset['Expiry'] == exp_map[current_exp]].copy()
                    
                    def calc_row_metrics(row):
                        vol = float(row['Vol']) if pd.notna(row['Vol']) else 30.0
                        margin = float(row['UnitMargin']) if 'UnitMargin' in row else 0.0
                        px, delta = calculate_price_and_delta(st.session_state.ticker, row.get('Style', 'American'), row['Type'], st.session_state.spot_price, row['Strike'], days_diff_exact, vol, current_exp)
                        return pd.Series([px, delta, vol, margin])

                    metrics = day_chain.apply(calc_row_metrics, axis=1)
                    metrics.columns = ['Calc_Price', 'Calc_Delta', 'Calc_Vol', 'Calc_Margin']
                    day_chain = pd.concat([day_chain, metrics], axis=1)
                    
                    calls = day_chain[day_chain['Type'] == 'Call'].sort_values('Code', ascending=False).drop_duplicates(subset=['Strike']).set_index('Strike')
                    puts = day_chain[day_chain['Type'] == 'Put'].sort_values('Code', ascending=False).drop_duplicates(subset=['Strike']).set_index('Strike')
                    all_strikes = sorted(list(set(calls.index) | set(puts.index)))
                    
                    df_view = pd.DataFrame({'STRIKE': all_strikes})
                    df_view.insert(0, 'C_Buy', False); df_view.insert(1, 'C_Sell', False)
                    df_view['C_Code'] = df_view['STRIKE'].map(calls['Code'])
                    df_view['C_Style_Full'] = df_view['STRIKE'].map(calls['Style']).fillna('American')
                    df_view['C_Price'] = df_view['STRIKE'].map(calls['Calc_Price'])
                    df_view['C_Vol'] = df_view['STRIKE'].map(calls['Calc_Vol'])
                    df_view['C_Delta'] = df_view['STRIKE'].map(calls['Calc_Delta'])
                    df_view['C_Margin'] = df_view['STRIKE'].map(calls['Calc_Margin'])
                    
                    df_view['P_Code'] = df_view['STRIKE'].map(puts['Code'])
                    df_view['P_Style_Full'] = df_view['STRIKE'].map(puts['Style']).fillna('American')
                    df_view['P_Price'] = df_view['STRIKE'].map(puts['Calc_Price'])
                    df_view['P_Vol'] = df_view['STRIKE'].map(puts['Calc_Vol'])
                    df_view['P_Delta'] = df_view['STRIKE'].map(puts['Calc_Delta'])
                    df_view['P_Margin'] = df_view['STRIKE'].map(puts['Calc_Margin'])
                    df_view['P_Buy'] = False; df_view['P_Sell'] = False

        if not df_view.empty and current_exp:
            center = st.session_state.preselect_strike if (st.session_state.preselect_strike and current_exp == st.session_state.preselect_expiry) else st.session_state.spot_price
            radius = 15 if view_mode == "Standard View (30 Strikes)" else len(df_view)

            if center > 0 and radius < len(df_view):
                df_view['Diff'] = abs(df_view['STRIKE'] - center)
                atm_idx = df_view['Diff'].idxmin()
                df_view = df_view.iloc[max(0, atm_idx - radius):min(len(df_view), atm_idx + radius + 1)].drop(columns=['Diff'])
            
            st.markdown(f"**Chain: {format_date_ui(current_exp)}**")
            disp = df_view[['C_Buy', 'C_Sell', 'C_Code', 'C_Price', 'C_Vol', 'C_Delta', 'STRIKE', 'P_Price', 'P_Vol', 'P_Delta', 'P_Code', 'P_Buy', 'P_Sell']].copy()
            
            def highlight_itm(row):
                spot, strike, target_code = st.session_state.spot_price, row['STRIKE'], str(st.session_state.preselect_code)
                styles = []
                for col in row.index:
                    s = ""
                    if col in ['C_Buy', 'C_Sell', 'C_Code', 'C_Price', 'C_Vol', 'C_Delta'] and strike < spot: s += "background-color: rgba(128,128,128,0.15); "
                    elif col in ['P_Code', 'P_Price', 'P_Vol', 'P_Delta', 'P_Buy', 'P_Sell'] and strike > spot: s += "background-color: rgba(128,128,128,0.15); "
                    if col == 'STRIKE': s += "font-weight: bold; background-color: rgba(128,128,128,0.1); "
                    if col in ['C_Code', 'P_Code'] and str(row[col]) == target_code and target_code != "None": s += "color: white; border: 1px solid #1DBFD2; background-color: rgba(29, 191, 210, 0.4); "
                    styles.append(s)
                return styles

            styled_disp = disp.style.apply(highlight_itm, axis=1).format({
                'C_Price': '{:.3f}', 'C_Vol': '{:.1f}', 'C_Delta': '{:.3f}', 'STRIKE': '{:.3f}',
                'P_Price': '{:.3f}', 'P_Vol': '{:.1f}', 'P_Delta': '{:.3f}'
            })

            edited_df = st.data_editor(
                styled_disp,
                column_config={
                    "C_Buy": st.column_config.CheckboxColumn("☑ Buy", default=False),
                    "C_Sell": st.column_config.CheckboxColumn("☑ Sell", default=False),
                    "C_Code": st.column_config.TextColumn("Call Code", help=TOOLTIPS["Code"]),
                    "C_Price": st.column_config.NumberColumn("Theo", format="%.3f", help=TOOLTIPS["Theo"]),
                    "C_Vol": st.column_config.NumberColumn("IV %", format="%.1f", help=TOOLTIPS["IV"]),
                    "C_Delta": st.column_config.NumberColumn("Delta", format="%.3f", help="Pure option delta (probability)"),
                    "STRIKE": st.column_config.NumberColumn("Strike", format="%.3f", help=TOOLTIPS["Strike"]),
                    "P_Price": st.column_config.NumberColumn("Theo", format="%.3f", help=TOOLTIPS["Theo"]),
                    "P_Vol": st.column_config.NumberColumn("IV %", format="%.1f", help=TOOLTIPS["IV"]),
                    "P_Delta": st.column_config.NumberColumn("Delta", format="%.3f", help="Pure option delta (probability)"),
                    "P_Code": st.column_config.TextColumn("Put Code", help=TOOLTIPS["Code"]),
                    "P_Buy": st.column_config.CheckboxColumn("☑ Buy", default=False),
                    "P_Sell": st.column_config.CheckboxColumn("☑ Sell", default=False),
                },
                hide_index=True, use_container_width=True, key=f"chain_{current_exp}_{st.session_state.ticker}_{st.session_state.editor_reset}",
                disabled=["C_Code", "C_Price", "C_Vol", "C_Delta", "STRIKE", "P_Price", "P_Vol", "P_Delta", "P_Code"]
            )
            
            selected_legs, conflict = [], False
            for idx in range(len(edited_df)):
                row, edited_row = df_view.iloc[idx], edited_df.iloc[idx]
                c_buy, c_sell = edited_row.get('C_Buy', False), edited_row.get('C_Sell', False)
                p_buy, p_sell = edited_row.get('P_Buy', False), edited_row.get('P_Sell', False)
                
                if c_buy and c_sell: conflict = True
                elif c_buy: selected_legs.append({"side": "Buy", "kind": "Call", "row": row})
                elif c_sell: selected_legs.append({"side": "Sell", "kind": "Call", "row": row})
                    
                if p_buy and p_sell: conflict = True
                elif p_buy: selected_legs.append({"side": "Buy", "kind": "Put", "row": row})
                elif p_sell: selected_legs.append({"side": "Sell", "kind": "Put", "row": row})

            if conflict: st.error("⚠️ Conflict detected: You cannot select both 'Buy' and 'Sell' for the same exact option contract.")
            elif len(selected_legs) > 4: st.warning("⚠️ Please select a maximum of 4 legs at once.")
            elif len(selected_legs) > 0:
                st.write("")
                b_c1, b_c2, _ = st.columns([2.5, 1.5, 6], gap="small")
                with b_c1:
                    if st.button(f"+ Add {len(selected_legs)} Leg(s) to Builder", type="primary", use_container_width=True):
                        for leg in selected_legs:
                            r, kind = leg['row'], leg['kind']
                            px = r['C_Price'] if kind == 'Call' else r['P_Price']
                            code = r['C_Code'] if kind == 'Call' else r['P_Code']
                            delta = r['C_Delta'] if kind == 'Call' else r['P_Delta']
                            style = r['C_Style_Full'] if kind == 'Call' else r['P_Style_Full']
                            vol = r['C_Vol'] if kind == 'Call' else r['P_Vol']
                            margin = r['C_Margin'] if kind == 'Call' else r['P_Margin']
                                
                            st.session_state.legs.append({
                                "id": str(uuid.uuid4()), "Qty": 1 if leg['side'] == "Buy" else -1, 
                                "Type": kind, "Style": str(style), "Strike": float(r['STRIKE']), 
                                "ExpDateStr": current_exp, "Vol": float(vol), "Entry": float(px), 
                                "Code": str(code) if pd.notna(code) else "N/A", "Delta": float(delta), "MarginUnit": float(margin)
                            })
                        st.session_state.legs = sorted(st.session_state.legs, key=lambda x: float(x['Strike']), reverse=True)
                        st.session_state.editor_reset += 1; st.session_state.preselect_code = None; builder_needs_rerun = True
                with b_c2:
                    if st.button("Clear Selection", use_container_width=True):
                        st.session_state.editor_reset += 1; builder_needs_rerun = True

    if st.session_state.legs:
        st.markdown("---")
        st.subheader("Strategy")
        contract_multiplier = 10 if st.session_state.ticker == 'XJO' else 100
        
        h_col_spec = [0.8, 1.2, 0.6, 0.8, 1.5, 1.8, 1.0, 1.0, 1.0, 1.2, 1.3, 0.4]
        cols_header = st.columns(h_col_spec)
        headers = ["Qty", "Code", "Style", "Type", "Expiry", "Strike", "Vol", "Theo", "Delta", "Premium", "Expected Margin"]
        
        for col, h in zip(cols_header, headers):
            tooltip = TOOLTIPS.get(h, "")
            col.markdown(f'<div class="trade-header" title="{tooltip}">{h}</div>', unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 0 0 10px 0; border-top: 1px solid #334155;'>", unsafe_allow_html=True)

        scen_cols = [c for c in st.session_state.ref_data.columns if 'Scenario' in str(c)] if st.session_state.ref_data is not None else []
        leg_risk_arrays = []
        tkr = st.session_state.ticker.replace(".AX", "")
        
        for leg in st.session_state.legs:
            match = pd.DataFrame()
            if st.session_state.ref_data is not None and not st.session_state.ref_data.empty:
                ticker_mask = st.session_state.ref_data['Ticker'].isin(['XJO', 'XJOW']) if tkr == 'XJO' else st.session_state.ref_data['Ticker'] == tkr
                match = st.session_state.ref_data[ticker_mask & (st.session_state.ref_data['Type'] == leg['Type']) & (st.session_state.ref_data['Strike'] == float(leg['Strike'])) & (st.session_state.ref_data['Expiry'].dt.strftime("%Y-%m-%d") == leg['ExpDateStr'])]
            leg_risk_arrays.append(match.iloc[0][scen_cols].values.astype(float) if not match.empty and scen_cols else np.zeros(len(scen_cols) if scen_cols else 1))

        def compute_gross_margin(legs_list, arrays_list):
            if not legs_list: return 0.0
            subset_premium = sum(-(l['Qty'] * l['Entry'] * contract_multiplier) for l in legs_list)
            array_span_loss = abs(min(0.0, np.min(sum(r * l['Qty'] for r, l in zip(arrays_list, legs_list))))) if (len(arrays_list) > 0 and len(arrays_list[0]) > 1) else 0.0
            
            S = st.session_state.spot_price
            scan_pct = 0.065 if st.session_state.ticker == 'XJO' else 0.15
            test_spots = [S * (1 - scan_pct), S * (1 + scan_pct)]
            synthetic_pnls = [sum(l['Qty'] * ((max(0.0, spot - float(l['Strike'])) if l['Type'] == 'Call' else max(0.0, float(l['Strike']) - spot)) - l['Entry']) * contract_multiplier for l in legs_list) for spot in test_spots]
            
            unbounded_gross_risk = max(0.0, max(array_span_loss, abs(min(0.0, min(synthetic_pnls)))) + subset_premium)
            if (sum(l['Qty'] for l in legs_list if l['Type'] == 'Call') < 0) or (sum(l['Qty'] for l in legs_list if l['Type'] == 'Put') < 0): return unbounded_gross_risk
            
            strikes = [float(l['Strike']) for l in legs_list]
            if not strikes: return unbounded_gross_risk
            bound_pnls = [sum(l['Qty'] * max(0.0, spot - float(l['Strike'])) * contract_multiplier if l['Type'] == 'Call' else l['Qty'] * max(0.0, float(l['Strike']) - spot) * contract_multiplier for l in legs_list) for spot in strikes + [0.0, max(strikes) * 3.0]]
            return min(unbounded_gross_risk, abs(min(0.0, min(bound_pnls))))

        total_margin = compute_gross_margin(st.session_state.legs, leg_risk_arrays)
        total_premium, raw_theo_sum = 0, 0
        max_qty = max(abs(leg['Qty']) for leg in st.session_state.legs) if st.session_state.legs else 1
        
        for i, leg in enumerate(st.session_state.legs):
            if 'id' not in leg: leg['id'] = str(uuid.uuid4())
            if 'Style' not in leg: leg['Style'] = 'American'
            
            exp_dt = datetime.strptime(leg['ExpDateStr'], "%Y-%m-%d").replace(hour=16, minute=0)
            precise_days_diff = (exp_dt - st.session_state.get('fetch_time', get_sydney_time())).total_seconds() / 86400.0
            new_theo, new_delta = calculate_price_and_delta(st.session_state.ticker, leg['Style'], leg['Type'], st.session_state.spot_price, leg['Strike'], precise_days_diff, leg['Vol'], leg['ExpDateStr'])
            
            st.session_state.legs[i]['Entry'] = new_theo
            
            # Leg displays Fractional Delta (between -1.0 and 1.0) based on Buy/Sell direction
            fractional_delta = new_delta * (1 if leg['Qty'] >= 0 else -1)
            
            premium = -(leg['Qty'] * new_theo * contract_multiplier)
            row_margin = total_margin - compute_gross_margin(st.session_state.legs[:i] + st.session_state.legs[i+1:], leg_risk_arrays[:i] + leg_risk_arrays[i+1:])
            
            total_premium += premium; raw_theo_sum += leg['Qty'] * new_theo
            p_color = '#4ade80' if premium >= 0 else '#f87171'
            row_bg = "rgba(74, 222, 128, 0.10)" if leg['Qty'] > 0 else "rgba(248, 113, 113, 0.10)"
            
            premium_str = f"${premium:,.2f}" if premium >= 0 else f"-${abs(premium):,.2f}"
            margin_str = f"${row_margin:,.0f}" if row_margin >= 0 else f"-${abs(row_margin):,.0f}"
            
            c = st.columns(h_col_spec)
            with c[0]: 
                new_qty = st.number_input("Qty", value=int(leg['Qty']), step=1, key=f"qty_{leg['id']}", label_visibility="collapsed")
                if new_qty != leg['Qty']: st.session_state.legs[i]['Qty'] = new_qty; builder_needs_rerun = True
                    
            with c[1]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>{leg['Code']}</div>", unsafe_allow_html=True)
            with c[2]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>{str(leg['Style'])[0]}</div>", unsafe_allow_html=True)
            with c[3]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg}; font-weight:600;'>{leg['Type']}</div>", unsafe_allow_html=True)
            
            with c[4]: 
                tkr = st.session_state.ticker.replace(".AX", "")
                subset_st = pd.DataFrame()
                if st.session_state.ref_data is not None and not st.session_state.ref_data.empty:
                    ticker_mask = st.session_state.ref_data['Ticker'].isin(['XJO', 'XJOW']) if tkr == 'XJO' else st.session_state.ref_data['Ticker'] == tkr
                    subset_st = st.session_state.ref_data[ticker_mask & (st.session_state.ref_data['Type'] == leg['Type']) & (st.session_state.ref_data['Expiry'] >= get_sydney_time().replace(hour=0, minute=0, second=0, microsecond=0))]
                    
                valid_exps = sorted(subset_st['Expiry'].dropna().unique()) if not subset_st.empty else []
                exp_strs = [d.strftime("%Y-%m-%d") for d in valid_exps] or [leg['ExpDateStr']]
                exp_idx = exp_strs.index(leg['ExpDateStr']) if leg['ExpDateStr'] in exp_strs else 0
                new_exp = st.selectbox("Expiry", options=exp_strs, index=exp_idx, key=f"sb_exp_{leg['id']}", label_visibility="collapsed", format_func=format_date_ui)
                
                if new_exp != leg['ExpDateStr']:
                    st.session_state.legs[i]['ExpDateStr'] = new_exp
                    subset_exp = subset_st[subset_st['Expiry'].dt.strftime("%Y-%m-%d") == new_exp]
                    if not subset_exp.empty:
                        avail_stk = sorted(subset_exp['Strike'].unique().tolist())
                        closest = min(avail_stk, key=lambda x: abs(x - float(leg['Strike']))) if avail_stk else float(leg['Strike'])
                        st.session_state.legs[i]['Strike'] = float(closest)
                        match = subset_exp[subset_exp['Strike'] == closest].sort_values('Code', ascending=False)
                        if not match.empty:
                            st.session_state.legs[i]['Code'] = str(match.iloc[0]['Code'])
                            st.session_state.legs[i]['Vol'] = float(match.iloc[0]['Vol'])
                            st.session_state.legs[i]['Style'] = match.iloc[0].get('Style', 'American')
                            st.session_state.legs[i]['MarginUnit'] = float(match.iloc[0]['UnitMargin'])
                            rem_days = (datetime.strptime(new_exp, "%Y-%m-%d").replace(hour=16, minute=0) - st.session_state.get('fetch_time', get_sydney_time())).total_seconds() / 86400.0
                            new_entry, _ = calculate_price_and_delta(st.session_state.ticker, st.session_state.legs[i]['Style'], leg['Type'], st.session_state.spot_price, closest, rem_days, st.session_state.legs[i]['Vol'], new_exp)
                            st.session_state.legs[i]['Entry'] = new_entry
                    builder_needs_rerun = True
            
            with c[5]: 
                subset_st = pd.DataFrame()
                if st.session_state.ref_data is not None and not st.session_state.ref_data.empty:
                    ticker_mask = st.session_state.ref_data['Ticker'].isin(['XJO', 'XJOW']) if tkr == 'XJO' else st.session_state.ref_data['Ticker'] == tkr
                    subset_st = st.session_state.ref_data[ticker_mask & (st.session_state.ref_data['Type'] == leg['Type']) & (st.session_state.ref_data['Expiry'].dt.strftime("%Y-%m-%d") == leg['ExpDateStr'])]
                
                available_strikes = sorted(subset_st['Strike'].unique().tolist()) if not subset_st.empty else [float(leg['Strike'])]
                current_idx = available_strikes.index(float(leg['Strike'])) if float(leg['Strike']) in available_strikes else 0
                new_strike = st.selectbox("Strike", options=available_strikes, index=current_idx, key=f"stk_{leg['id']}", label_visibility="collapsed", format_func=lambda x: f"{x:.2f}")
                
                if new_strike != float(leg['Strike']):
                    st.session_state.legs[i]['Strike'] = new_strike
                    if not subset_st.empty:
                        match = subset_st[subset_st['Strike'] == new_strike].sort_values('Code', ascending=False)
                        if not match.empty:
                            new_vol = float(match.iloc[0]['Vol'])
                            new_style = match.iloc[0].get('Style', 'American')
                            st.session_state.legs[i]['Code'] = str(match.iloc[0]['Code'])
                            st.session_state.legs[i]['Vol'] = new_vol
                            st.session_state.legs[i]['Style'] = new_style
                            st.session_state.legs[i]['MarginUnit'] = float(match.iloc[0]['UnitMargin'])
                            matched_theo, _ = calculate_price_and_delta(st.session_state.ticker, new_style, leg['Type'], st.session_state.spot_price, new_strike, precise_days_diff, new_vol, leg['ExpDateStr'])
                            st.session_state.legs[i]['Entry'] = matched_theo
                    builder_needs_rerun = True
                    
            with c[6]: 
                new_vol_input = st.number_input("Vol", value=float(leg['Vol']), step=0.5, format="%.1f", key=f"vol_{leg['id']}", label_visibility="collapsed")
                if new_vol_input != leg['Vol']:
                    st.session_state.legs[i]['Vol'] = new_vol_input
                    calibrated_theo, _ = calculate_price_and_delta(st.session_state.ticker, leg['Style'], leg['Type'], st.session_state.spot_price, leg['Strike'], precise_days_diff, new_vol_input, leg['ExpDateStr'])
                    st.session_state.legs[i]['Entry'] = calibrated_theo
                    builder_needs_rerun = True
                    
            with c[7]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>{new_theo:.3f}</div>", unsafe_allow_html=True)
            with c[8]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>{fractional_delta:.3f}</div>", unsafe_allow_html=True)
            with c[9]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'><span style='color:{p_color}; font-weight:600;'>{premium_str}</span></div>", unsafe_allow_html=True)
            with c[10]: st.markdown(f"<div class='strategy-text' style='background-color:{row_bg};'><span style='font-weight:600;'>{margin_str}</span></div>", unsafe_allow_html=True)
            with c[11]:
                st.markdown("<div style='height: 1px;'></div>", unsafe_allow_html=True)
                if st.button("✕", key=f"d_{leg['id']}", type="tertiary", use_container_width=True):
                    st.session_state.legs.pop(i); builder_needs_rerun = True; break
                    
        st.markdown("<hr style='margin: -12px 0 8px 0; border-top: 1px solid #334155;'>", unsafe_allow_html=True)
        strategy_net_theo = raw_theo_sum / max_qty if max_qty != 0 else 0.0
        tot_prem_str = f"${total_premium:,.2f}" if total_premium >= 0 else f"-${abs(total_premium):,.2f}"
        tot_mar_str = f"${total_margin:,.2f}" if total_margin >= 0 else f"-${abs(total_margin):,.2f}"
        tot_p_color = '#4ade80' if total_premium >= 0 else '#f87171'

        with st.container():
            f = st.columns(h_col_spec)
            with f[1]: st.markdown("<div class='strategy-text' style='font-weight:bold;'>TOTAL STRATEGY</div>", unsafe_allow_html=True)
            with f[7]: st.markdown(f"<div class='strategy-text' style='font-weight:bold;'>{strategy_net_theo:.3f}</div>", unsafe_allow_html=True)
            
            # Leave the net delta column cleanly blank
            with f[8]: st.markdown(f"<div class='strategy-text'></div>", unsafe_allow_html=True)
            
            with f[9]: st.markdown(f"<div class='strategy-text'><span style='color:{tot_p_color}; font-weight:bold;'>{tot_prem_str}</span></div>", unsafe_allow_html=True)
            with f[10]: st.markdown(f"<div class='strategy-text'><span style='font-weight:bold;'>{tot_mar_str}</span></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("💾 Save to Portfolio")
        s_c1, s_c2 = st.columns([3, 1])
        with s_c1: strat_name = st.text_input("Strategy Name", value=f"{st.session_state.ticker} Option Strategy", label_visibility="collapsed")
        with s_c2:
            if st.button("Save Strategy", type="primary", use_container_width=True):
                st.session_state.portfolio.append({
                    "id": str(uuid.uuid4()), "name": strat_name, "ticker": st.session_state.ticker,
                    "spot_at_entry": st.session_state.spot_price, "legs": [leg.copy() for leg in st.session_state.legs]
                })
                st.session_state.trigger_db_save = True 
                st.success(f"Saved! Switch to the Portfolio Tracker tab to view it.")

        try:
            encoded_state = base64.urlsafe_b64encode(json.dumps({"t": st.session_state.ticker, "p": st.session_state.spot_price, "m": st.session_state.manual_spot, "l": st.session_state.legs}).encode()).decode()
            st.query_params["s"] = encoded_state
        except: pass

        # MATRIX
        st.markdown("---")
        st.subheader("Matrix")
        matrix_view = st.radio("Matrix Display Mode", ["Profit / Loss", "Theoretical Price"], horizontal=True)
        m1, m2 = st.columns([1, 1.2], gap="large")
        
        with m1:
            time_step = st.slider("Step (Days)", 1, 30, 1)
            vol_opts = ["IV -10%", "IV Flat", "IV +10%"]
            current_vol_idx = 0 if st.session_state.matrix_vol_mod == -10.0 else (2 if st.session_state.matrix_vol_mod == 10.0 else 1)
            vol_shift_sel = st.radio("Simulate Volatility Shift", vol_opts, index=current_vol_idx, horizontal=True)
            st.session_state.matrix_vol_mod = -10.0 if vol_shift_sel == "IV -10%" else (10.0 if vol_shift_sel == "IV +10%" else 0.0)

        with m2:
            slider_placeholder = st.empty()
            step_type = st.radio("Step Type", ["Percentage (%)", "Points/Dollars ($)"], horizontal=True)
            if step_type == "Percentage (%)":
                step_val = slider_placeholder.select_slider("Price Step", options=[x / 200.0 for x in range(1, 11)], value=0.01, format_func=lambda x: f"{x*100:.1f}%")
            else:
                pts_opts = [10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0] if st.session_state.spot_price > 1000 else ([1.0, 2.0, 5.0, 10.0, 20.0, 25.0] if st.session_state.spot_price > 100 else [0.10, 0.25, 0.50, 1.00, 2.00, 5.00])
                step_val = slider_placeholder.select_slider("Price Step", options=pts_opts, value=pts_opts[2], format_func=lambda x: f"{x:g}")

        spot = st.session_state.spot_price
        prices = [spot * (1 + step_val * i) for i in range(6, -7, -1)] if step_type == "Percentage (%)" else [spot + (step_val * i) for i in range(6, -7, -1)]
        chart_prices = np.linspace(spot * (1 - step_val * 8), spot * (1 + step_val * 8), 200) if step_type == "Percentage (%)" else np.linspace(spot - step_val * 8, spot + step_val * 8, 200)
        dates = [d * time_step for d in range(7)]
        
        matrix_data = []
        for p in prices:
            row_label = f"» ${p:.2f} (SPOT) «" if math.isclose(p, spot, rel_tol=1e-5) else f"${p:.2f}"
            row = {"Price": row_label}
            for d in dates:
                pnl, net_theo_sum = 0, 0
                eval_dt = st.session_state.get('fetch_time', get_sydney_time()) + timedelta(days=d)
                for leg in st.session_state.legs:
                    exp_exact_dt = datetime.strptime(leg['ExpDateStr'], "%Y-%m-%d").replace(hour=16, minute=0)
                    if eval_dt.date() >= exp_exact_dt.date():
                        active_eval_dt = exp_exact_dt
                        rem_days = 0.0
                    else:
                        active_eval_dt = eval_dt
                        rem_days = (exp_exact_dt - active_eval_dt).total_seconds() / 86400.0
                        
                    exit_px, _ = calculate_price_and_delta(st.session_state.ticker, leg['Style'], leg['Type'], p, leg['Strike'], rem_days, max(1.0, leg['Vol'] + st.session_state.matrix_vol_mod), leg['ExpDateStr'], eval_date=active_eval_dt)
                    pnl += (exit_px - leg['Entry']) * leg['Qty'] * contract_multiplier
                    net_theo_sum += exit_px * leg['Qty']
                col_name = eval_dt.strftime("%d-%b-%Y")
                row[f"Today ({col_name})" if d == 0 else col_name] = pnl if matrix_view == "Profit / Loss" else (net_theo_sum / max_qty if max_qty != 0 else 0.0)
            matrix_data.append(row)
            
        df_mx = pd.DataFrame(matrix_data).set_index("Price")
        capital_at_risk = max(total_margin, abs(total_premium)) if max(total_margin, abs(total_premium)) > 0 else 1.0
        
        def format_pnl(val):
            try: return f"${float(val):,.0f} ({'+' if float(val)>0 else ''}{(float(val)/capital_at_risk)*100:.1f}%)" if pd.notna(val) else ""
            except: return ""

        def make_heatmap(df):
            abs_max = max(abs(df.max().max()), abs(df.min().min()), 1)
            styles_df = pd.DataFrame('', index=df.index, columns=df.columns)
            for idx in df.index:
                is_spot = "SPOT" in str(idx)
                for col in df.columns:
                    val = df.loc[idx, col]
                    s = f"background-color: rgba(74, 222, 128, {min(val/abs_max,1.0)*0.35+0.05:.2f}); " if val > 0 else (f"background-color: rgba(248, 113, 113, {min(abs(val)/abs_max,1.0)*0.35+0.05:.2f}); " if val < 0 else "")
                    if is_spot: 
                        s += "font-weight: bold; border-top: 2px solid rgba(255,255,255,0.5); border-bottom: 2px solid rgba(255,255,255,0.5);"
                    styles_df.loc[idx, col] = s
            return styles_df
            
        def highlight_spot(df):
            styles_df = pd.DataFrame('', index=df.index, columns=df.columns)
            for idx in df.index:
                if "SPOT" in str(idx):
                    styles_df.loc[idx, :] = "font-weight: bold; background-color: rgba(255,255,255,0.05); border-top: 2px solid rgba(255,255,255,0.5); border-bottom: 2px solid rgba(255,255,255,0.5);"
            return styles_df

        if matrix_view == "Profit / Loss":
            st.dataframe(df_mx.style.apply(make_heatmap, axis=None).format(format_pnl).set_table_styles([
                {'selector': 'th', 'props': [('color', 'var(--text-color)'), ('font-weight', 'bold')]}
            ]), use_container_width=True, height=500)
        else:
            format_dict = {col: "{:.3f}" for col in df_mx.columns}
            st.dataframe(df_mx.style.apply(highlight_spot, axis=None).format(format_dict).set_table_styles([
                {'selector': 'th', 'props': [('color', 'var(--text-color)'), ('font-weight', 'bold')]}
            ]), use_container_width=True, height=500)

        # PAYOFF CHART
        st.markdown("### Payoff Chart")
        pnl_today, pnl_expiry = [], []
        for p in chart_prices:
            val_t0, val_tF = 0, 0
            for leg in st.session_state.legs:
                rem_days = ((datetime.strptime(leg['ExpDateStr'], "%Y-%m-%d").replace(hour=16, minute=0)) - st.session_state.get('fetch_time', get_sydney_time())).total_seconds() / 86400.0
                price_t0, _ = calculate_price_and_delta(st.session_state.ticker, leg['Style'], leg['Type'], p, leg['Strike'], rem_days, leg['Vol'], leg['ExpDateStr'])
                val_t0 += (price_t0 - leg['Entry']) * leg['Qty'] * contract_multiplier
                val_tF += ((max(0, p - leg['Strike']) if leg['Type'] == 'Call' else max(0, leg['Strike'] - p)) - leg['Entry']) * leg['Qty'] * contract_multiplier
            pnl_today.append(val_t0); pnl_expiry.append(val_tF)
            
        breakevens = [chart_prices[i] - pnl_expiry[i] * (chart_prices[i+1] - chart_prices[i]) / (pnl_expiry[i+1] - pnl_expiry[i]) for i in range(len(chart_prices)-1) if pnl_expiry[i] * pnl_expiry[i+1] < 0]
            
        fig = go.Figure()
        fig.add_hrect(y0=0, y1=1e6, fillcolor="rgba(74, 222, 128, 0.08)", layer="below", line_width=0)
        fig.add_hrect(y0=-1e6, y1=0, fillcolor="rgba(248, 113, 113, 0.08)", layer="below", line_width=0)
        
        for be in breakevens:
            fig.add_vline(x=be, line_dash="dot", line_color="#10b981", opacity=0.8)
            fig.add_annotation(x=be, y=0, text=f"BE: ${be:.2f}", showarrow=True, arrowhead=2, arrowcolor="#10b981", ax=0, ay=-40, bgcolor="#0f172a", bordercolor="#10b981", font=dict(color="#10b981", size=11))
        
        fig.add_trace(go.Scatter(x=chart_prices, y=pnl_today, name="Today", line=dict(color='#0050FF', width=3)))
        fig.add_trace(go.Scatter(x=chart_prices, y=pnl_expiry, name="Expiry", line=dict(color='#1DBFD2', dash='dash', width=3)))
        fig.add_vline(x=spot, line_dash="dot", line_color="grey")
        
        padding = max(abs(max(max(pnl_expiry), max(pnl_today))), abs(min(min(pnl_expiry), min(pnl_today)))) * 0.1
        fig.update_layout(height=450, template="plotly_white", margin=dict(t=30, b=30), xaxis=dict(title="Stock Price @ Expiry", tickprefix="$"), yaxis=dict(title="Profit / Loss ($)", tickprefix="$", zeroline=True, zerolinewidth=2, zerolinecolor='black', range=[min(min(pnl_expiry), min(pnl_today)) - padding, max(max(pnl_expiry), max(pnl_today)) + padding]), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        if builder_needs_rerun: st.rerun()

elif current_view == "💼 Portfolio Tracker":
    
    # CALLBACK HANDLERS
    def set_active_strat(s_id): st.session_state.open_strat_id = s_id

    def on_spot_override_change(s_id, k):
        val = st.session_state[k]
        for s in st.session_state.portfolio:
            if s['id'] == s_id: s['override_spot'] = float(val) if val else None; break
        st.session_state.trigger_db_save = True; st.session_state.open_strat_id = s_id

    def on_name_change(s_id, k):
        val = st.session_state[k]
        for s in st.session_state.portfolio:
            if s['id'] == s_id: s['name'] = val; break
        st.session_state.trigger_db_save = True; st.session_state.open_strat_id = s_id

    def on_net_entry_change(s_id, k, mq, old_val):
        val = st.session_state[k]
        if val is not None and not math.isclose(val, old_val, abs_tol=1e-5) and mq != 0:
            tot_change = (val - old_val) * mq
            for s in st.session_state.portfolio:
                if s['id'] == s_id:
                    if len(s['legs']) > 0:
                        c_per_leg = tot_change / len(s['legs'])
                        for l in s['legs']:
                            if l['Qty'] != 0: l['Entry'] += c_per_leg / l['Qty']
                    break
        st.session_state.trigger_db_save = True; st.session_state.open_strat_id = s_id

    st.markdown("### Saved Strategies")
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.2, 1, 1])
    
    with ctrl_c1:
        if st.session_state.portfolio:
            if st.button("🔄 Refresh All Prices", type="primary", use_container_width=True):
                with st.spinner("Fetching live market data and updating Volatility..."):
                    orig_manual = st.session_state.manual_spot
                    st.session_state.manual_spot = False
                    st.session_state.portfolio_last_refresh = get_sydney_time()
                    data, msg, ext_spreads, d_date = load_databases(OPTIONS_SHEET_URL, FWD_CURVE_URL, str(uuid.uuid4())[:8])
                    st.session_state.ref_data, st.session_state.sheet_msg, st.session_state.fwd_spreads, st.session_state.data_date = data, msg, ext_spreads, d_date
                    try: 
                        if d_date != "Unknown":
                            st.session_state.is_monday_data = (pd.to_datetime(d_date).weekday() == 0)
                    except: pass
                    
                    for strat in st.session_state.portfolio:
                        _, spot, _ = fetch_data(strat.get('ticker', 'XJO'))
                        strat['current_spot'] = spot if spot > 0 else strat.get('spot_at_entry', 0.0)
                        if 'override_spot' in strat: strat['override_spot'] = None 
                        ui_ovr_key = f"ui_ovr_spot_{strat['id']}"
                        if ui_ovr_key in st.session_state: del st.session_state[ui_ovr_key]
                        
                        for leg in strat['legs']:
                            if not data.empty:
                                match = data[data['Code'] == leg['Code']]
                                if not match.empty: 
                                    live_v = float(match.iloc[0]['Vol'])
                                    leg['Current_Vol'] = live_v
                                    leg['Vol'] = live_v
                            
                    st.session_state.manual_spot = orig_manual
                    st.session_state.trigger_db_save = True
                    st.rerun()
        else: st.button("🔄 Refresh All Prices", type="primary", use_container_width=True, disabled=True)
            
    with ctrl_c2:
        if st.session_state.portfolio:
            flat_port = [{"StrategyID": s['id'], "StrategyName": s['name'], "Ticker": s.get('ticker', 'Unknown'), "EntrySpot": s.get('spot_at_entry', 0.0), "LegID": l['id'], "Qty": l['Qty'], "Type": l['Type'], "Style": l['Style'], "Strike": l['Strike'], "Expiry": l['ExpDateStr'], "Vol": l['Vol'], "EntryPrice": l['Entry'], "Code": l['Code']} for s in st.session_state.portfolio for l in s['legs']]
            st.download_button("💾 Download Backup (CSV)", data=pd.DataFrame(flat_port).to_csv(index=False), file_name="tc_portfolio.csv", mime="text/csv", use_container_width=True)
        else: st.button("💾 Download Backup (CSV)", disabled=True, use_container_width=True)
            
    with ctrl_c3:
        uploaded_file = st.file_uploader("Upload", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            file_hash = hash(uploaded_file.getvalue())
            if st.session_state.last_upload_hash != file_hash:
                try:
                    df_up = pd.read_csv(uploaded_file)
                    new_port = []
                    for strat_id, group in df_up.groupby('StrategyID'):
                        new_port.append({
                            "id": str(strat_id), "name": str(group['StrategyName'].iloc[0]),
                            "ticker": str(group['Ticker'].iloc[0] if 'Ticker' in group.columns else 'Unknown'),
                            "spot_at_entry": float(group['EntrySpot'].iloc[0] if 'EntrySpot' in group.columns else 0.0),
                            "legs": [{"id": str(r['LegID']), "Qty": int(r['Qty']), "Type": str(r['Type']), "Style": str(r['Style']), "Strike": float(r['Strike']), "ExpDateStr": str(r['Expiry']), "Vol": float(r['Vol']), "Entry": float(r['EntryPrice']), "Code": str(r['Code']), "Delta": 0.0, "MarginUnit": 0.0} for _, r in group.iterrows()]
                        })
                    st.session_state.portfolio = new_port
                    st.session_state.portfolio_last_refresh = None
                    st.session_state.last_upload_hash = file_hash
                    st.session_state.trigger_db_save = True
                    st.success("Portfolio Loaded! Click Refresh to see live values.")
                except Exception as e: st.error(f"Error loading file: {e}")

    if st.session_state.portfolio_last_refresh:
        st.info(f"⏱️ **Live Snapshot Taken:** {st.session_state.portfolio_last_refresh.strftime('%d %b %Y, %I:%M %p AEST')}")
    elif st.session_state.portfolio:
        st.info("ℹ️ Click 'Refresh All Prices' to load current market data and calculate your Open P&L.")

    st.markdown("---")
    
    # Portfolio Display Engine
    for i, strat in enumerate(st.session_state.portfolio):
        port_needs_rerun = False
        ticker_display = strat.get('ticker', 'Unknown')
        ui_ovr_key = f"ui_ovr_spot_{strat['id']}"
        
        override_val = strat.get('override_spot', None)
        if override_val is not None:
            override_val = float(override_val)
            
        current_spot_val = override_val if override_val is not None else float(strat.get('current_spot', strat.get('spot_at_entry', 0.0)))
        
        max_qty = max([abs(leg['Qty']) for leg in strat['legs']]) if strat['legs'] else 1
        net_entry_theo = (sum([leg['Qty'] * leg['Entry'] for leg in strat['legs']])) / max_qty if max_qty != 0 else 0.0
        contract_multiplier = 10 if ticker_display == 'XJO' else 100
        
        strat_pnl, display_legs, net_live_theo_sum = 0.0, [], 0.0
        ref_time = st.session_state.get('portfolio_last_refresh') or get_sydney_time()
        
        for leg in strat['legs']:
            rem_days = ((datetime.strptime(leg['ExpDateStr'], "%Y-%m-%d").replace(hour=16, minute=0)) - ref_time).total_seconds() / 86400.0
            cur_theo, _ = calculate_price_and_delta(ticker_display, leg['Style'], leg['Type'], current_spot_val, leg['Strike'], rem_days, leg.get('Current_Vol', leg['Vol']), leg['ExpDateStr'], eval_date=ref_time)
            
            strat_pnl += (cur_theo - leg['Entry']) * leg['Qty'] * contract_multiplier
            net_live_theo_sum += cur_theo * leg['Qty']
            live_premium = -(leg['Qty'] * cur_theo * contract_multiplier)
            
            display_legs.append({
                "Code": leg['Code'], "Action": "Buy" if leg['Qty'] > 0 else "Sell", "Qty": abs(leg['Qty']), "Type": leg['Type'],
                "Strike": f"${leg['Strike']:.2f}", "Expiry": format_date_ui(leg['ExpDateStr']), "Entry Theo": f"{leg['Entry']:.3f}",
                "Live Theo": f"{cur_theo:.3f}", "Premium": f"${live_premium:,.2f}" if live_premium >= 0 else f"-${abs(live_premium):,.2f}", "Raw_Premium": live_premium
            })
            
        net_live_theo = net_live_theo_sum / max_qty if max_qty != 0 else 0.0
        pnl_str = f" | Spot: :green[${current_spot_val:.2f}] | {'🟢' if strat_pnl >= 0 else '🔴'} Open P&L: :{'green' if strat_pnl >= 0 else 'red'}[{'+' if strat_pnl >= 0 else ''}${strat_pnl:,.2f}]"
            
        with st.expander(f"📁 **{strat.get('name', 'Strategy')}** ({ticker_display}){pnl_str}", expanded=(st.session_state.get('open_strat_id') == strat['id'])):
            
            c_head0, c_head1, c_head2, c_head3, c_head4 = st.columns([1.5, 1, 1, 1, 1.2])
            with c_head0:
                st.markdown(f"**Strategy Name:**")
                ui_name_key = f"rename_{strat['id']}"
                st.text_input("Name", value=strat.get('name', 'Strategy'), key=ui_name_key, label_visibility="collapsed", on_change=on_name_change, args=(strat['id'], ui_name_key))
            with c_head1:
                st.markdown(f"**Spot at Entry:**"); st.markdown(f"${strat.get('spot_at_entry', 0.0):.2f}")
            with c_head2:
                st.markdown(f"**Net Entry Theo:**")
                ui_net_key = f"net_entry_{strat['id']}"
                st.number_input("Net Entry", value=float(net_entry_theo), step=0.01, format="%.3f", key=ui_net_key, label_visibility="collapsed", on_change=on_net_entry_change, args=(strat['id'], ui_net_key, max_qty, net_entry_theo))
            with c_head3:
                st.markdown(f"**Net Live Theo:**"); st.markdown(f"{net_live_theo:.3f}")
            with c_head4:
                st.markdown(f"**Spot Price Override:**")
                st.number_input("Override", value=override_val, step=0.10, key=ui_ovr_key, label_visibility="collapsed", placeholder="Enter price here", on_change=on_spot_override_change, args=(strat['id'], ui_ovr_key))
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            def port_compute_gross_margin(legs_list, arrays_list):
                if not legs_list: return 0.0
                subset_premium = sum(-(l['Qty'] * l['Entry'] * contract_multiplier) for l in legs_list)
                array_span_loss = abs(min(0.0, np.min(sum(r * l['Qty'] for r, l in zip(arrays_list, legs_list))))) if (len(arrays_list) > 0 and len(arrays_list[0]) > 1) else 0.0
                
                S = current_spot_val
                scan_pct = 0.065 if ticker_display == 'XJO' else 0.15
                test_spots = [S * (1 - scan_pct), S * (1 + scan_pct)]
                synthetic_pnls = [sum(l['Qty'] * ((max(0.0, spot - float(l['Strike'])) if l['Type'] == 'Call' else max(0.0, float(l['Strike']) - spot)) - l['Entry']) * contract_multiplier for l in legs_list) for spot in test_spots]
                
                unbounded_gross_risk = max(0.0, max(array_span_loss, abs(min(0.0, min(synthetic_pnls)))) + subset_premium)
                if (sum(l['Qty'] for l in legs_list if l['Type'] == 'Call') < 0) or (sum(l['Qty'] for l in legs_list if l['Type'] == 'Put') < 0): return unbounded_gross_risk
                
                strikes = [float(l['Strike']) for l in legs_list]
                if not strikes: return unbounded_gross_risk
                bound_pnls = [sum(l['Qty'] * max(0.0, spot - float(l['Strike'])) * contract_multiplier if l['Type'] == 'Call' else l['Qty'] * max(0.0, float(l['Strike']) - spot) * contract_multiplier for l in legs_list) for spot in strikes + [0.0, max(strikes) * 3.0]]
                return min(unbounded_gross_risk, abs(min(0.0, min(bound_pnls))))

            scen_cols_p = [c for c in st.session_state.ref_data.columns if 'Scenario' in str(c)] if st.session_state.ref_data is not None else []
            leg_risk_arrays_p = []
            tkr_p = ticker_display.replace(".AX", "")
            
            for leg in strat['legs']:
                match = pd.DataFrame()
                if st.session_state.ref_data is not None and not st.session_state.ref_data.empty:
                    ticker_mask = st.session_state.ref_data['Ticker'].isin(['XJO', 'XJOW']) if tkr_p == 'XJO' else st.session_state.ref_data['Ticker'] == tkr_p
                    match = st.session_state.ref_data[ticker_mask & (st.session_state.ref_data['Type'] == leg['Type']) & (st.session_state.ref_data['Strike'] == float(leg['Strike'])) & (st.session_state.ref_data['Expiry'].dt.strftime("%Y-%m-%d") == leg['ExpDateStr'])]
                leg_risk_arrays_p.append(match.iloc[0][scen_cols_p].values.astype(float) if not match.empty and scen_cols_p else np.zeros(len(scen_cols_p) if scen_cols_p else 1))

            port_total_margin = port_compute_gross_margin(strat['legs'], leg_risk_arrays_p)

            # IN-LINE TABLE
            p_h_col_spec = [0.8, 1.2, 0.8, 1.4, 1.3, 0.9, 1.1, 1.0, 1.2, 1.2, 0.4]
            h_cols = st.columns(p_h_col_spec)
            headers = ["Qty", "Code", "Type", "Expiry", "Strike", "Vol", "Entry $", "Live Theo", "Premium", "Margin", ""]
            for col, h in zip(h_cols, headers): col.markdown(f'<div class="trade-header">{h}</div>', unsafe_allow_html=True)

            for j, leg in enumerate(strat['legs']):
                disp_data = display_legs[j]
                c = st.columns(p_h_col_spec)
                row_bg = "rgba(74, 222, 128, 0.10)" if leg['Qty'] > 0 else "rgba(248, 113, 113, 0.10)"
                
                new_qty = c[0].number_input("Qty", value=int(leg['Qty']), step=1, key=f"p_qty_{strat['id']}_{j}", label_visibility="collapsed", on_change=set_active_strat, args=(strat['id'],))
                if new_qty != leg['Qty']: strat['legs'][j]['Qty'] = new_qty; st.session_state.trigger_db_save = True; port_needs_rerun = True
                    
                c[1].markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>{leg['Code']}</div>", unsafe_allow_html=True)
                c[2].markdown(f"<div class='strategy-text' style='background-color:{row_bg}; font-weight:600;'>{leg['Type']}</div>", unsafe_allow_html=True)
                
                tkr = ticker_display.replace(".AX", "")
                subset_st = pd.DataFrame()
                if st.session_state.ref_data is not None and not st.session_state.ref_data.empty:
                    ticker_mask = st.session_state.ref_data['Ticker'].isin(['XJO', 'XJOW']) if tkr == 'XJO' else st.session_state.ref_data['Ticker'] == tkr
                    subset_st = st.session_state.ref_data[ticker_mask & (st.session_state.ref_data['Type'] == leg['Type']) & (st.session_state.ref_data['Expiry'] >= get_sydney_time().replace(hour=0, minute=0, second=0, microsecond=0))]
                    
                valid_exps = sorted(subset_st['Expiry'].dropna().unique()) if not subset_st.empty else []
                exp_strs = [d.strftime("%Y-%m-%d") for d in valid_exps] or [leg['ExpDateStr']]
                exp_idx = exp_strs.index(leg['ExpDateStr']) if leg['ExpDateStr'] in exp_strs else 0
                    
                new_exp = c[3].selectbox("Expiry", options=exp_strs, index=exp_idx, key=f"p_exp_{strat['id']}_{j}", label_visibility="collapsed", on_change=set_active_strat, args=(strat['id'],), format_func=format_date_ui)
                if new_exp != leg['ExpDateStr']:
                    strat['legs'][j]['ExpDateStr'] = new_exp
                    subset_exp = subset_st[subset_st['Expiry'].dt.strftime("%Y-%m-%d") == new_exp]
                    if not subset_exp.empty:
                        avail_stk = sorted(subset_exp['Strike'].unique().tolist())
                        closest = min(avail_stk, key=lambda x: abs(x - float(leg['Strike']))) if avail_stk else float(leg['Strike'])
                        strat['legs'][j]['Strike'] = float(closest)
                        match = subset_exp[subset_exp['Strike'] == closest].sort_values('Code', ascending=False)
                        if not match.empty:
                            strat['legs'][j]['Code'] = str(match.iloc[0]['Code'])
                            strat['legs'][j]['Vol'] = float(match.iloc[0]['Vol'])
                            strat['legs'][j]['Style'] = match.iloc[0].get('Style', 'American')
                    st.session_state.trigger_db_save = True; st.session_state.open_strat_id = strat['id']; port_needs_rerun = True

                subset_exp = subset_st[subset_st['Expiry'].dt.strftime("%Y-%m-%d") == leg['ExpDateStr']] if not subset_st.empty else pd.DataFrame()
                avail_stk = sorted(subset_exp['Strike'].unique().tolist()) if not subset_exp.empty else [float(leg['Strike'])]
                stk_idx = avail_stk.index(float(leg['Strike'])) if float(leg['Strike']) in avail_stk else 0
                
                new_stk = c[4].selectbox("Strike", options=avail_stk, index=stk_idx, key=f"p_stk_{strat['id']}_{j}", label_visibility="collapsed", format_func=lambda x: f"{x:.2f}", on_change=set_active_strat, args=(strat['id'],))
                if new_stk != float(leg['Strike']):
                    strat['legs'][j]['Strike'] = new_stk
                    if not subset_exp.empty:
                        match = subset_exp[subset_exp['Strike'] == new_stk].sort_values('Code', ascending=False)
                        if not match.empty:
                            strat['legs'][j]['Code'] = str(match.iloc[0]['Code'])
                            strat['legs'][j]['Vol'] = float(match.iloc[0]['Vol'])
                            strat['legs'][j]['Style'] = match.iloc[0].get('Style', 'American')
                    st.session_state.trigger_db_save = True; st.session_state.open_strat_id = strat['id']; port_needs_rerun = True

                new_vol = c[5].number_input("Vol", value=float(leg['Vol']), step=0.5, format="%.1f", key=f"p_vol_{strat['id']}_{j}", label_visibility="collapsed", on_change=set_active_strat, args=(strat['id'],))
                if new_vol != leg['Vol']: strat['legs'][j]['Vol'] = new_vol; st.session_state.trigger_db_save = True; port_needs_rerun = True
                    
                new_entry = c[6].number_input("Entry $", value=float(leg['Entry']), step=0.01, format="%.3f", key=f"p_ent_{strat['id']}_{j}", label_visibility="collapsed", on_change=set_active_strat, args=(strat['id'],))
                if new_entry != leg['Entry']: strat['legs'][j]['Entry'] = new_entry; st.session_state.trigger_db_save = True; port_needs_rerun = True

                c[7].markdown(f"<div class='strategy-text' style='background-color:{row_bg};'>{disp_data['Live Theo']}</div>", unsafe_allow_html=True)
                c[8].markdown(f"<div class='strategy-text' style='background-color:{row_bg};'><span style='color:{'#4ade80' if disp_data['Raw_Premium']>=0 else '#f87171'}; font-weight:600;'>{disp_data['Premium']}</span></div>", unsafe_allow_html=True)
                
                row_margin = port_total_margin - port_compute_gross_margin(strat['legs'][:j] + strat['legs'][j+1:], leg_risk_arrays_p[:j] + leg_risk_arrays_p[j+1:])
                c[9].markdown(f"<div class='strategy-text' style='background-color:{row_bg};'><span style='font-weight:600;'>${row_margin:,.0f}</span></div>", unsafe_allow_html=True)

                with c[10]:
                    st.markdown("<div style='height: 1px;'></div>", unsafe_allow_html=True)
                    if st.button("✕", key=f"p_d_{strat['id']}_{j}", type="tertiary", width='content'):
                        strat['legs'].pop(j); st.session_state.trigger_db_save = True; st.session_state.open_strat_id = strat['id']; port_needs_rerun = True; break 
            
            st.markdown("<hr style='margin: -12px 0 8px 0; border-top: 1px solid #334155;'>", unsafe_allow_html=True)
            
            # SUMMARY ROW
            port_tot_prem = sum(disp['Raw_Premium'] for disp in display_legs)
            tot_prem_str = f"${port_tot_prem:,.2f}" if port_tot_prem >= 0 else f"-${abs(port_tot_prem):,.2f}"
            tot_mar_str = f"${port_total_margin:,.2f}" if port_total_margin >= 0 else f"-${abs(port_total_margin):,.2f}"

            with st.container():
                f = st.columns(p_h_col_spec)
                with f[1]: st.markdown("<div class='strategy-text' style='font-weight:bold;'>TOTAL STRATEGY</div>", unsafe_allow_html=True)
                with f[7]: st.markdown(f"<div class='strategy-text' style='font-weight:bold;'>{net_live_theo:.3f}</div>", unsafe_allow_html=True)
                with f[8]: st.markdown(f"<div class='strategy-text'><span style='color:{'#4ade80' if port_tot_prem>=0 else '#f87171'}; font-weight:bold;'>{tot_prem_str}</span></div>", unsafe_allow_html=True)
                with f[9]: st.markdown(f"<div class='strategy-text'><span style='font-weight:bold;'>{tot_mar_str}</span></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            a_c1, a_c2, a_c3, _ = st.columns([1.5, 1.5, 1.5, 2.5])
            with a_c1:
                if st.button("🗑️ Delete Trade", key=f"del_{strat['id']}", use_container_width=True):
                    st.session_state.portfolio.pop(i); st.session_state.trigger_db_save = True; st.session_state.open_strat_id = None; port_needs_rerun = True
            with a_c2:
                if st.button("📋 Duplicate", key=f"dup_{strat['id']}", use_container_width=True):
                    new_strat = copy.deepcopy(strat)
                    new_strat['id'] = str(uuid.uuid4())
                    new_strat['name'] = new_strat.get('name', 'Strategy') + " (Copy)"
                    for l in new_strat['legs']: l['id'] = str(uuid.uuid4())
                    st.session_state.portfolio.insert(i + 1, new_strat); st.session_state.trigger_db_save = True; st.session_state.open_strat_id = new_strat['id']; port_needs_rerun = True
            with a_c3:
                if st.button("🔄 Refresh", key=f"ref_live_{strat['id']}", use_container_width=True):
                    with st.spinner("Fetching live data..."):
                        data, msg, ext_spreads, d_date = load_databases(OPTIONS_SHEET_URL, FWD_CURVE_URL, str(uuid.uuid4())[:8])
                        if not data.empty:
                            st.session_state.ref_data = data
                            st.session_state.sheet_msg = msg
                            st.session_state.fwd_spreads = ext_spreads
                            st.session_state.data_date = d_date
                            try: 
                                if d_date != "Unknown":
                                    st.session_state.is_monday_data = (pd.to_datetime(d_date).weekday() == 0)
                            except: pass
                        
                        _, spot, _ = fetch_data(strat.get('ticker', 'XJO'))
                        if spot > 0: strat['current_spot'] = spot
                        
                        if 'override_spot' in strat: strat['override_spot'] = None 
                        ui_ovr_key = f"ui_ovr_spot_{strat['id']}"
                        if ui_ovr_key in st.session_state: del st.session_state[ui_ovr_key]
                        
                        for leg in strat['legs']:
                            if not data.empty:
                                match = data[data['Code'] == leg['Code']]
                                if not match.empty: 
                                    live_v = float(match.iloc[0]['Vol'])
                                    leg['Current_Vol'] = live_v
                                    leg['Vol'] = live_v
                        
                        st.session_state.trigger_db_save = True
                        st.session_state.open_strat_id = strat['id']
                        st.rerun()

            # THEO MATRIX
            show_matrix = st.checkbox("📈 Show Matrix", key=f"show_mx_{strat['id']}", on_change=set_active_strat, args=(strat['id'],))
            if show_matrix:
                st.markdown("##### Matrix")
                matrix_view_p = st.radio("Matrix Display Mode", ["Profit / Loss", "Theoretical Price"], horizontal=True, key=f"mx_mode_{strat['id']}", on_change=set_active_strat, args=(strat['id'],))
                mx_c1, mx_c2 = st.columns([1, 1.2], gap="large")
                
                with mx_c1:
                    mx_time_step = st.slider("Step (Days)", 1, 30, 1, key=f"mx_ts_{strat['id']}", on_change=set_active_strat, args=(strat['id'],))
                    vol_shift_sel = st.radio("Simulate Volatility Shift", ["IV -10%", "IV Flat", "IV +10%"], index=1, horizontal=True, key=f"mx_vs_{strat['id']}", on_change=set_active_strat, args=(strat['id'],))
                    mx_vol_mod = -10.0 if vol_shift_sel == "IV -10%" else (10.0 if vol_shift_sel == "IV +10%" else 0.0)

                with mx_c2:
                    mx_slider_placeholder = st.empty()
                    mx_step_type = st.radio("Step Type", ["Percentage (%)", "Points/Dollars ($)"], horizontal=True, key=f"mx_st_{strat['id']}", on_change=set_active_strat, args=(strat['id'],))
                    spot = float(current_spot_val)
                    if mx_step_type == "Percentage (%)":
                        mx_step_val = mx_slider_placeholder.select_slider("Price Step", options=[x / 200.0 for x in range(1, 11)], value=0.01, format_func=lambda x: f"{x*100:.1f}%", key=f"mx_sv_{strat['id']}", on_change=set_active_strat, args=(strat['id'],))
                        prices = [spot * (1 + mx_step_val * j) for j in range(6, -7, -1)]
                    else:
                        pts_opts = [10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0] if spot > 1000 else ([1.0, 2.0, 5.0, 10.0, 20.0, 25.0] if spot > 100 else [0.10, 0.25, 0.50, 1.00, 2.00, 5.00])
                        mx_step_val = mx_slider_placeholder.select_slider("Price Step", options=pts_opts, value=pts_opts[2], format_func=lambda x: f"{x:g}", key=f"mx_sv_{strat['id']}", on_change=set_active_strat, args=(strat['id'],))
                        prices = [spot + (mx_step_val * j) for j in range(6, -7, -1)]

                mx_dates = [d * mx_time_step for d in range(8)] 
                matrix_data = []
                for p in prices:
                    row = {"Price": f"» ${p:.2f} (SPOT) «" if math.isclose(p, spot, rel_tol=1e-5) else f"${p:.2f}"}
                    for d in mx_dates:
                        pnl, net_theo_sum = 0, 0
                        eval_dt_mx = st.session_state.get('fetch_time', get_sydney_time()) + timedelta(days=d)
                        for leg in strat['legs']:
                            exp_exact_dt = datetime.strptime(leg['ExpDateStr'], "%Y-%m-%d").replace(hour=16, minute=0)
                            if eval_dt_mx.date() >= exp_exact_dt.date():
                                active_eval_dt = exp_exact_dt
                                rem_days = 0.0
                            else:
                                active_eval_dt = eval_dt_mx
                                rem_days = (exp_exact_dt - active_eval_dt).total_seconds() / 86400.0
                                
                            exit_px, _ = calculate_price_and_delta(ticker_display, leg['Style'], leg['Type'], p, leg['Strike'], rem_days, max(1.0, leg.get('Current_Vol', leg['Vol']) + mx_vol_mod), leg['ExpDateStr'], eval_date=active_eval_dt)
                            pnl += (exit_px - leg['Entry']) * leg['Qty'] * contract_multiplier
                            net_theo_sum += exit_px * leg['Qty']
                        col_name = eval_dt_mx.strftime("%d-%b-%Y")
                        row[f"Today ({col_name})" if d == 0 else col_name] = pnl if matrix_view_p == "Profit / Loss" else (net_theo_sum / max_qty if max_qty != 0 else 0.0)
                    matrix_data.append(row)
                    
                df_mx = pd.DataFrame(matrix_data).set_index("Price")
                capital_at_risk = max(port_total_margin, abs(sum(-(l['Qty'] * l['Entry'] * contract_multiplier) for l in strat['legs']))) or 1.0
                
                def format_pnl(val):
                    try: return f"${float(val):,.0f} ({'+' if float(val)>0 else ''}{(float(val)/capital_at_risk)*100:.1f}%)" if pd.notna(val) else ""
                    except: return ""

                def make_heatmap(df):
                    abs_max = max(abs(df.max().max()), abs(df.min().min()), 1)
                    styles_df = pd.DataFrame('', index=df.index, columns=df.columns)
                    for idx in df.index:
                        is_spot = "SPOT" in str(idx)
                        for col in df.columns:
                            val = df.loc[idx, col]
                            s = f"background-color: rgba(74, 222, 128, {min(val/abs_max,1.0)*0.35+0.05:.2f}); " if val > 0 else (f"background-color: rgba(248, 113, 113, {min(abs(val)/abs_max,1.0)*0.35+0.05:.2f}); " if val < 0 else "")
                            if is_spot: 
                                s += "font-weight: bold; border-top: 2px solid rgba(255,255,255,0.5); border-bottom: 2px solid rgba(255,255,255,0.5);"
                            styles_df.loc[idx, col] = s
                    return styles_df

                def highlight_spot(df):
                    styles_df = pd.DataFrame('', index=df.index, columns=df.columns)
                    for idx in df.index:
                        if "SPOT" in str(idx):
                            styles_df.loc[idx, :] = "font-weight: bold; background-color: rgba(255,255,255,0.05); border-top: 2px solid rgba(255,255,255,0.5); border-bottom: 2px solid rgba(255,255,255,0.5);"
                    return styles_df

                if matrix_view_p == "Profit / Loss":
                    st.dataframe(df_mx.style.apply(make_heatmap, axis=None).format(format_pnl).set_table_styles([
                        {'selector': 'th', 'props': [('color', 'var(--text-color)'), ('font-weight', 'bold')]}
                    ]), use_container_width=True, height=500)
                else:
                    format_dict = {col: "{:.3f}" for col in df_mx.columns}
                    st.dataframe(df_mx.style.apply(highlight_spot, axis=None).format(format_dict).set_table_styles([
                        {'selector': 'th', 'props': [('color', 'var(--text-color)'), ('font-weight', 'bold')]}
                    ]), use_container_width=True, height=500)

            if port_needs_rerun: st.rerun()

# --- BACKGROUND WORDPRESS DATABASE SYNC ENGINE ---
if st.session_state.trigger_db_save and wp_uid:
    wp_save_portfolio(wp_uid, st.session_state.portfolio)
    st.session_state.trigger_db_save = False
