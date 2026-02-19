# ==========================================
# TradersCircle Options Calculator
# VERSION: 9.5 (Reverted to 9.3 state)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, time, timedelta
import pytz
import math

# --- 1. CONFIGURATION & THEME ---
st.set_page_config(layout="wide", page_title="TradersCircle Options")
RAW_SHEET_URL = "https://docs.google.com/spreadsheets/d/1d9FQ5mn--MSNJ_WJkU--IvoSRU0gQBqE0f9s9zEb0Q4/edit?usp=sharing"

# --- CSS STYLING ---
st.markdown("""
<style>
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
        background-color: #f8fafc !important; color: #334155 !important; border: 1px solid #cbd5e1;
    }
    
    /* --- SLIDER COLOR FIX (Electric Blue) --- */
    div[data-baseweb="slider"] > div > div > div { background-color: #0050FF !important; }
    div[role="slider"] { background-color: #0050FF !important; border: none !important; box-shadow: none !important; }
    div[data-testid="stSlider"] svg path { fill: #0050FF !important; stroke: #0050FF !important; }
    div[data-testid="stSlider"] p { color: white !important; }
    input[type=range] { accent-color: #0050FF !important; }
    
    /* Dataframe Row Selection Highlight (Teal) */
    [data-testid="stDataFrame"] [aria-selected="true"] > div {
        background-color: rgba(29, 191, 210, 0.4) !important;
        color: white !important;
    }
    
    .stDataFrame { border: none !important; }
    .trade-header {
        font-weight: 700; color: #94a3b8; font-size: 12px; text-transform: uppercase;
        margin-bottom: 5px; cursor: help; user-select: none;
    }
    
    /* Prevent accidental text highlighting in Strategy Table */
    .strategy-text { user-select: none; }
    
    button[kind="secondary"] {
        padding: 0rem 0.5rem !important; min-height: 0px !important; height: 32px !important;
    }
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
if 'manual_spot' not in st.session_state: st.session_state.manual_spot = False
if 'is_market_open' not in st.session_state: st.session_state.is_market_open = True
if 'div_info' not in st.session_state: st.session_state.div_info = None
if 'matrix_vol_mod' not in st.session_state: st.session_state.matrix_vol_mod = 0.0

if 'preselect_code' not in st.session_state: st.session_state.preselect_code = None
if 'preselect_expiry' not in st.session_state: st.session_state.preselect_expiry = None
if 'preselect_strike' not in st.session_state: st.session_state.preselect_strike = None

TOOLTIPS = {
    "Theo": "The theoretical fair value of the option calculated using the Black-Scholes or Bjerksund-Stensland pricing model.",
    "IV": "Implied Volatility: The market's forecast of a likely movement in the security's price.",
    "Delta": "The rate of change of the option price with respect to the underlying asset's price.",
    "Strike": "The set price at which the option contract can be exercised.",
    "Code": "The unique ASX exchange ticker symbol for this specific option contract.",
    "Entry": "The price you entered the trade at (or current market price).",
    "Premium": "The total cost or credit for the trade. Calculated as Price × Quantity × 100.",
    "Margin": "The estimated collateral required to hold this position."
}

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=600)
def load_sheet(raw_url):
    try:
        csv_url = f"{raw_url.split('/edit')[0]}/export?format=csv&gid=0" if "/edit" in raw_url else raw_url
        df = pd.read_csv(csv_url, on_bad_lines='skip', dtype=str)
        
        header_map = {
            'ASXCode': 'Code', 'Underlying': 'Ticker', 'OptType': 'Type', 
            'ExpDate': 'Expiry', 'Strike': 'Strike', 'Volatility': 'Vol', 
            'Settlement': 'Settlement', 'Style': 'Style', 'Lookup Key': 'LookupKey'
        }
        df = df.rename(columns=header_map)
        
        required = ['Code', 'Ticker', 'Strike',
