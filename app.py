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

# --- CSS STYLING ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    
    /* HEADER */
    .main-header {
        background-color: #0e1b32; 
        padding: 1rem 1.5rem; 
        color: white;
        border-radius: 8px; 
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title { font-size: 22px; font-weight: 700; white-space: nowrap; }
    
    /* STATUS BADGE */
    .status-badge {
        font-size: 11px; background-color: #f1f5f9; color: #475569; 
        padding: 2px 6px; border-radius: 4px; border: 1px solid #cbd5e1;
        display: inline-block; margin-top: 4px;
    }
    .status-success { color: #16a34a; background-color: #f0fdf4; border-color: #bbf7d0; }
    .status-error { color: #dc2626; background-color: #fef2f2; border-color: #fecaca; }
    
    /* BUTTONS */
    div[data-testid="stButton"] > button {
        background-color: #ef4444 !important; 
        color: white !important;
        border: none;
        font-weight: 600;
    }
    
    /* TABLE TWEAKS */
    .stDataFrame { border: none !important; }
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
