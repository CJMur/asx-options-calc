import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="TradersCircle Options")

# ==========================================
# 🛑 YOUR GOOGLE SHEET LINK
# ==========================================
SHEET_URL = "https://docs.google.com/spreadsheets/d/1d9FQ5mn--MSNJ_WJkU--IvoSRU0gQBqE0f9s9zEb0Q4/export?format=csv"

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .main-header {
        background-color: #0e1b32; padding: 1.5rem 2rem; color: white;
        border-radius: 8px; display: flex; justify_content: space-between; align_items: center;
        margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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

# --- 3. ROBUST DATA ENGINE ---

@st.cache_data(ttl=600)
def load_reference_data(url):
    try:
        # Load columns A-E only
        df = pd.read_csv(url, usecols=[0, 1, 2, 3, 4], on_bad_lines='skip')
        
        # Standardize Header Names
        df.columns = ['Ticker', 'Expiry', 'Strike', 'Type', 'Code']
        
        # Clean Data Types for Matching
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
        df['Type'] = df['Type'].astype(str).str.title().str.strip() 
        df['Strike'] = df['Strike'].astype(str).str.replace('$', '').astype(float)
        
        # Flexible Date Parsing
        df['Expiry'] = pd.to_datetime(df['Expiry'], dayfirst=True, errors='coerce') 
        
        return df
    except Exception as e:
        return pd.DataFrame()

# Load Data on Startup
st.session_state.ref_data = load_reference_data(SHEET_URL)

def lookup_code(ticker, expiry_str, strike, kind):
    df = st.session_state.ref_data
    if df is None or df.empty: return "ENTER_CODE"

    try:
        # Normalize the Yahoo Date to match Sheet Date
        target_date = pd.to_datetime(expiry_str)
        
        # Filter Logic
