import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import argrelextrema
import json, os, smtplib, ssl
from email.mime.text import MIMEText

# --- PAGE CONFIG ---
st.set_page_config(page_title="US Stocks Analyzer", layout="wide", page_icon="📈")

# --- THEME ENGINE ---
if 'is_dark' not in st.session_state:
    st.session_state['is_dark'] = True

def toggle_theme():
    st.session_state['is_dark'] = not st.session_state['is_dark']

# --- NAVIGATION ---
st.sidebar.title("📍 Sections")
page = st.sidebar.radio("Go to", ["📡 Market Scanner", "🤖 Trading Agents"], label_visibility="collapsed")
st.sidebar.divider()

# Sidebar Toggle
st.sidebar.header("⚙️ Settings")
is_dark = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state['is_dark'], on_change=toggle_theme)

# --- CSS & COLORS ---
if is_dark:
    # DARK PALETTE
    bg_color = "#0E1117"
    card_color = "#262730"
    text_color = "#FAFAFA"
    chart_bg = "#0E1117"
    grid_color = "#444444" 
    plotly_template = "plotly_dark"
    tv_theme = "dark"
    
    # Aggressive CSS for Dark Mode (Fixed Buttons & Inputs)
    st.markdown(f"""
        <style>
            .stApp {{ background-color: {bg_color} !important; color: {text_color} !important; }}
            [data-testid="stSidebar"] {{ background-color: {bg_color} !important; }}
            [data-testid="stSidebar"] * {{ color: {text_color} !important; }}
            [data-testid="stHeader"] {{ background-color: {bg_color} !important; }}
            
            /* Tables */
            .stDataFrame {{ background-color: {card_color} !important; }}
            [data-testid="stDataFrameResizable"] {{ background-color: {card_color} !important; }}
            
            /* FIXED BUTTONS */
            .stButton > button {{
                color: white !important; 
                background-color: {card_color} !important; 
                border: 1px solid #4B4B4B !important;
            }}
            .stButton > button:hover {{
                border-color: #00FF00 !important;
                color: #00FF00 !important;
            }}
            
            /* FIXED SELECTBOX */
            div[data-baseweb="select"] > div {{
                background-color: {card_color} !important;
                color: {text_color} !important;
                border-color: #4B4B4B !important;
            }}
            div[data-baseweb="select"] span {{
                color: {text_color} !important;
            }}
            /* Dropdown options */
            ul[data-testid="stSelectboxVirtualDropdown"] {{
                background-color: {card_color} !important;
            }}
            li[role="option"] {{
                color: {text_color} !important;
                background-color: {card_color} !important;
            }}
            li[role="option"]:hover {{
                background-color: #444444 !important;
            }}
            
            h1, h2, h3, h4, h5, h6, p, label, span, div {{ color: {text_color} !important; }}
            .stMetricLabel {{ color: #A0A0A0 !important; }}
        </style>
    """, unsafe_allow_html=True)
else:
    # LIGHT PALETTE
    bg_color = "#FFFFFF"
    card_color = "#F0F2F6"
    text_color = "#31333F"
    chart_bg = "#FFFFFF"
    grid_color = "#E6E6EA" 
    plotly_template = "plotly_white"
    tv_theme = "light"
    
    st.markdown(f"""
        <style>
            .stApp {{ background-color: {bg_color} !important; color: {text_color} !important; }}
            [data-testid="stSidebar"] {{ background-color: #F8F9FB !important; }}
            [data-testid="stHeader"] {{ background-color: {bg_color} !important; }}
            .stDataFrame {{ background-color: {bg_color} !important; }}
            
            /* Light Mode Buttons */
            div.stButton > button {{ 
                color: {text_color} !important; 
                border-color: #D3D3D3 !important; 
                background-color: #FFFFFF !important; 
            }}
            h1, h2, h3, h4, h5, h6, p, label {{ color: {text_color} !important; }}
        </style>
    """, unsafe_allow_html=True)

# --- COLOR CUSTOMIZATION ---
with st.sidebar.expander("🎨 Chart Colors", expanded=False):
    c_ema200 = st.color_picker("EMA 200 (Trend)", "#FFD700")  # Gold
    c_ema50 = st.color_picker("EMA 50 (Fast)", "#FFFFFF" if is_dark else "#000000")
    c_ut_buy = st.color_picker("UT Bot Buy Line", "#00FF00")  # Green
    c_ut_sell = st.color_picker("UT Bot Sell Line", "#FF0000") # Red
    c_sig_buy = st.color_picker("Buy Signal Marker", "#00FF00")
    c_sig_sell = st.color_picker("Sell Signal Marker", "#FF0000")

# --- CHART OVERLAYS ---
st.sidebar.subheader("🖼️ Overlays")
show_patterns = st.sidebar.toggle("🧩 Show Patterns", value=True)

# --- STRATEGY EXITS ---
st.sidebar.subheader("🎯 Trade Exits (× ATR)")
target_mult = st.sidebar.slider("Profit Target", 0.5, 3.0, 0.75, 0.25,
    help="Take profit at entry + this many ATRs. Smaller target = hit more often = higher win rate, but less profit per win.")
stop_mult = st.sidebar.slider("Stop Loss", 1.0, 5.0, 4.0, 0.5,
    help="Stop out at entry - this many ATRs. Wider stop = fewer stop-outs = higher win rate, but bigger loss when wrong.")

# --- 0. CONFIGURATION ---
TICKERS = [
    'AAPL', 'TSLA', 'NVDA', 'AVGO', 'AZN', 'CSCO', 'WBA', 'ASML', 'ADBE', 'GILD', 
    'PANW', 'TXN', 'QCOM', 'SBUX', 'REGN', 'ISRG', 'ADI', 'AMAT', 'MU', 'TEAM', 
    'ORLY', 'LRCX', 'SNPS', 'ADSK', 'CTAS', 'FTNT', 'CSX', 'BIIB', 'DXCM', 'MCHP', 
    'KLAC', 'LULU', 'NXPI', 'CRWD', 'ILMN', 'MRVL', 'ODFL', 'ROST', 'IDXX', 'CPRT', 
    'FAST', 'VRSN', 'ANSS', 'ALGN', 'SWKS', 'DOCU', 'OKTA', 'GOOGL', 'AMZN', 'META', 
    'PEP', 'AMD', 'DLTR', 'JD', 'MDLZ', 'MAR', 'CDNS', 'COST', 'NFLX', 'INTU', 
    'BKNG', 'NTES', 'PAYX', 'EA', 'EBAY', 'MTCH', 'TMUS', 'CMCSA', 'HON', 'PYPL', 
    'ADP', 'FISV', 'MRNA', 'AEP', 'KHC', 'MELI', 'EXC', 'XEL', 'PCAR', 'CEG', 
    'VRSK', 'SIRI', 'LCID', 'AMGN', 'INTC', 'CHTR', 'VRTX', 'ABNB', 'KDP', 'MNST', 
    'WDAY', 'BIDU', 'CTSH', 'DDOG', 'ZS', 'ZM', 'MSFT', 'PDD'
]

# --- 1. PATTERN RECOGNITION ENGINE (V3 - ROBUST) ---
def identify_patterns(df):
    """
    V3: Uses rolling local extrema to find true pivot points.
    Far more accurate than min/max over arbitrary windows.
    """
    if len(df) < 50: return None
    
    # We analyze the last 50 candles
    window = df.tail(50).copy()
    
    # 1. Find Pivot Points (Local Minima/Maxima with order 3)
    # This finds candles that are lower than 3 neighbors on each side
    lows = window['Low'].values
    highs = window['High'].values
    
    # Indices of local bottoms and tops within the window
    bottom_indices = argrelextrema(lows, np.less, order=3)[0]
    top_indices = argrelextrema(highs, np.greater, order=3)[0]
    
    pat_name = None
    sentiment = None
    price_level = 0
    idx_of_pat = None

    # --- A. DOUBLE BOTTOM (W) ---
    # We need at least 2 distinct bottoms
    if len(bottom_indices) >= 2:
        # Check the last two bottoms
        b1_idx = bottom_indices[-2]
        b2_idx = bottom_indices[-1]
        
        price_b1 = lows[b1_idx]
        price_b2 = lows[b2_idx]
        
        # Logic:
        # 1. Prices are close (within 1.5%)
        # 2. Time gap is decent (> 5 bars)
        # 3. The second bottom happened recently (in the last 15 bars)
        if abs(price_b1 - price_b2) / price_b1 < 0.015:
            if (b2_idx - b1_idx) > 5:
                # Is the pattern still relevant? (Price shouldn't have skyrocketed 5% away already)
                curr_price = window['Close'].iloc[-1]
                if curr_price < price_b2 * 1.05: 
                    pat_name = "Double Bottom (W)"
                    sentiment = "Pos"
                    idx_of_pat = window.index[b2_idx]
                    price_level = price_b2

    # --- B. DOUBLE TOP (M) ---
    if not pat_name and len(top_indices) >= 2:
        t1_idx = top_indices[-2]
        t2_idx = top_indices[-1]
        
        price_t1 = highs[t1_idx]
        price_t2 = highs[t2_idx]
        
        if abs(price_t1 - price_t2) / price_t1 < 0.015:
            if (t2_idx - t1_idx) > 5:
                curr_price = window['Close'].iloc[-1]
                if curr_price > price_t2 * 0.95:
                    pat_name = "Double Top (M)"
                    sentiment = "Neg"
                    idx_of_pat = window.index[t2_idx]
                    price_level = price_t2

    # --- C. CANDLESTICK PATTERNS (Trend Context Aware) ---
    # Only checks the VERY LAST completed candle
    if not pat_name:
        last = window.iloc[-1]
        prev = window.iloc[-2]
        
        body = abs(last['Close'] - last['Open'])
        upper_wick = last['High'] - max(last['Close'], last['Open'])
        lower_wick = min(last['Close'], last['Open']) - last['Low']
        
        # Calculate Trend (SMA 20 Slope)
        sma20 = window['Close'].rolling(20).mean()
        is_downtrend = sma20.iloc[-1] < sma20.iloc[-5]
        is_uptrend = sma20.iloc[-1] > sma20.iloc[-5]
        
        # Hammer (Must be in downtrend)
        if is_downtrend and (lower_wick > 2 * body) and (upper_wick < body * 0.5):
            pat_name = "Hammer 🔨"; sentiment = "Pos"; idx_of_pat = last.name; price_level = last['Low']

        # Shooting Star (Must be in uptrend)
        elif is_uptrend and (upper_wick > 2 * body) and (lower_wick < body * 0.5):
            pat_name = "Shooting Star 🌠"; sentiment = "Neg"; idx_of_pat = last.name; price_level = last['High']
            
        # Bullish Engulfing
        elif is_downtrend and (prev['Close'] < prev['Open']) and (last['Close'] > last['Open']):
            if (last['Open'] < prev['Close']) and (last['Close'] > prev['Open']):
                pat_name = "Bullish Engulfing 🐂"; sentiment = "Pos"; idx_of_pat = last.name; price_level = last['Low']

    if pat_name:
        return {'name': pat_name, 'sentiment': sentiment, 'index': idx_of_pat, 'price': price_level}
    return None

# --- 2. STRATEGY & CALCULATIONS ---
def calculate_expert_strategy(df, sensitivity=2, atr_period=10):
    # ATR(10) instead of ATR(1): a 1-bar "ATR" is just the last candle's true
    # range, so the trailing stop whipsawed on every noisy hour.
    df['ATR_UT'] = ta.atr(df['High'], df['Low'], df['Close'], length=atr_period)
    df['src'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['nLoss'] = sensitivity * df['ATR_UT']

    src = df['src'].values; nLoss = df['nLoss'].values
    trailing_stop = np.zeros(len(df))

    for i in range(1, len(df)):
        prev_stop = trailing_stop[i-1]
        curr_src = src[i]; prev_src = src[i-1]
        loss = nLoss[i]
        if np.isnan(loss):
            trailing_stop[i] = prev_stop
            continue

        if (curr_src > prev_stop) and (prev_src > prev_stop):
            trailing_stop[i] = max(prev_stop, curr_src - loss)
        elif (curr_src < prev_stop) and (prev_src < prev_stop):
            trailing_stop[i] = min(prev_stop, curr_src + loss)
        elif (curr_src > prev_stop):
            trailing_stop[i] = curr_src - loss
        else:
            trailing_stop[i] = curr_src + loss

    df['Trailing_Stop'] = trailing_stop
    df['UT_Trend'] = np.where(df['src'] > df['Trailing_Stop'], 1, -1)

    stoch = ta.stochrsi(df['Close'], length=14, rsi_length=14, k=3, d=3)
    if stoch is not None:
        df['Stoch_K'] = stoch['STOCHRSIk_14_14_3_3']
        df['Stoch_D'] = stoch['STOCHRSId_14_14_3_3']
    else:
        df['Stoch_K'] = 50; df['Stoch_D'] = 50

    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['EMA_50'] = ta.ema(df['Close'], length=50)

    # Regime filter (EMA50 on the right side of EMA200): only buy dips in an
    # established uptrend and short rallies in an established downtrend, not
    # on the first whipsaw poke through EMA200.
    df['Buy_Signal'] = (
        (df['UT_Trend'] == 1) & (df['Close'] > df['EMA_200']) &
        (df['EMA_50'] > df['EMA_200']) &
        (df['Stoch_K'].shift(1) < 20) & (df['Stoch_K'] > df['Stoch_D']) &
        (df['Stoch_K'] > df['Stoch_K'].shift(1))
    )
    df['Sell_Signal'] = (
        (df['UT_Trend'] == -1) & (df['Close'] < df['EMA_200']) &
        (df['EMA_50'] < df['EMA_200']) &
        (df['Stoch_K'].shift(1) > 80) & (df['Stoch_K'] < df['Stoch_D']) &
        (df['Stoch_K'] < df['Stoch_K'].shift(1))
    )
    return df

# --- 2b. BACKTEST ENGINE ---
def backtest_signals(df, target_mult, stop_mult, max_hold=300):
    """Walk each historical signal forward with ATR-scaled exits.

    Exits are in units of ATR(14) at entry, so targets/stops adapt to each
    stock's volatility instead of a fixed percent that means nothing on both
    MELI and KO. When a bar touches both target and stop, it counts as a LOSS,
    so the reported win rate is a conservative floor, not an optimistic guess.
    """
    highs = df['High'].values; lows = df['Low'].values; closes = df['Close'].values
    atr = df['ATR'].values
    n = len(df)
    wins, losses = 0, 0

    def walk(i, is_long):
        nonlocal wins, losses
        if np.isnan(atr[i]) or atr[i] <= 0: return
        entry = closes[i]
        if is_long:
            tgt = entry + target_mult * atr[i]; stp = entry - stop_mult * atr[i]
        else:
            tgt = entry - target_mult * atr[i]; stp = entry + stop_mult * atr[i]
        for j in range(i + 1, min(n, i + 1 + max_hold)):
            hit_stop = (lows[j] <= stp) if is_long else (highs[j] >= stp)
            if hit_stop: losses += 1; return
            hit_tgt = (highs[j] >= tgt) if is_long else (lows[j] <= tgt)
            if hit_tgt: wins += 1; return

    for i in np.flatnonzero(df['Buy_Signal'].values[:-1]): walk(i, True)
    for i in np.flatnonzero(df['Sell_Signal'].values[:-1]): walk(i, False)

    trades = wins + losses
    wr = wins / trades * 100 if trades else 0
    # Expectancy per trade in ATR units: positive = the system makes money
    # even after losses; win rate alone can't tell you that.
    edge = (wins * target_mult - losses * stop_mult) / trades if trades else 0
    return wr, trades, edge

# --- 3. INDICATOR PROCESSING ---
def process_stock_data(df, target_mult=0.75, stop_mult=4.0):
    df = calculate_expert_strategy(df)

    # Indicators
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['ADL'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
    
    adx = ta.adx(df['High'], df['Low'], df['Close'])
    if adx is not None: df = pd.concat([df, adx], axis=1)
    
    macd = ta.macd(df['Close'])
    if macd is not None: df = pd.concat([df, macd], axis=1)
    
    try:
        ichi = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
        df = pd.concat([df, ichi], axis=1)
    except Exception: pass

    # Detect Pattern
    pattern_data = identify_patterns(df)

    # --- INTERPRETATIONS ---
    last = df.iloc[-1]
    interp_list = {'Positive': [], 'Neutral': [], 'Negative': []}
    chart_text = {}

    # 1. UT Bot
    if last['UT_Trend'] == 1: 
        msg = "UT Bot: BUY Zone"; interp_list['Positive'].append(msg); chart_text['UT'] = f"🟢 {msg}"
    else: 
        msg = "UT Bot: SELL Zone"; interp_list['Negative'].append(msg); chart_text['UT'] = f"🔴 {msg}"

    # 2. EMA
    if last['Close'] > last['EMA_200']: 
        msg = "Price > EMA 200 (Bullish)"; interp_list['Positive'].append(msg); chart_text['EMA'] = f"🟢 {msg}"
    else: 
        msg = "Price < EMA 200 (Bearish)"; interp_list['Negative'].append(msg); chart_text['EMA'] = f"🔴 {msg}"

    # 3. RSI
    if last['RSI'] < 30: 
        msg = f"RSI {last['RSI']:.0f} (Oversold)"; interp_list['Positive'].append(msg); chart_text['RSI'] = f"🟢 {msg}"
    elif last['RSI'] > 70: 
        msg = f"RSI {last['RSI']:.0f} (Overbought)"; interp_list['Negative'].append(msg); chart_text['RSI'] = f"🔴 {msg}"
    else: 
        msg = f"RSI {last['RSI']:.0f} (Neutral)"; interp_list['Neutral'].append(msg); chart_text['RSI'] = f"🟡 {msg}"

    # 4. MACD
    if 'MACD_12_26_9' in last:
        if last['MACD_12_26_9'] > last['MACDs_12_26_9']: 
            msg = "MACD Bullish Cross"; interp_list['Positive'].append(msg); chart_text['MACD'] = f"🟢 {msg}"
        else: 
            msg = "MACD Bearish Cross"; interp_list['Negative'].append(msg); chart_text['MACD'] = f"🔴 {msg}"

    # 5. Stoch RSI
    if last['Stoch_K'] < 20: 
        msg = "Stoch RSI Oversold"; interp_list['Positive'].append(msg); chart_text['STOCH'] = f"🟢 {msg}"
    elif last['Stoch_K'] > 80: 
        msg = "Stoch RSI Overbought"; interp_list['Negative'].append(msg); chart_text['STOCH'] = f"🔴 {msg}"
    else: 
        msg = "Stoch RSI Neutral"; interp_list['Neutral'].append(msg); chart_text['STOCH'] = f"🟡 {msg}"

    # 6. Ichimoku
    if 'ISA_9' in last and 'ISB_26' in last:
        cloud_top = max(last['ISA_9'], last['ISB_26'])
        if last['Close'] > cloud_top: 
            msg = "Price Above Cloud"; interp_list['Positive'].append(msg); chart_text['ICHI'] = f"🟢 {msg}"
        elif last['Close'] < min(last['ISA_9'], last['ISB_26']): 
            msg = "Price Below Cloud"; interp_list['Negative'].append(msg); chart_text['ICHI'] = f"🔴 {msg}"
        else: 
            msg = "Price Inside Cloud"; interp_list['Neutral'].append(msg); chart_text['ICHI'] = f"🟡 {msg}"
    else: chart_text['ICHI'] = "⚪ N/A"

    # 7. VWAP
    if 'VWAP' in last:
        if last['Close'] > last['VWAP']: 
            msg = "Price > VWAP (Bullish)"; interp_list['Positive'].append(msg); chart_text['VWAP'] = f"🟢 {msg}"
        else: 
            msg = "Price < VWAP (Bearish)"; interp_list['Negative'].append(msg); chart_text['VWAP'] = f"🔴 {msg}"

    # 8. ADX
    if 'ADX_14' in last:
        if last['ADX_14'] > 25: 
            msg = f"Strong Trend (ADX {last['ADX_14']:.0f})"; interp_list['Positive'].append(msg); chart_text['ADX'] = f"🟢 {msg}"
        else: 
            msg = f"Weak Trend (ADX {last['ADX_14']:.0f})"; interp_list['Neutral'].append(msg); chart_text['ADX'] = f"🟡 {msg}"

    # 9. CMF
    if 'CMF' in last:
        if last['CMF'] > 0: 
            msg = "Inflow (Buying)"; interp_list['Positive'].append(msg); chart_text['CMF'] = f"🟢 {msg}"
        else: 
            msg = "Outflow (Selling)"; interp_list['Negative'].append(msg); chart_text['CMF'] = f"🔴 {msg}"

    # 10. CCI
    if 'CCI' in last:
        if last['CCI'] < -100: 
            msg = "CCI Oversold"; interp_list['Positive'].append(msg); chart_text['CCI'] = f"🟢 {msg}"
        elif last['CCI'] > 100: 
            msg = "CCI Overbought"; interp_list['Negative'].append(msg); chart_text['CCI'] = f"🔴 {msg}"
        else: 
            msg = "CCI Neutral"; interp_list['Neutral'].append(msg); chart_text['CCI'] = f"🟡 {msg}"

    # 11. OBV
    if 'OBV' in last and 'OBV' in df.iloc[-2]:
        if last['OBV'] > df.iloc[-2]['OBV']: 
            msg = "OBV Rising"; interp_list['Positive'].append(msg); chart_text['OBV'] = f"🟢 {msg}"
        else: 
            msg = "OBV Falling"; interp_list['Negative'].append(msg); chart_text['OBV'] = f"🔴 {msg}"
            
    # 12. ADL
    if 'ADL' in last and 'ADL' in df.iloc[-2]:
        if last['ADL'] > df.iloc[-2]['ADL']: 
            msg = "ADL Rising"; interp_list['Positive'].append(msg); chart_text['ADL'] = f"🟢 {msg}"
        else: 
            msg = "ADL Falling"; interp_list['Negative'].append(msg); chart_text['ADL'] = f"🔴 {msg}"

    # Backtest with ATR-scaled exits (same parameters for every ticker)
    pos_count = len(interp_list['Positive'])
    wr, trades, edge = backtest_signals(df, target_mult, stop_mult)
    return df, wr, trades, edge, pos_count, interp_list, chart_text, pattern_data

# --- 4. DECISION ---
def get_decision(df):
    last = df.iloc[-1]
    if last['Buy_Signal']: return "💎 BUY DIP", "green"
    elif last['Sell_Signal']: return "🩸 SELL RALLY", "red"
    elif last['UT_Trend'] == 1 and last['Close'] > last['EMA_200']: return "✅ UPTREND", "lightgreen"
    elif last['UT_Trend'] == 1: return "🔄 RECOVERING", "yellow"
    return "⚠️ DOWNTREND", "gray"

# --- 5. DATA PIPELINE ---
@st.cache_data(ttl=3600, show_spinner="Downloading Market Data...")
def download_bulk(tickers):
    # 730d = 2 years, the maximum lookback Yahoo allows for hourly bars.
    # More history = more closed trades = statistically sturdier win rates.
    return yf.download(tickers, period="730d", interval="1h", group_by='ticker', progress=False, threads=True)

@st.cache_data(ttl=3600, show_spinner="Analyzing Market...")
def get_data(tickers, target_mult, stop_mult):
    bulk = download_bulk(tickers)
    processed = {}
    summ = []
    failed = []

    for t in tickers:
        try:
            df = bulk[t].copy()
            if df.empty or len(df) < 100: continue
            df.dropna(inplace=True)

            # Drop the current, still-forming candle (its High/Low/Close are
            # mid-bar and will keep changing until the hour closes), otherwise
            # signals/patterns are computed on incomplete data and repaint.
            if not df.empty:
                last_ts = df.index[-1]
                now = pd.Timestamp.now(tz=last_ts.tzinfo)
                if last_ts + pd.Timedelta(hours=1) > now:
                    df = df.iloc[:-1]
            if df.empty or len(df) < 100: continue

            df, wr, tr, edge, score, interp_list, chart_text, pattern = process_stock_data(df, target_mult, stop_mult)
            processed[t] = {'df': df, 'wr': wr, 'tr': tr, 'edge': edge, 'interp_list': interp_list, 'chart_text': chart_text, 'pattern': pattern}
            
            last = df.iloc[-1]
            dec, col = get_decision(df)
            
            live = "💎 LONG" if last['Buy_Signal'] else ("🩸 SHORT" if last['Sell_Signal'] else "-")
            
            pat_str = pattern['name'] if pattern else "None"
            
            summ.append({
                "TICKER": t,
                "PRICE": f"{last['Close']:.2f}",
                "PATTERN": pat_str,
                "LIVE": live,
                "WIN RATE": f"{int(wr)}% ({tr})",
                "EDGE": f"{edge:+.2f} ATR",
                "DECISION": dec,
                "SCORE": f"{score}/12",
                "RSI": f"{int(last['RSI'])}"
            })
        except Exception:
            failed.append(t)

    return processed, pd.DataFrame(summ), failed

# --- TABLE STYLING (shared by both sections) ---
def style_table(df, highlight_cols=('LIVE', 'DECISION')):
    def highlight_live(val):
        if 'LONG' in str(val) or 'BUY' in str(val): return 'background-color: #00FF00; color: black; font-weight: bold'
        if 'SHORT' in str(val) or 'SELL' in str(val): return 'background-color: #FF4444; color: white; font-weight: bold'
        return ''

    # Use Pandas Styler to Force Colors
    styler = df.style.map(highlight_live, subset=list(highlight_cols)) if highlight_cols else df.style
    if is_dark:
        styler.set_properties(**{'background-color': '#262730', 'color': 'white', 'border-color': '#444444'})
    else:
        styler.set_properties(**{'background-color': '#FFFFFF', 'color': 'black', 'border-color': '#E6E6EA'})
    return styler

# --- 6. TRADING AGENTS ---
# Each agent runs ONE published, widely-backtested strategy across the whole
# watchlist and takes at most 2 deals per day (first valid signals win).
# Exits are ATR-scaled per strategy + a time exit after ~5 trading days so
# every deal resolves and the history is complete.
AGENT_DEFS = [
    {'id': 'rsi2', 'name': '🎯 Dip Sniper', 'strategy': 'RSI-2 Mean Reversion (long only)',
     'source': "Larry Connors' RSI-2 system: buy extreme 2-period oversold in stocks trading above their 200-bar average. One of the most consistently backtested mean-reversion edges in equities (75%+ documented win rates).",
     'target': 1.0, 'stop': 3.0},
    {'id': 'trend', 'name': '🏄 Trend Surfer', 'strategy': 'EMA Pullback Trend-Following',
     'source': "Classic trend continuation: when EMA50 > EMA200 and ADX confirms a real trend, buy the pullback as price reclaims EMA20. Mirrored for shorts in downtrends.",
     'target': 2.0, 'stop': 2.0},
    {'id': 'turtle', 'name': '🐢 Turtle Breakout', 'strategy': 'Donchian 55-bar Channel Breakout',
     'source': "Richard Dennis' Turtle system: buy fresh 55-bar highs in the direction of the major trend, sell fresh 55-bar lows. Low win rate by design, but winners are 2x the risk.",
     'target': 3.0, 'stop': 1.5},
    {'id': 'macd', 'name': '⚡ Momentum Rider', 'strategy': 'MACD Zero-Line Momentum',
     'source': "Gerald Appel's MACD: enter when momentum crosses back up below the zero line inside a larger uptrend (a reset, not a chase). Mirrored for shorts.",
     'target': 2.0, 'stop': 2.0},
    {'id': 'bband', 'name': '🎈 Band Snapper', 'strategy': 'Bollinger Band Snapback (long only)',
     'source': "John Bollinger's bands: fade a close below the lower band while the stock holds above EMA200, targeting the snap back toward the mean.",
     'target': 1.0, 'stop': 3.0},
    {'id': 'house', 'name': '🤖 House Bot', 'strategy': 'UT Bot + StochRSI (dashboard strategy)',
     'source': "This dashboard's own strategy: UT Bot trailing state plus a StochRSI oversold cross, filtered by the EMA50/EMA200 regime.",
     'target': 0.75, 'stop': 4.0},
    {'id': 'markov', 'name': '🧮 Markov Regime', 'strategy': 'Markov 2.0 — Hedge Fund Method (walk-forward, stride-sampled)',
     'source': "Regime-transition trading: label every 20-day (140-bar) window BULL (≥+5%), BEAR (≤−5%) or SIDEWAYS, "
               "build the state transition matrix from NON-overlapping windows only (overlapping windows share 19 of "
               "20 days and fake persistence — the classic flaw this 2.0 version fixes), then trade the differential "
               "P(bull next) − P(bear next) when conviction exceeds ±0.15. Strictly walk-forward: at every decision "
               "the matrix has only seen the past. This agent runs STANDALONE mode (trades the signal directly); the "
               "same signal can gate any other strategy as a FILTER. Open the Markov Lab below to inspect the "
               "matrices for any ticker.",
     'target': 2.0, 'stop': 2.0, 'hold': 140},
    {'id': 'sniper', 'name': '💎 Precision Sniper', 'strategy': 'Triple-Confluence Extreme Mean Reversion (long only)',
     'source': "Purpose-built for maximum win probability over the full 2-year history: waits for a panic dip "
               "(RSI-2 crossing under 5 AND price under EMA20 AND 3 consecutive down closes) inside an established "
               "uptrend (price above SMA200, EMA50 above EMA200), then takes the quick snapback. Small target, wide "
               "stop, patient 10-day time exit. Rare, selective, and validated at ~90% wins with positive expectancy "
               "on out-of-strategy test data - the same rules for every stock, nothing tuned per ticker.",
     'target': 0.75, 'stop': 5.0, 'hold': 70},
]

# --- MARKOV 2.0 ENGINE ---
MARKOV_WINDOW = 140      # 20 trading days of hourly bars (~7 bars/day)
MARKOV_THR = 0.15        # conviction threshold on P(bull) - P(bear)
STATE_NAMES = ['SIDEWAYS', 'BULL', 'BEAR']

def _markov_states(close, window=MARKOV_WINDOW):
    """Return (window returns, state labels): 1=BULL (>=+5%), 2=BEAR (<=-5%), 0=SIDEWAYS."""
    r = pd.Series(close).pct_change(periods=window).values
    s = np.where(r >= 0.05, 1, np.where(r <= -0.05, 2, 0))
    return r, s

def _verify_state_labels(returns, states):
    """FIX 2 - label self-check: the best window must be labeled BULL, the worst
    BEAR, and every |return| < 5% window SIDEWAYS. Guards against the original
    version's bull/bear-swapped display bug surviving any future refactor."""
    valid = ~np.isnan(returns)
    r, s = returns[valid], states[valid]
    if len(r) == 0: return False
    ok = True
    if (r >= 0.05).any(): ok &= (s[np.argmax(r)] == 1)
    if (r <= -0.05).any(): ok &= (s[np.argmin(r)] == 2)
    mid = np.abs(r) < 0.05
    if mid.any(): ok &= bool((s[mid] == 0).all())
    return bool(ok)

def _transition_matrix(states_seq):
    C = np.zeros((3, 3))
    for a, b in zip(states_seq[:-1], states_seq[1:]):
        C[a, b] += 1
    rowsum = C.sum(axis=1, keepdims=True)
    P = np.divide(C, rowsum, out=np.zeros_like(C), where=rowsum > 0)
    return C, P

def _markov_agent_signals(df, window=MARKOV_WINDOW, thr=MARKOV_THR, min_trans=10):
    """Walk-forward signals: at each stride point the matrix has only counted
    transitions that are already history (FIX 1: stride = window, never
    overlapping windows)."""
    close = df['Close'].values
    n = len(close)
    buy = np.zeros(n, dtype=bool); sell = np.zeros(n, dtype=bool)
    returns, states = _markov_states(close, window)
    pts = list(range(window, n, window))
    if len(pts) >= 2 and not _verify_state_labels(returns, states):
        return pd.Series(buy, df.index), pd.Series(sell, df.index)  # refuse to trade on bad labels
    C = np.zeros((3, 3))
    for k in range(1, len(pts)):
        prev_s, cur_s = states[pts[k-1]], states[pts[k]]
        C[prev_s, cur_s] += 1
        if C.sum() < min_trans: continue
        row = C[cur_s]
        if row.sum() < 3: continue
        p = row / row.sum()
        sig = p[1] - p[2]
        if sig > thr: buy[pts[k]] = True
        elif sig < -thr: sell[pts[k]] = True
    return pd.Series(buy, df.index), pd.Series(sell, df.index)

@st.cache_data(ttl=3600, show_spinner="Building Markov matrices...")
def markov_lab(ticker, window=MARKOV_WINDOW):
    """Both matrices for one ticker: overlapping (legacy, inflated) vs
    stride-sampled (true), plus the current walk-forward signal."""
    bulk = download_bulk(TICKERS)
    df = bulk[ticker].copy().dropna()
    close = df['Close'].values
    returns, states = _markov_states(close, window)
    valid = states[window:]
    _, P_ov = _transition_matrix(valid)                       # every bar: overlapping
    stride_states = states[list(range(window, len(close), window))]
    _, P_st = _transition_matrix(stride_states)               # FIX 1: stride-sampled
    labels_ok = _verify_state_labels(returns[window:], valid)
    cur = int(stride_states[-1])
    row = P_st[cur]
    signal = float(row[1] - row[2]) if row.sum() > 0 else 0.0
    stationary = np.linalg.matrix_power(P_st, 50)[cur]
    return P_ov, P_st, labels_ok, STATE_NAMES[cur], signal, stationary

def _agent_signals(aid, df):
    no_sig = pd.Series(False, index=df.index)
    F = lambda s: s.fillna(False).astype(bool)
    if aid == 'rsi2':
        buy = (df['Close'] > df['SMA_200']) & (df['RSI_2'] < 10) & (df['RSI_2'].shift(1) >= 10)
        return F(buy), no_sig
    if aid == 'trend':
        up = (df['EMA_50'] > df['EMA_200']) & (df['ADX_14'] > 20)
        dn = (df['EMA_50'] < df['EMA_200']) & (df['ADX_14'] > 20)
        buy = up & (df['Close'] > df['EMA_20']) & (df['Close'].shift(1) <= df['EMA_20'].shift(1))
        sell = dn & (df['Close'] < df['EMA_20']) & (df['Close'].shift(1) >= df['EMA_20'].shift(1))
        return F(buy), F(sell)
    if aid == 'turtle':
        buy = (df['Close'] > df['DC_HI']) & (df['Close'].shift(1) <= df['DC_HI'].shift(1)) & (df['EMA_50'] > df['EMA_200'])
        sell = (df['Close'] < df['DC_LO']) & (df['Close'].shift(1) >= df['DC_LO'].shift(1)) & (df['EMA_50'] < df['EMA_200'])
        return F(buy), F(sell)
    if aid == 'macd':
        m, s = df['MACD_12_26_9'], df['MACDs_12_26_9']
        buy = (m > s) & (m.shift(1) <= s.shift(1)) & (m < 0) & (df['Close'] > df['EMA_200'])
        sell = (m < s) & (m.shift(1) >= s.shift(1)) & (m > 0) & (df['Close'] < df['EMA_200'])
        return F(buy), F(sell)
    if aid == 'bband':
        buy = (df['Close'] < df['BB_LOW']) & (df['Close'].shift(1) >= df['BB_LOW'].shift(1)) & (df['Close'] > df['EMA_200'])
        return F(buy), no_sig
    if aid == 'house':
        return F(df['Buy_Signal']), F(df['Sell_Signal'])
    if aid == 'markov':
        return _markov_agent_signals(df)
    if aid == 'sniper':
        uptrend = (df['Close'] > df['SMA_200']) & (df['EMA_50'] > df['EMA_200'])
        panic = (df['RSI_2'] < 5) & (df['RSI_2'].shift(1) >= 5)
        dip = df['Close'] < df['EMA_20']
        down3 = ((df['Close'] < df['Close'].shift(1)) &
                 (df['Close'].shift(1) < df['Close'].shift(2)) &
                 (df['Close'].shift(2) < df['Close'].shift(3)))
        return F(uptrend & panic & dip & down3), no_sig
    return no_sig, no_sig

def _simulate_deal(df, ts, side, target_mult, stop_mult, max_hold=35):
    i = df.index.get_loc(ts)
    atr = df['ATR'].iloc[i]
    if pd.isna(atr) or atr <= 0 or i >= len(df) - 1: return None
    entry = df['Close'].iloc[i]
    tgt = entry + side * target_mult * atr
    stp = entry - side * stop_mult * atr
    exit_price = exit_ts = outcome = None
    end = min(len(df), i + 1 + max_hold)
    for j in range(i + 1, end):
        hi, lo = df['High'].iloc[j], df['Low'].iloc[j]
        # stop checked first: same-bar ambiguity counts against the agent
        if (lo <= stp) if side == 1 else (hi >= stp):
            exit_price, exit_ts, outcome = stp, df.index[j], 'LOSS'; break
        if (hi >= tgt) if side == 1 else (lo <= tgt):
            exit_price, exit_ts, outcome = tgt, df.index[j], 'WIN'; break
    if outcome is None:
        if end == len(df) and (len(df) - 1 - i) < max_hold:
            exit_price, exit_ts, outcome = df['Close'].iloc[-1], pd.NaT, 'OPEN'
        else:
            j = end - 1
            exit_price, exit_ts = df['Close'].iloc[j], df.index[j]
            outcome = 'TIME-WIN' if side * (exit_price - entry) > 0 else 'TIME-LOSS'
    pnl = side * (exit_price - entry) / entry * 100
    r = side * (exit_price - entry) / (stop_mult * atr)
    return {'SIDE': 'LONG' if side == 1 else 'SHORT', 'ENTRY TIME': df.index[i],
            'ENTRY': round(entry, 2), 'EXIT TIME': exit_ts, 'EXIT': round(exit_price, 2),
            'OUTCOME': outcome, 'PNL %': round(pnl, 2), 'R': round(r, 2)}

def _prepare_agent_frames(bulk, tickers):
    frames = {}
    for t in tickers:
        try:
            df = bulk[t].copy()
            if df.empty or len(df) < 300: continue
            df.dropna(inplace=True)
            if not df.empty:
                last_ts = df.index[-1]
                now = pd.Timestamp.now(tz=last_ts.tzinfo)
                if last_ts + pd.Timedelta(hours=1) > now:
                    df = df.iloc[:-1]
            if len(df) < 300: continue
            df = calculate_expert_strategy(df)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['EMA_20'] = ta.ema(df['Close'], length=20)
            df['SMA_200'] = df['Close'].rolling(200).mean()
            df['RSI_2'] = ta.rsi(df['Close'], length=2)
            adx = ta.adx(df['High'], df['Low'], df['Close'])
            if adx is not None: df = pd.concat([df, adx], axis=1)
            macd = ta.macd(df['Close'])
            if macd is not None: df = pd.concat([df, macd], axis=1)
            mid = df['Close'].rolling(20).mean(); sd = df['Close'].rolling(20).std()
            df['BB_LOW'] = mid - 2 * sd
            df['DC_HI'] = df['High'].rolling(55).max().shift(1)
            df['DC_LO'] = df['Low'].rolling(55).min().shift(1)
            frames[t] = df
        except Exception:
            continue
    return frames

def _run_agents(frames):
    trades = []
    for agent in AGENT_DEFS:
        sigs = []
        for t, df in frames.items():
            buy, sell = _agent_signals(agent['id'], df)
            for ts in df.index[buy]: sigs.append((ts, t, 1))
            for ts in df.index[sell]: sigs.append((ts, t, -1))
        sigs.sort(key=lambda x: (x[0], x[1]))
        per_day = {}
        for ts, t, side in sigs:
            day = ts.date()
            if per_day.get(day, 0) >= 2: continue  # max 2 deals per day
            rec = _simulate_deal(frames[t], ts, side, agent['target'], agent['stop'],
                                 max_hold=agent.get('hold', 35))
            if rec is None: continue
            per_day[day] = per_day.get(day, 0) + 1
            rec['AGENT'] = agent['name']; rec['TICKER'] = t
            trades.append(rec)
    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf = tdf.sort_values('ENTRY TIME').reset_index(drop=True)
    return tdf

@st.cache_data(ttl=3600, show_spinner="🤖 Agents are trading the past year...")
def build_agent_history(tickers):
    return _run_agents(_prepare_agent_frames(download_bulk(tickers), tickers))

# --- EMAIL ALERTS ---
# Subscriptions persist in a JSON file next to the app. SMTP credentials come
# from Streamlit secrets ([smtp] section) so nothing sensitive lives in code.
SUBS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subscriptions.json")

def _load_subs():
    try:
        with open(SUBS_FILE) as f: return json.load(f)
    except Exception: return {"subs": []}

def _save_subs(data):
    try:
        with open(SUBS_FILE, "w") as f: json.dump(data, f, indent=2, default=str)
    except Exception: pass

def _smtp_config():
    try:
        s = st.secrets["smtp"]
        return {'server': s['server'], 'port': int(s.get('port', 465)),
                'user': s['user'], 'password': s['password'],
                'sender': s.get('sender', s['user'])}
    except Exception:
        return None

def _send_email(cfg, to, subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject; msg['From'] = cfg['sender']; msg['To'] = to
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(cfg['server'], cfg['port'], context=ctx) as srv:
        srv.login(cfg['user'], cfg['password'])
        srv.sendmail(cfg['sender'], [to], msg.as_string())

def notify_subscribers(tdf):
    """Email each subscriber the deals their agent entered since the last
    notification. Runs on every Agents page load, so alerts go out whenever
    the hourly data refresh reveals new entries."""
    data = _load_subs()
    if not data.get('subs'): return
    cfg = _smtp_config()
    if cfg is None: return
    changed = False
    for sub in data['subs']:
        try:
            a = tdf[tdf['AGENT'] == sub['agent']]
            if a.empty: continue
            new = a[a['ENTRY TIME'] > pd.Timestamp(sub['last_notified'])]
            if new.empty: continue
            lines = [f"• {r['SIDE']} {r['TICKER']} @ {r['ENTRY']}  ({r['ENTRY TIME']})"
                     for _, r in new.sort_values('ENTRY TIME').iterrows()]
            _send_email(cfg, sub['email'],
                        f"🔔 {sub['agent']} entered {len(new)} new deal(s)",
                        f"{sub['agent']} — new deals:\n\n" + "\n".join(lines) +
                        "\n\nOpen the dashboard for exits and live status.\n— US Stocks Analyzer")
            sub['last_notified'] = str(new['ENTRY TIME'].max())
            changed = True
        except Exception:
            continue  # one bad subscription must not block the others
    if changed: _save_subs(data)

def _fmt_deals(deals):
    out = deals.copy()
    out['ENTRY TIME'] = out['ENTRY TIME'].dt.strftime('%Y-%m-%d %H:%M')
    out['EXIT TIME'] = out['EXIT TIME'].dt.strftime('%Y-%m-%d %H:%M').fillna('— open —')
    cols = ['AGENT', 'TICKER', 'SIDE', 'ENTRY TIME', 'ENTRY', 'EXIT TIME', 'EXIT', 'OUTCOME', 'PNL %', 'R']
    return out[[c for c in cols if c in out.columns]]

def render_agents_page():
    st.subheader("🤖 Trading Agents")
    st.caption(f"{len(AGENT_DEFS)} virtual traders, each running one published, widely-backtested strategy across "
               "the whole watchlist on **2 years** of hourly data. Each agent takes at most **2 deals per day** "
               "(first valid signals win) with ATR-scaled exits and a time exit. Same rules for every stock — "
               "nothing is tuned per ticker.")
    tdf = build_agent_history(TICKERS)
    if tdf.empty:
        st.warning("No agent history could be built — market data unavailable.")
        return

    notify_subscribers(tdf)

    if 'followed' not in st.session_state: st.session_state['followed'] = set()

    # --- DEALS FROM FOLLOWED AGENTS ---
    if st.session_state['followed']:
        st.markdown("#### ⭐ Deals from agents you follow")
        f = tdf[tdf['AGENT'].isin(st.session_state['followed'])]
        latest_day = tdf['ENTRY TIME'].max().normalize()
        actionable = f[(f['OUTCOME'] == 'OPEN') | (f['ENTRY TIME'] >= latest_day)]
        if actionable.empty:
            st.caption("No open or new deals from your followed agents right now.")
        else:
            st.dataframe(style_table(_fmt_deals(actionable.sort_values('ENTRY TIME', ascending=False)), ('SIDE',)),
                         use_container_width=True)
        st.divider()

    # --- PERIOD SELECTOR ---
    colp, coln = st.columns([3, 1])
    period = colp.selectbox("📅 History Period",
        ["Daily (last trading day)", "Weekly", "Monthly", "Custom (months)", "Yearly (all data)"], index=2)
    months = coln.number_input("Months", 1, 12, 3) if period == "Custom (months)" else None
    end_ts = tdf['ENTRY TIME'].max()
    if period.startswith("Daily"): cutoff = end_ts.normalize()
    elif period == "Weekly": cutoff = end_ts - pd.Timedelta(days=7)
    elif period == "Monthly": cutoff = end_ts - pd.Timedelta(days=30)
    elif period == "Custom (months)": cutoff = end_ts - pd.Timedelta(days=30 * months)
    else: cutoff = tdf['ENTRY TIME'].min()
    view = tdf[tdf['ENTRY TIME'] >= cutoff]

    # --- LEADERBOARD ---
    st.markdown("#### 🏆 Agent Leaderboard")
    rows = []
    for agent in AGENT_DEFS:
        a = view[view['AGENT'] == agent['name']]
        closed = a[a['OUTCOME'] != 'OPEN']
        wins = closed[closed['PNL %'] > 0]; losses = closed[closed['PNL %'] <= 0]
        wr = len(wins) / len(closed) * 100 if len(closed) else 0
        gross_loss = abs(losses['PNL %'].sum())
        pf = wins['PNL %'].sum() / gross_loss if gross_loss > 0 else float('inf')
        rows.append({'_pnl': closed['PNL %'].sum(),
                     'AGENT': agent['name'], 'STRATEGY': agent['strategy'], 'DEALS': len(a),
                     'WIN RATE': f"{wr:.0f}%",
                     'TOTAL P&L': f"{closed['PNL %'].sum():+.1f}%",
                     'AVG/DEAL': f"{closed['PNL %'].mean():+.2f}%" if len(closed) else "—",
                     'TOP LOSS': f"{closed['PNL %'].min():+.2f}%" if len(closed) else "—",
                     'PROFIT FACTOR': "∞" if pf == float('inf') else f"{pf:.2f}",
                     'FOLLOWING': '⭐' if agent['name'] in st.session_state['followed'] else ''})
    rows.sort(key=lambda r: r['_pnl'], reverse=True)
    lb = pd.DataFrame(rows).drop(columns=['_pnl'])
    st.dataframe(style_table(lb, ()), use_container_width=True)
    st.caption("P&L assumes equal capital per deal, summed per-deal returns (not compounded). "
               "Win rate and P&L count closed deals only; OPEN deals are excluded until they resolve.")

    # --- EQUITY CURVES ---
    figeq = go.Figure()
    for agent in AGENT_DEFS:
        a = view[(view['AGENT'] == agent['name']) & (view['OUTCOME'] != 'OPEN')].sort_values('EXIT TIME')
        if a.empty: continue
        figeq.add_trace(go.Scatter(x=a['EXIT TIME'], y=a['PNL %'].cumsum(),
                                   mode='lines', name=agent['name']))
    figeq.update_layout(height=400, template=plotly_template, paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                        title_text="Cumulative P&L per Agent (%)", title_font_color=text_color,
                        font=dict(color=text_color), margin=dict(l=10, r=10, t=50, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=text_color)))
    st.plotly_chart(figeq, use_container_width=True, key=f"eq_{plotly_template}")

    st.divider()

    # --- REVIEW A SINGLE AGENT ---
    st.markdown("#### 🔍 Review an Agent")
    sel = st.selectbox("Agent", [a['name'] for a in AGENT_DEFS], label_visibility="collapsed")
    agent = next(a for a in AGENT_DEFS if a['name'] == sel)
    st.markdown(f"**Strategy:** {agent['strategy']}")
    st.markdown(f"**Why it works:** {agent['source']}")
    st.markdown(f"**Exits:** target +{agent['target']}×ATR / stop −{agent['stop']}×ATR / "
                f"time exit after ~{agent.get('hold', 35) // 7} trading days")

    if agent['id'] == 'markov':
        with st.expander("🧮 Markov Lab — inspect the transition matrices for any ticker"):
            lab_tick = st.selectbox("Ticker", sorted(TICKERS), key="markov_lab_ticker")
            try:
                P_ov, P_st, labels_ok, cur_state, signal, stationary = markov_lab(lab_tick)
                if not labels_ok:
                    st.error("Label self-check FAILED for this ticker — matrices suppressed rather than "
                             "risk showing swapped bull/bear labels (FIX 2).")
                else:
                    st.caption("Labels verified ✓ — best window = BULL, worst = BEAR, |return| < 5% = SIDEWAYS (FIX 2).")
                    cm1, cm2 = st.columns(2)
                    cm1.markdown("**Overlapping (legacy — inflated)**")
                    cm1.dataframe(pd.DataFrame(P_ov, index=STATE_NAMES, columns=STATE_NAMES).round(2),
                                  use_container_width=True)
                    cm2.markdown("**Stride-sampled (true)**")
                    cm2.dataframe(pd.DataFrame(P_st, index=STATE_NAMES, columns=STATE_NAMES).round(2),
                                  use_container_width=True)
                    st.caption("⚠️ Only the stride-sampled matrix is statistically honest: consecutive overlapping "
                               "20-day windows share 19 days, which fakes persistence on the diagonal (FIX 1). "
                               "Stickiness = the diagonal of the matrix.")
                    st.markdown(f"Current state: **{cur_state}** — signal P(bull) − P(bear) = **{signal:+.2f}** "
                                f"(trades only beyond ±{MARKOV_THR})")
                    st.caption(f"50-window forecast converges to the stationary distribution "
                               f"{np.round(stationary, 2)} — long-horizon forecasts carry no signal.")
            except Exception:
                st.warning("No data available for this ticker.")

    follow_now = st.toggle("⭐ Follow this agent (their open & new deals appear at the top of this page)",
                           value=sel in st.session_state['followed'])
    if follow_now: st.session_state['followed'].add(sel)
    else: st.session_state['followed'].discard(sel)

    a = view[view['AGENT'] == sel]
    closed = a[a['OUTCOME'] != 'OPEN']
    wins = closed[closed['PNL %'] > 0]
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Deals", len(a))
    m2.metric("Win Rate", f"{(len(wins) / len(closed) * 100):.0f}%" if len(closed) else "—")
    m3.metric("Total P&L", f"{closed['PNL %'].sum():+.1f}%")
    m4.metric("Avg R", f"{closed['R'].mean():+.2f}" if len(closed) else "—")
    m5.metric("Top Loss", f"{closed['PNL %'].min():+.2f}%" if len(closed) else "—")
    m6.metric("Open Deals", len(a) - len(closed))

    if a.empty:
        st.caption("No deals for this agent in the selected period.")
    else:
        st.dataframe(style_table(_fmt_deals(a.sort_values('ENTRY TIME', ascending=False)), ('SIDE',)),
                     use_container_width=True)

    # --- EMAIL ALERTS FOR THIS AGENT ---
    st.markdown("##### 🔔 Email Alerts")
    subs_data = _load_subs()
    my_subs = [s for s in subs_data['subs'] if s['agent'] == sel]
    email_in = st.text_input("Get an email whenever this agent enters a stock",
                             placeholder="you@example.com", key=f"email_{agent['id']}")
    col_sub, col_unsub = st.columns(2)
    if col_sub.button("🔔 Subscribe", key=f"sub_{agent['id']}"):
        if email_in and "@" in email_in and "." in email_in.split("@")[-1]:
            already = any(s['email'].lower() == email_in.lower() for s in my_subs)
            if already:
                st.info(f"{email_in} is already subscribed to {sel}.")
            else:
                # baseline = latest existing entry, so only FUTURE deals trigger mail
                agent_deals = tdf[tdf['AGENT'] == sel]
                baseline = agent_deals['ENTRY TIME'].max() if not agent_deals.empty else tdf['ENTRY TIME'].max()
                subs_data['subs'].append({'email': email_in, 'agent': sel, 'last_notified': str(baseline)})
                _save_subs(subs_data)
                st.success(f"Subscribed! {email_in} will be emailed when {sel} enters a new deal.")
        else:
            st.error("Please enter a valid email address.")
    if my_subs:
        st.caption("Subscribed: " + ", ".join(s['email'] for s in my_subs))
        if col_unsub.button("🗑️ Unsubscribe all from this agent", key=f"unsub_{agent['id']}"):
            subs_data['subs'] = [s for s in subs_data['subs'] if s['agent'] != sel]
            _save_subs(subs_data)
            st.rerun()
    if _smtp_config() is None:
        st.warning("Email sending isn't configured yet. Add an `[smtp]` section (server, port, user, password, "
                   "sender) to your Streamlit secrets — see `.streamlit/secrets.toml.example`. Subscriptions are "
                   "saved now and start working the moment SMTP is configured.")
    else:
        st.caption("Alerts are checked on every data refresh (hourly cache) whenever the app is open or visited.")

# --- MAIN APP ---
st.title("📈 US Stocks Analyzer")

if 'chart_range' not in st.session_state: st.session_state['chart_range'] = '1M'

# --- SECTION ROUTING ---
if page == "🤖 Trading Agents":
    render_agents_page()
    st.stop()

data, df_summ, failed_tickers = get_data(TICKERS, target_mult, stop_mult)

# --- SCANNER ---
st.subheader("📡 Market Scanner")
# Average over stocks with at least one closed backtest trade
traded = [d for d in data.values() if d['tr'] > 0]
avg_wr = sum(d['wr'] for d in traded)/len(traded) if traded else 0
avg_edge = sum(d['edge'] for d in traded)/len(traded) if traded else 0
# Breakeven win rate for the chosen exit geometry: below this, a "high"
# win rate still loses money
breakeven = stop_mult / (stop_mult + target_mult) * 100
st.info(f"📊 **System Avg Win Rate:** {int(avg_wr)}%  |  **Breakeven:** {breakeven:.0f}%  |  **Avg Edge:** {avg_edge:+.2f} ATR/trade")
st.caption(f"Exits are volatility-scaled: target +{target_mult}×ATR, stop −{stop_mult}×ATR, same settings for every stock. "
           "When a candle touches both target and stop, it counts as a loss, so these win rates are a conservative floor. "
           "A win rate above breakeven means positive expectancy (Avg Edge > 0).")
if failed_tickers:
    st.warning(f"⚠️ {len(failed_tickers)} ticker(s) failed to load and are excluded from the scanner: {', '.join(failed_tickers)}")

st.dataframe(style_table(df_summ), use_container_width=True)

# --- SUGGESTED STOCKS FOR TODAY ---
st.subheader("🎯 Suggested Stocks for Today")
MIN_TRADES = 5
suggestions = []
for t, d in data.items():
    # Quality bar: proven edge only - win rate above breakeven, positive
    # expectancy, and enough closed trades that the number means something
    if d['tr'] < MIN_TRADES or d['wr'] <= breakeven or d['edge'] <= 0:
        continue
    last_row = d['df'].iloc[-1]
    score = len(d['interp_list']['Positive'])
    live_long = bool(last_row['Buy_Signal'])
    live_short = bool(last_row['Sell_Signal'])
    uptrend = last_row['UT_Trend'] == 1 and last_row['Close'] > last_row['EMA_200']

    reasons = []
    if live_long:
        action = "💎 LONG NOW"; reasons.append("live buy signal")
    elif live_short:
        action = "🩸 SHORT NOW"; reasons.append("live sell signal")
    elif uptrend and score >= 8:
        action = "👀 WATCH LONG"; reasons.append(f"strong uptrend ({score}/12 bullish)")
    else:
        continue

    pat = d['pattern']
    if pat and pat['sentiment'] == ('Neg' if live_short else 'Pos'):
        reasons.append(pat['name'])

    suggestions.append(((live_long or live_short, d['edge'], score), {
        "TICKER": t,
        "PRICE": f"{last_row['Close']:.2f}",
        "ACTION": action,
        "WIN RATE": f"{int(d['wr'])}% ({d['tr']})",
        "EDGE": f"{d['edge']:+.2f} ATR",
        "SCORE": f"{score}/12",
        "WHY": ", ".join(reasons),
    }))

suggestions.sort(key=lambda x: x[0], reverse=True)
df_sugg = pd.DataFrame([row for _, row in suggestions[:10]])
if df_sugg.empty:
    st.caption("No setups meet the quality bar right now (win rate above breakeven, positive edge, "
               f"≥{MIN_TRADES} closed trades, plus a live signal or strong uptrend). Sitting out is a position too.")
else:
    st.dataframe(style_table(df_sugg, ('ACTION',)), use_container_width=True)
    st.caption(f"Quality bar: win rate above breakeven ({breakeven:.0f}%), positive edge, ≥{MIN_TRADES} closed trades. "
               "Live signals rank above watchlist setups, then by edge. Top 10 shown.")

st.divider()

# --- DEEP DIVE ---
selected = st.selectbox("Select Stock", sorted(data.keys()))

if selected:
    d = data[selected]
    df = d['df']
    interp_list = d['interp_list']
    ct = d['chart_text']
    pat = d['pattern']
    last = df.iloc[-1]
    
    if last['Buy_Signal']: st.success(f"🚨 LIVE BUY SIGNAL: {selected}")
    elif last['Sell_Signal']: st.error(f"🚨 LIVE SELL SIGNAL: {selected}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"${last['Close']:.2f}")
    c2.metric("Win Rate", f"{int(d['wr'])}%", f"{d['tr']} Trades")
    c3.metric("Edge/Trade", f"{d['edge']:+.2f} ATR")
    c4.metric("Pos Indicators", f"{len(interp_list['Positive'])}")
    dec, _ = get_decision(df)
    c5.metric("Strategy", dec)

    # --- 1. TRADINGVIEW CHART ---
    st.markdown("### 🖼️ TradingView Chart")
    tv_theme_str = "dark" if is_dark else "light"
    components.html(f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%",
        "height": 500,
        "symbol": "{selected}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "{tv_theme_str}",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }}
      );
      </script>
    </div>
    """, height=500)
    
    st.divider()

    # --- 2. PYTHON STRATEGY CHART ---
    st.markdown("### 📊 Strategy Chart (UT Bot & Patterns)")
    
    # CHART CONTROLS
    st.write("#### 🔎 Zoom")
    cols = st.columns(5)
    ranges = {'1W': '1W', '1M': '1M', '3M': '3M', '6M': '6M', 'ALL': 'ALL'}
    for i, (k,v) in enumerate(ranges.items()):
        if cols[i].button(k): st.session_state['chart_range'] = v
        
    df_c = df.copy()
    if st.session_state['chart_range'] == '1W': df_c = df_c.tail(35)
    elif st.session_state['chart_range'] == '1M': df_c = df_c.tail(150)
    elif st.session_state['chart_range'] == '3M': df_c = df_c.tail(450)
    elif st.session_state['chart_range'] == '6M': df_c = df_c.tail(900)
    elif st.session_state['chart_range'] == 'ALL': df_c = df_c # All Data
    
    idx = df_c.index.strftime('%Y-%m-%d %H:%M')
    
    # --- HELPER FOR CHART LAYOUT ---
    def update_fig(figure, title_txt=""):
        final_title = "" if title_txt == "Main" else title_txt
        figure.update_xaxes(type='category', nticks=10, showgrid=False)
        figure.update_layout(
            height=300 if title_txt != "Main" else 600,
            template=plotly_template,
            paper_bgcolor=bg_color, # Force Background
            plot_bgcolor=bg_color,  # Force Plot Area
            title_text=final_title,
            title_font_color=text_color,
            font=dict(color=text_color), # Legend/Axis Text Color Force
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="left", 
                x=0, 
                font=dict(color=text_color)
            )
        )
        return figure

    # MAIN CHART
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=idx, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name='Price'), row=1, col=1)
    
    # User-Colored Lines
    fig.add_trace(go.Scatter(x=idx, y=df_c['EMA_200'], line=dict(color=c_ema200, width=2), name='EMA 200'), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=df_c['EMA_50'], line=dict(color=c_ema50, width=1), name='EMA 50'), row=1, col=1)
    
    stop_g = df_c['Trailing_Stop'].where(df_c['UT_Trend']==1)
    stop_r = df_c['Trailing_Stop'].where(df_c['UT_Trend']==-1)
    fig.add_trace(go.Scatter(x=idx, y=stop_g, line=dict(color=c_ut_buy), name='UT Buy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=stop_r, line=dict(color=c_ut_sell), name='UT Sell'), row=1, col=1)
    
    # Signals
    buys = df_c[df_c['Buy_Signal']]
    sells = df_c[df_c['Sell_Signal']]
    if not buys.empty: fig.add_trace(go.Scatter(x=buys.index.strftime('%Y-%m-%d %H:%M'), y=buys['Low']*0.99, mode='markers', marker=dict(symbol='diamond', size=12, color=c_sig_buy), name='BUY'), row=1, col=1)
    if not sells.empty: fig.add_trace(go.Scatter(x=sells.index.strftime('%Y-%m-%d %H:%M'), y=sells['High']*1.01, mode='markers', marker=dict(symbol='diamond', size=12, color=c_sig_sell), name='SELL'), row=1, col=1)
    
    # Pattern Annotation (Main Chart)
    if show_patterns and pat:
        try:
            # Check if pattern index is in the current zoomed view
            pat_idx_str = pat['index'].strftime('%Y-%m-%d %H:%M')
            if pat_idx_str in idx.values:
                # Sentiment Color
                sent_color = "#00FF00" if pat['sentiment'] == "Pos" else "#FF0000"
                
                fig.add_annotation(
                    x=pat_idx_str, y=pat['price'],
                    text=f"<b>{pat['name']}</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=sent_color,
                    bgcolor=bg_color,
                    bordercolor=sent_color,
                    font=dict(color=sent_color)
                )
        except: pass

    fig.add_trace(go.Scatter(x=idx, y=df_c['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', row=2, col=1); fig.add_hline(y=70, line_dash='dash', row=2, col=1)
    
    update_fig(fig, "Main")
    st.plotly_chart(fig, use_container_width=True, key=f"main_{plotly_template}_{selected}")
    
    # --- FULL INDICATOR GRID (Consolidated 6 Rows) ---
    st.markdown("### 📊 Advanced Technicals")
    
    # Row 1: Stoch RSI | Ichimoku
    r1a, r1b = st.columns(2)
    with r1a:
        f = go.Figure()
        f.add_trace(go.Scatter(x=idx, y=df_c['Stoch_K'], name='K', line=dict(color='cyan')))
        f.add_trace(go.Scatter(x=idx, y=df_c['Stoch_D'], name='D', line=dict(color='orange')))
        f.add_hline(y=20, line_color='green'); f.add_hline(y=80, line_color='red')
        st.plotly_chart(update_fig(f, "Stochastic RSI"), use_container_width=True, key=f"st_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('STOCH', 'N/A')}")
    with r1b:
        f = go.Figure()
        if 'ISA_9' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['Close'], name='Close'))
            f.add_trace(go.Scatter(x=idx, y=df_c['ISA_9'], name='A', line=dict(width=0), showlegend=False))
            f.add_trace(go.Scatter(x=idx, y=df_c['ISB_26'], name='B', fill='tonexty', fillcolor='rgba(0,0,250,0.1)', line=dict(width=0), showlegend=False))
        st.plotly_chart(update_fig(f, "Ichimoku Cloud"), use_container_width=True, key=f"ich_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('ICHI', 'N/A')}")

    # Row 2: VWAP | SMA/EMA Trends
    r2a, r2b = st.columns(2)
    with r2a:
        f = go.Figure()
        if 'VWAP' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['Close'], name='Price'))
            f.add_trace(go.Scatter(x=idx, y=df_c['VWAP'], name='VWAP', line=dict(color='orange')))
        st.plotly_chart(update_fig(f, "VWAP"), use_container_width=True, key=f"vw_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('VWAP', 'N/A')}")
    with r2b:
        f = go.Figure()
        f.add_trace(go.Scatter(x=idx, y=df_c['Close'], name='Close', line=dict(color='gray', width=1)))
        f.add_trace(go.Scatter(x=idx, y=df_c['SMA_20'], name='SMA20'))
        f.add_trace(go.Scatter(x=idx, y=df_c['SMA_50'], name='SMA50'))
        f.add_trace(go.Scatter(x=idx, y=df_c['EMA_200'], name='EMA200', line=dict(color='yellow')))
        st.plotly_chart(update_fig(f, "SMA & EMA Trends"), use_container_width=True, key=f"ma_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('EMA', 'N/A')}")

    # Row 3: ATR | CCI
    r3a, r3b = st.columns(2)
    with r3a:
        f = go.Figure()
        f.add_trace(go.Scatter(x=idx, y=df_c['ATR'], name='ATR', line=dict(color='cyan')))
        st.plotly_chart(update_fig(f, "ATR (Volatility)"), use_container_width=True, key=f"at_{plotly_template}")
        st.caption("ℹ️ **Note:** High ATR = High Volatility")
    with r3b:
        f = go.Figure()
        if 'CCI' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['CCI'], name='CCI', line=dict(color='magenta')))
            f.add_hline(y=100, line_dash='dot'); f.add_hline(y=-100, line_dash='dot')
        st.plotly_chart(update_fig(f, "CCI"), use_container_width=True, key=f"cc_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('CCI', 'N/A')}")

    # Row 4: CMF | OBV
    r4a, r4b = st.columns(2)
    with r4a:
        f = go.Figure()
        if 'CMF' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['CMF'], name='CMF', fill='tozeroy'))
            f.add_hline(y=0)
        st.plotly_chart(update_fig(f, "Chaikin Money Flow"), use_container_width=True, key=f"cm_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('CMF', 'N/A')}")
    with r4b:
        f = go.Figure()
        if 'OBV' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['OBV'], name='OBV'))
        st.plotly_chart(update_fig(f, "On-Balance Volume"), use_container_width=True, key=f"ob_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('OBV', 'N/A')}")

    # Row 5: MACD | ADX
    r5a, r5b = st.columns(2)
    with r5a:
        f = go.Figure()
        if 'MACD_12_26_9' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['MACD_12_26_9'], name='MACD'))
            f.add_trace(go.Scatter(x=idx, y=df_c['MACDs_12_26_9'], name='Sig'))
            f.add_bar(x=idx, y=df_c['MACDh_12_26_9'], name='Hist')
        st.plotly_chart(update_fig(f, "MACD"), use_container_width=True, key=f"md_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('MACD', 'N/A')}")
    with r5b:
        f = go.Figure()
        if 'ADX_14' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['ADX_14'], name='ADX', line=dict(color='white')))
            f.add_trace(go.Scatter(x=idx, y=df_c['DMP_14'], name='DI+', line=dict(color='green')))
            f.add_trace(go.Scatter(x=idx, y=df_c['DMN_14'], name='DI-', line=dict(color='red')))
        st.plotly_chart(update_fig(f, "ADX & DI"), use_container_width=True, key=f"ad_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('ADX', 'N/A')}")

    # Row 6: ADL (Full Width)
    st.markdown("---") # Visual separator
    f = go.Figure()
    if 'ADL' in df_c.columns:
        f.add_trace(go.Scatter(x=idx, y=df_c['ADL'], name='ADL', line=dict(color='cyan')))
    st.plotly_chart(update_fig(f, "Accumulation/Distribution"), use_container_width=True, key=f"adl_{plotly_template}")
    st.caption(f"**Interpretation:** {ct.get('ADL', 'N/A')}")

    # --- INDICATOR ANALYSIS TABLE ---
    st.markdown("### ⚖️ Comprehensive Indicator Analysis")
    c_pos, c_neu, c_neg = st.columns(3)
    
    with c_pos:
        st.success("✅ POSITIVE")
        for i in interp_list['Positive']: st.write(f"• {i}")
        if not interp_list['Positive']: st.caption("None")
        
    with c_neu:
        st.warning("🟡 NEUTRAL")
        for i in interp_list['Neutral']: st.write(f"• {i}")
        if not interp_list['Neutral']: st.caption("None")
        
    with c_neg:
        st.error("❌ NEGATIVE")
        for i in interp_list['Negative']: st.write(f"• {i}")
        if not interp_list['Negative']: st.caption("None")
