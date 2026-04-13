import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="AI Trading System", layout="wide")

# =============================
# SIDEBAR
# =============================
st.sidebar.title("⚙️ Settings")

interval = st.sidebar.selectbox(
    "Interval", ["1m", "5m", "15m", "30m", "1h", "2h", "4h"], index=2
)

period = st.sidebar.selectbox(
    "Period", ["1d", "5d", "1mo", "3mo"], index=2
)

refresh_interval = st.sidebar.slider("Auto Refresh (seconds)", 5, 60, 10)

target_price = st.sidebar.number_input("🔔 Price Alert", value=0.0)
rsi_alert = st.sidebar.slider("RSI Alert Level", 10, 90, 70)

mode = st.sidebar.selectbox(
    "Mode", ["Intraday Predictor", "Decision Assistant", "Strategy Simulator"]
)

st_autorefresh(interval=refresh_interval * 1000, key="refresh")

# =============================
# DATA FETCH
# =============================
@st.cache_data
def get_data(interval, period):
    base_interval = interval
    resample = None

    if interval == "2h":
        base_interval = "1h"
        resample = "2H"
    elif interval == "4h":
        base_interval = "1h"
        resample = "4H"

    df = yf.download("^NSEI", interval=base_interval, period=period, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    if resample:
        df.set_index("Datetime", inplace=True)
        df = df.resample(resample).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()
        df.reset_index(inplace=True)

    df.rename(columns={"Close": "close"}, inplace=True)
    return df

data = get_data(interval, period)

if data.empty:
    st.error("No data")
    st.stop()

# =============================
# LIVE PRICE
# =============================
def get_live_price():
    try:
        df = yf.download("^NSEI", period="1d", interval="1m", progress=False)

        if df.empty or "Close" not in df.columns:
            return None

        df.dropna(inplace=True)

        if df.empty:
            return None

        return float(df["Close"].iloc[-1])

    except:
        return None


# 🔥 ALWAYS DEFINE VARIABLE
live_price = get_live_price()

# 🔥 FALLBACK (CRITICAL)
if live_price is None:
    live_price = float(data['close'].iloc[-1])

# =============================
# INDICATORS
# =============================
data['Return'] = data['close'].pct_change()

def compute_rsi(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = compute_rsi(data)
data['EMA12'] = data['close'].ewm(span=12).mean()
data['EMA26'] = data['close'].ewm(span=26).mean()
data['MACD'] = data['EMA12'] - data['EMA26']

data.dropna(inplace=True)

latest_rsi = float(data['RSI'].iloc[-1])
latest_macd = float(data['MACD'].iloc[-1])



# =============================
# TREND ZONE
# =============================
def detect_trend(df):
    df['HH'] = df['High'].rolling(10).max()
    df['LL'] = df['Low'].rolling(10).min()

    last_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-5]

    if last_price > prev_price:
        return "UPTREND 📈"
    elif last_price < prev_price:
        return "DOWNTREND 📉"
    else:
        return "SIDEWAYS ➖"

trend = detect_trend(data)

# =============================
# LIQUIDITY ZONES
# =============================
def detect_liquidity(df):
    highs = df['High'].rolling(5).max()
    lows = df['Low'].rolling(5).min()

    eq_high = highs.iloc[-1]
    eq_low = lows.iloc[-1]

    return eq_high, eq_low

liq_high, liq_low = detect_liquidity(data)

# =============================
# SMART MONEY CONCEPTS
# =============================
def detect_structure(df):
    recent_high = df['High'].iloc[-5:].max()
    recent_low = df['Low'].iloc[-5:].min()
    price = df['close'].iloc[-1]

    if price > recent_high:
        return "BOS ↑ (Bullish Break)"
    elif price < recent_low:
        return "BOS ↓ (Bearish Break)"
    else:
        return "No Break"

structure = detect_structure(data)


# =============================
# SUPPORT & RESISTANCE
# =============================
# =============================
# SUPPORT & RESISTANCE (SMART LOGIC)
# =============================
def get_support_resistance(df):

    highs = df['High']
    lows = df['Low']

    # 🔹 Swing highs & lows
    swing_highs = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    swing_lows = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]

    # 🔹 Default fallback
    resistance = float(highs.iloc[-1])
    support = float(lows.iloc[-1])

    if not swing_highs.empty:
        resistance = float(swing_highs.tail(3).mean())

    if not swing_lows.empty:
        support = float(swing_lows.tail(3).mean())

    # 🔹 Trend adjustment (VERY IMPORTANT)
    if "DOWN" in trend:
        resistance = max(resistance, highs.tail(5).max())
        support = lows.tail(10).min()

    elif "UP" in trend:
        support = min(support, lows.tail(5).min())
        resistance = highs.tail(10).max()

    # 🔹 Add back to dataframe (so chart still works)
    df['support'] = support
    df['resistance'] = resistance

    return df


# SAME CALL (DO NOT CHANGE)
data = get_support_resistance(data)

latest_support = float(data['support'].iloc[-1])
latest_resistance = float(data['resistance'].iloc[-1])


# =============================
# BREAKOUT DETECTION
# =============================
breakout = None

if live_price > latest_resistance:
    breakout = "BREAKOUT UP 🚀"

elif live_price < latest_support:
    breakout = "BREAKDOWN 🔻"

else:
    breakout = "RANGE"
    


# =============================
# ML MODEL
# =============================
ml_data = data.copy()

ml_data['Target'] = ml_data['close'].shift(-1)
ml_data = ml_data.dropna()

# 🔥 SAFETY CHECK
if ml_data.empty or len(ml_data) < 20:
    predicted_price = live_price
    expected_move = 0
else:
    X = ml_data[['close', 'RSI', 'MACD']]
    y = ml_data['Target']

    model = LinearRegression()
    model.fit(X, y)

    predicted_price = model.predict([[live_price, latest_rsi, latest_macd]])[0]
    expected_move = predicted_price - live_price
    
    
    # =============================
# 🌍 WORLD MODEL (ACTION SIMULATION)
# =============================
def simulate_action(data, action, steps=20):

    returns = data['close'].pct_change().dropna().values.flatten()

    if len(returns) < 10:
        return 0, 0

    simulations = []

    for _ in range(50):  # 50 scenarios
        future_price = live_price

        sampled_returns = np.random.choice(returns, size=steps)

        for r in sampled_returns:
            future_price *= (1 + float(r))

        if action == "BUY":
            pnl = future_price - live_price

        elif action == "SELL":
            pnl = live_price - future_price

        else:  # HOLD
            pnl = (future_price - live_price) * 0.1  # minimal exposure

        simulations.append(pnl)

    # Expected outcome + risk
    avg_pnl = np.mean(simulations)
    risk = np.std(simulations)

    return avg_pnl, risk


# =============================
# SIMULATE ALL ACTIONS
# =============================
buy_pnl, buy_risk = simulate_action(data, "BUY")
sell_pnl, sell_risk = simulate_action(data, "SELL")
hold_pnl, hold_risk = simulate_action(data, "HOLD")



# =============================
# 🧠 WORLD MODEL DECISION
# =============================
actions = {
    "BUY": buy_pnl,
    "SELL": sell_pnl,
    "HOLD": hold_pnl
}

best_action = max(actions, key=actions.get)

# Confidence based on difference
confidence_score = abs(max(actions.values()) - np.mean(list(actions.values())))
confidence_pct = min(100, confidence_score * 5)


    
    
    
    
    
    
    
    
# =============================
# MTF SYSTEM
# =============================
def generate_signal(df):
    df['EMA12'] = df['close'].ewm(span=12).mean()
    df['EMA26'] = df['close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)

    rsi = float(df['RSI'].iloc[-1])
    macd = float(df['MACD'].iloc[-1])

    if macd > 0 and rsi < 70:
        return "BUY"
    elif macd < 0 and rsi > 30:
        return "SELL"
    return "HOLD"

def get_mtf_data():
    df_15m = yf.download("^NSEI", interval="15m", period="5d")
    df_1h = yf.download("^NSEI", interval="1h", period="1mo")
    df_4h = yf.download("^NSEI", interval="1h", period="3mo")

    def clean(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        df.rename(columns={"Close": "close"}, inplace=True)
        return df

    return clean(df_15m), clean(df_1h), clean(df_4h)

df_15m, df_1h, df_4h = get_mtf_data()

signal_15m = generate_signal(df_15m.copy())
signal_1h = generate_signal(df_1h.copy())
signal_4h = generate_signal(df_4h.copy())

signals = [signal_15m, signal_1h, signal_4h]

buy_count = signals.count("BUY")
sell_count = signals.count("SELL")

if buy_count >= 2:
    mtf_signal, mtf_color = "BUY", "green"
elif sell_count >= 2:
    mtf_signal, mtf_color = "SELL", "red"
else:
    mtf_signal, mtf_color = "HOLD", "orange"


# =============================
# SUPER AI SIGNAL
# =============================
if breakout == "BREAKOUT UP 🚀" and trend.startswith("UP"):
    signal = "STRONG BUY 🚀"

elif breakout == "BREAKDOWN 🔻" and trend.startswith("DOWN"):
    signal = "STRONG SELL 🔻"

elif structure.startswith("BOS ↑"):
    signal = "BUY (Structure Break)"

elif structure.startswith("BOS ↓"):
    signal = "SELL (Structure Break)"

elif predicted_price > live_price:
    signal = "BUY"

elif predicted_price < live_price:
    signal = "SELL"

else:
    signal = "HOLD"


# =============================
# ENHANCED SIGNAL (WITH BREAKOUT)
# =============================
if breakout == "BREAKOUT UP 🚀":
    signal = "STRONG BUY"
    color = "green"

elif breakout == "BREAKDOWN 🔻":
    signal = "STRONG SELL"
    color = "red"

elif predicted_price > live_price and latest_rsi < 70:
    signal = "BUY"
    color = "green"

elif predicted_price < live_price and latest_rsi > 30:
    signal = "SELL"
    color = "red"

else:
    signal = "HOLD"
    color = "orange"



# =============================
# UI
# =============================
st.title("🧠 NIFTY50 Index Trading AI Prediction on World Model")
st.metric("📈 Live NIFTY", round(live_price, 2))

# MTF
st.markdown("## 🧠 Multi-Timeframe Signal")
c1, c2, c3 = st.columns(3)
c1.metric("15m", signal_15m)
c2.metric("1h", signal_1h)
c3.metric("4h", signal_4h)

st.markdown(f"### 📊 Final MTF Signal: :{mtf_color}[{mtf_signal}]")

# MAIN SIGNAL
if predicted_price > live_price and latest_rsi < 70:
    signal, color = "BUY", "green"
elif predicted_price < live_price and latest_rsi > 30:
    signal, color = "SELL", "red"
else:
    signal, color = "HOLD", "orange"

st.markdown(f"## 📊 Signal: :{color}[{signal}]")

st.write(f"💰 Expected Move: {round(expected_move,2)} pts")


st.markdown("## 📊 Support & Resistance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Support", round(latest_support, 2))

with col2:
    st.metric("Resistance", round(latest_resistance, 2))

st.write(f"📡 Market State: {breakout}")


st.markdown("## 🧠 Market Intelligence")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Trend", trend)

with col2:
    st.metric("Liquidity High", round(liq_high,2))

with col3:
    st.metric("Liquidity Low", round(liq_low,2))

st.write(f"🏗️ Market Structure: {structure}")



# =============================
# 🌍 WORLD MODEL OUTPUT
# =============================
st.markdown("## 🌍 AI World Model Decision")

col1, col2, col3 = st.columns(3)

col1.metric("BUY (Simulated)", round(buy_pnl,2))
col2.metric("SELL (Simulated)", round(sell_pnl,2))
col3.metric("HOLD (Simulated)", round(hold_pnl,2))

st.markdown(f"### 🧠 Best Action: **{best_action}**")
st.metric("Confidence (Simulation)", f"{round(confidence_pct,1)}%")






# CHART
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data['Datetime'], 
    y=data['close'], 
    name="Price"
))

# Support line
fig.add_trace(go.Scatter(
    x=data['Datetime'],
    y=data['support'],
    name="Support",
    line=dict(dash='dot', color='green')
))

# Resistance line
fig.add_trace(go.Scatter(
    x=data['Datetime'],
    y=data['resistance'],
    name="Resistance",
    line=dict(dash='dot', color='red')
))

# Liquidity zones
fig.add_hline(y=liq_high, line_dash="dash", line_color="red")
fig.add_hline(y=liq_low, line_dash="dash", line_color="green")

fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# ALERTS
if target_price > 0 and live_price >= target_price:
    st.error(f"🚨 Price crossed {target_price}")

# MODES
if mode == "Intraday Predictor":
    st.write(f"Predicted Price: {round(predicted_price,2)}")

elif mode == "Strategy Simulator":
    st.line_chart(data['close'])
    
    
# =============================
# 🗣️ LLM-LIKE CHAT INTERFACE
# =============================
# =============================
# 🗣️ AI CHAT (FIXED)
# =============================
st.markdown("## 🗣️ AI Trading Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_query" not in st.session_state:
    st.session_state.last_query = None

user_input = st.text_input("Ask anything...", key="user_input")

# ✅ PROCESS ONLY NEW QUERY
if user_input and user_input != st.session_state.last_query:

    query = user_input.lower()

    if "buy" in query:
        response = f"""
🟢 BUY ANALYSIS:

Best Action: {best_action}
Expected Profit: {round(buy_pnl,2)}
Risk: {round(buy_risk,2)}
"""

    elif "sell" in query:
        response = f"""
🔴 SELL ANALYSIS:

Expected Profit: {round(sell_pnl,2)}
Risk: {round(sell_risk,2)}
"""

    else:
        response = f"""
🤖 Market Summary:

Best Action: {best_action}
Expected Move: {round(expected_move,2)}
"""

    # Save once
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", response))

    # ✅ Store query
    st.session_state.last_query = user_input

# DISPLAY CHAT
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.write(f"🧑 {msg}")
    else:
        st.write(f"🤖 {msg}")
