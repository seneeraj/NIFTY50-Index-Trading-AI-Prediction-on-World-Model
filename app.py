import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# =============================
# SIDEBAR
# =============================
st.sidebar.title("⚙️ Control Panel")

interval = st.sidebar.selectbox(
    "Interval", ["5m", "15m", "30m", "1h"], index=1
)

period = st.sidebar.selectbox(
    "Period", ["1d", "5d", "1mo", "3mo"], index=1
)

mode = st.sidebar.selectbox(
    "Mode", ["Intraday", "MTF"], index=0
)

# =============================
# FETCH DATA
# =============================
@st.cache_data(ttl=60)
def fetch_data(interval, period):
    df = yf.download("^NSEI", interval=interval, period=period, progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    df = df.reset_index()
    df.rename(columns={"Close": "close"}, inplace=True)

    return df

data = fetch_data(interval, period)

if data.empty:
    st.error("No data available")
    st.stop()

# =============================
# LIVE PRICE (FIXED)
# =============================
# Use latest candle instead of unreliable API
live_price = float(data['close'].iloc[-1])

# =============================
# SMART SUPPORT / RESISTANCE
# =============================
def swing_levels(df):

    highs = df['High']
    lows = df['Low']

    swing_high = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    swing_low = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]

    resistance = float(swing_high.tail(3).mean()) if not swing_high.empty else highs.iloc[-1]
    support = float(swing_low.tail(3).mean()) if not swing_low.empty else lows.iloc[-1]

    return support, resistance

support, resistance = swing_levels(data)

# =============================
# SIGNAL FUNCTION
# =============================
def generate_signal(df):

    if len(df) < 30:
        return "HOLD"

    df = df.copy()

    df['EMA12'] = df['close'].ewm(span=12).mean()
    df['EMA26'] = df['close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df = df.dropna()

    if df.empty:
        return "HOLD"

    macd = float(df['MACD'].iloc[-1])
    rsi = float(df['RSI'].iloc[-1])

    if macd > 0 and rsi < 70:
        return "BUY"
    elif macd < 0 and rsi > 30:
        return "SELL"
    else:
        return "HOLD"

# =============================
# MTF LOGIC
# =============================
if mode == "MTF":
    d1 = fetch_data("15m", "5d")
    d2 = fetch_data("1h", "1mo")
    d3 = fetch_data("1h", "3mo")

    s1 = generate_signal(d1)
    s2 = generate_signal(d2)
    s3 = generate_signal(d3)

    signals = [s1, s2, s3]

    if signals.count("BUY") >= 2:
        final_signal = "STRONG BUY"
    elif signals.count("SELL") >= 2:
        final_signal = "STRONG SELL"
    else:
        final_signal = "HOLD"
else:
    final_signal = generate_signal(data)

# =============================
# WORLD MODEL (SIMULATION)
# =============================
def simulate_action(data, action, steps=20):

    returns = data['close'].pct_change().dropna().values.flatten()

    if len(returns) < 10:
        return 0, 0

    simulations = []

    for _ in range(100):
        future = live_price
        sampled = np.random.choice(returns, steps)

        for r in sampled:
            future *= (1 + float(r))

        if action == "BUY":
            pnl = future - live_price
        elif action == "SELL":
            pnl = live_price - future
        else:
            pnl = (future - live_price) * 0.1

        simulations.append(pnl)

    return np.mean(simulations), np.std(simulations)

buy_pnl, buy_risk = simulate_action(data, "BUY")
sell_pnl, sell_risk = simulate_action(data, "SELL")
hold_pnl, hold_risk = simulate_action(data, "HOLD")

# =============================
# BEST ACTION
# =============================
actions = {"BUY": buy_pnl, "SELL": sell_pnl, "HOLD": hold_pnl}
best_action = max(actions, key=actions.get)

confidence = min(100, abs(max(actions.values())))

# =============================
# UI
# =============================
st.title("🧠 AI Trading World Model")

c1, c2, c3, c4 = st.columns(4)

c1.metric("📊 Signal", final_signal)
c2.metric("🎯 Confidence", f"{round(confidence,1)}%")
c3.metric("💰 BUY", round(buy_pnl,2))
c4.metric("💰 SELL", round(sell_pnl,2))

st.metric("💰 HOLD", round(hold_pnl,2))

c5, c6 = st.columns(2)
c5.metric("🟢 Support", round(support,2))
c6.metric("🔴 Resistance", round(resistance,2))

# =============================
# CHART
# =============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data['Datetime'],
    y=data['close'],
    name="Price"
))

fig.add_hrect(y0=support-20, y1=support+20, fillcolor="green", opacity=0.2)
fig.add_hrect(y0=resistance-20, y1=resistance+20, fillcolor="red", opacity=0.2)

fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# =============================
# DECISION
# =============================
st.markdown("## 🌍 World Model Decision")

st.success(f"👉 Best Action: {best_action}")
