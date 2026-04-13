
# NIFTY50 Index Trading AI Prediction on World Model with AI Assitant

An advanced **AI-powered trading decision system** for the NIFTY50 index, combining:

* 📊 Technical Indicators
* 🧠 Machine Learning
* 🌍 World Model Simulation
* 📈 Multi-Timeframe Analysis
* 🗣️ Conversational AI Interface

---

## 🚀 Key Features

### 📊 1. Real-Time Market Analysis

* Live NIFTY price (Yahoo Finance based)
* Auto-refresh dashboard
* Multiple intervals: `1m, 5m, 15m, 30m, 1h, 2h, 4h`

---

### 🧠 2. Multi-Timeframe (MTF) Signals

* 15m, 1h, 4h analysis
* Consensus-based BUY / SELL / HOLD signal

---

### 📉 3. Technical Indicators

* RSI (Relative Strength Index)
* MACD (Trend momentum)
* EMA (12 & 26)

---

### 🧠 4. Smart Market Intelligence

* Trend Detection (Uptrend / Downtrend / Sideways)
* Liquidity Zones
* Market Structure (Break of Structure - BOS)

---

### 📊 5. Smart Support & Resistance (Improved)

* Swing-based levels (not simple rolling averages)
* Trend-aware adjustments
* More realistic trading zones

---

### 🌍 6. World Model (Core Innovation)

Simulates future market behavior:

* Monte Carlo simulation of price movement
* Evaluates:

  * BUY
  * SELL
  * HOLD
* Selects **best action based on expected outcome**

---

### 🤖 7. Machine Learning Prediction

* Linear Regression model
* Predicts next price movement
* Calculates expected move

---

### 🧠 8. AI Decision Engine

Combines:

* MTF signals
* ML prediction
* Market structure
* Breakout detection

👉 Outputs final trading signal:

* STRONG BUY 🚀
* STRONG SELL 🔻
* HOLD

---

### 🗣️ 9. AI Chat Assistant

Ask questions like:

* “Should I buy?”
* “What is the risk?”
* “Why sell?”

👉 System explains decisions using live data + simulation

---

## ⚙️ How It Works

```text
Yahoo Finance → Market Data
        ↓
Indicators + ML + Structure
        ↓
World Model Simulation
        ↓
Best Action Selection
        ↓
AI Explanation (Chat Interface)
```

---

## 📦 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/nifty-ai-trading.git
cd nifty-ai-trading
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run App

```bash
streamlit run app.py
```

---

## 📁 Requirements

* Python 3.9+
* Streamlit
* Pandas
* NumPy
* yfinance
* Plotly
* scikit-learn

---

## ⚠️ Important Notes

### ❗ Yahoo Finance Limitation

* Data may be **delayed or cached**
* Not suitable for **real-time trading execution**

👉 For production:

* Use Zerodha Kite API (recommended)

---

### ❗ Simulation Disclaimer

* World Model uses probabilistic simulation
* Results are **not guaranteed**
* Use for **decision support only**

---

## 📊 Supported Modes

| Mode               | Description         |
| ------------------ | ------------------- |
| Intraday Predictor | ML-based prediction |
| Decision Assistant | Signal + reasoning  |
| Strategy Simulator | Chart + simulation  |

---

## 🧠 Future Improvements

* 🔌 Zerodha WebSocket integration (real-time ticks)
* 📊 Order Block & Fair Value Gap detection
* 🎯 Risk-Reward & Position Sizing
* 📈 Backtesting engine
* ☁️ Cloud optimization

---

## 👨‍💻 Author

**Neeraj Bhatia**

---

## 📜 License

MIT License

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub and share with others!

---

## 🚀 Vision

To build:

> 🧠 **Human-Alike AI Reasoning Search System for Financial Markets**

Combining:

* Simulation (World Model)
* Reasoning (AI)
* Real-time decision-making

---

**⚡ Built with passion for AI + Trading**

---




