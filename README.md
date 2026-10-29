# AI Trading Assistant 📈

A deep learning and quantitative trading framework featuring multivariate LSTM forecasting, technical indicator feature engineering, algorithmic signal generation, and an interactive Streamlit dashboard.

---

## 🚀 Key Features

- **Multivariate Time-Series Modeling:** 2-layer PyTorch LSTM with dropout (`0.2`) and non-linear MLP projection head trained on 20-day historical sequences.
- **Quantitative Feature Engineering:** Integrates Close price with technical indicators:
  - **RSI (Relative Strength Index):** 14-period momentum evaluation.
  - **20-day Moving Average (MA):** Short-term trend detection.
  - **MACD (Moving Average Convergence Divergence):** Trend following and momentum.
- **Hybrid Strategy Engine:** Combines predicted price direction ($\Delta = \text{pred}_t - \text{pred}_{t-1}$) with RSI boundary constraints ($<45$ for BUY, $>55$ for SELL).
- **Portfolio Backtesting:** Simulates trade executions, tracks capital growth, and compares against benchmark Buy-and-Hold returns.
- **Interactive Streamlit Web Dashboard:** Allows live ticker lookups (e.g., `AAPL`, `MSFT`, `RELIANCE.NS`), on-the-fly indicator calculation, real-time inference, and interactive visualization.

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch (`nn.LSTM`, `nn.Sequential`)
- **Data & Processing:** `yfinance`, `ta` (Technical Analysis library), `numpy`, `pandas`, `scikit-learn`
- **Frontend / Deployment:** Streamlit, Matplotlib

---

## 📂 Project Structure

```
AI-Trading-Assistant/
├── src/
│   ├── app.py           # Interactive Streamlit Web UI
│   ├── data_loader.py   # Yahoo Finance ingestion & indicator engineering
│   ├── preprocess.py    # MinMax scaling and sequence generation
│   ├── model.py         # 2-Layer PyTorch LSTM Architecture
│   ├── train.py         # PyTorch training pipeline and checkpoint saving
│   ├── strategy.py      # Multi-factor algorithmic trading signals
│   └── backtest.py      # Capital simulation and strategy backtesting
├── results/             # Evaluation plots and forecast curves
├── model.pth            # Pre-trained PyTorch model checkpoint
└── requirements.txt
```

---

## ⚡ Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Animesh2102/AI-Trading-Assistant.git
cd AI-Trading-Assistant
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run src/app.py
```

### 3. Train the Model
```bash
python src/train.py
```
