"""
Configuration module for Vanguard-Alpha Trading System
Centralized settings and parameters for the entire system
"""

import os
from datetime import datetime

# ============================================================================
# API CONFIGURATION
# ============================================================================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PKXIWGOGYPZZ67H6TQVQ")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "8VsWe9HJbVj92Ky98o68RqLvbzNkM47s5NYJHnSPXErk")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper Trading
ALPACA_API_VERSION = "v2"

# ============================================================================
# TRADING PARAMETERS
# ============================================================================
DEFAULT_SYMBOL = "AAPL"
DEFAULT_CAPITAL = 100000.0
RISK_PER_TRADE = 0.02  # 2% risk per trade
STOP_LOSS_PCT = 0.05  # 5% stop loss
TAKE_PROFIT_PCT = 0.10  # 10% take profit

# ============================================================================
# MARKET DATA PARAMETERS
# ============================================================================
HISTORICAL_PERIOD = "60d"  # Historical data period
MARKET_DATA_INTERVAL = "15m"  # 15-minute intervals
LOOKBACK_WINDOW = 20  # Lookback window for volatility calculation

# ============================================================================
# SENTIMENT ANALYSIS PARAMETERS
# ============================================================================
SENTIMENT_MODEL = "ProsusAI/finbert"  # FinBERT model for financial sentiment
SENTIMENT_THRESHOLD_BUY = 0.5
SENTIMENT_THRESHOLD_SELL = -0.5

# ============================================================================
# REINFORCEMENT LEARNING PARAMETERS
# ============================================================================
RL_TOTAL_TIMESTEPS = 100000
RL_LEARNING_RATE = 0.0003
RL_BATCH_SIZE = 64
RL_GAMMA = 0.99  # Discount factor
RL_ENTROPY_COEF = 0.01

# ============================================================================
# RISK MANAGEMENT PARAMETERS
# ============================================================================
MONTE_CARLO_ITERATIONS = 10000
VAR_CONFIDENCE_LEVEL = 0.99  # 99% confidence level for VaR
MAX_POSITION_SIZE = 0.1  # Maximum 10% of portfolio per position
CORRELATION_THRESHOLD = 0.85  # Alert if correlation > 85%

# ============================================================================
# BACKTESTING PARAMETERS
# ============================================================================
BACKTEST_START_DATE = "2013-01-01"
BACKTEST_END_DATE = "2023-12-31"
BACKTEST_INITIAL_CASH = 100000.0
BACKTEST_COMMISSION = 0.001  # 0.1% commission
BACKTEST_SLIPPAGE_PCT = 0.0005  # 0.05% slippage

# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================
LOG_DIR = "logs"
RESULTS_DIR = "results"
MODELS_DIR = "models"
DATA_DIR = "data"

# Create directories if they don't exist
for directory in [LOG_DIR, RESULTS_DIR, MODELS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"vanguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# ============================================================================
# PERFORMANCE METRICS TARGETS
# ============================================================================
TARGET_SHARPE_RATIO = 1.5
TARGET_WIN_RATE = 0.55  # 55% win rate
TARGET_MAX_DRAWDOWN = 0.15  # 15% maximum drawdown
TARGET_CALMAR_RATIO = 1.0

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================
DEBUG_MODE = False
VERBOSE = True
PAPER_TRADING = True  # Use paper trading mode
LIVE_TRADING = False  # Disable live trading by default
