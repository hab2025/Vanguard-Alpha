# Vanguard-Alpha: Advanced AI-Powered Trading System

A sophisticated algorithmic trading platform combining Deep Reinforcement Learning, sentiment analysis, and institutional-grade risk management for automated trading decisions.

## Overview

**Vanguard-Alpha** is an advanced trading system designed to bridge the gap between theoretical AI capabilities and practical trading execution. The system integrates multiple components:

- **Intelligent Market Analysis**: Combines technical indicators with sentiment analysis
- **Risk Management**: Monte Carlo simulations, position sizing, and correlation analysis
- **Live Calibration**: Real-time performance measurement and optimization
- **Paper Trading**: Test strategies without risking real capital

## Features

### 1. Market Data Processing
- Real-time data fetching from multiple sources
- Technical indicator calculation (SMA, EMA, RSI, MACD, Bollinger Bands)
- Historical data analysis and caching

### 2. Sentiment Analysis
- Financial news processing
- TextBlob-based sentiment analysis
- FinBERT support for advanced NLP (optional)
- Headline aggregation and scoring

### 3. Trading Engine
- Technical analysis-based signal generation
- Sentiment-weighted decision making
- Position management
- Trade execution and tracking

### 4. Risk Management
- Dynamic position sizing
- Volatility-adjusted risk
- Monte Carlo simulations
- Value at Risk (VaR) calculations
- Correlation breakdown detection

### 5. Live Calibration
- Paper trading integration with Alpaca
- Latency measurement
- Slippage tracking
- Performance reporting

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Internet connection for market data

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Vanguard-Alpha.git
cd Vanguard-Alpha
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys** (optional for paper trading)
```bash
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
```

## Quick Start

### 1. Run Market Analysis
```bash
python main.py
# Select option 1: Run Analysis
# Enter symbol: AAPL
```

### 2. Execute Trading Strategy
```bash
python main.py
# Select option 2: Execute Strategy
# Enter symbol: AAPL
```

### 3. Run Live Calibration
```bash
python main.py
# Select option 3: Run Calibration
# Enter number of iterations: 5
```

### 4. Direct Module Usage
```python
from trading_engine import TradingEngine
from data_fetcher import DataFetcher
from sentiment_analyzer import SentimentAnalyzer

# Initialize components
engine = TradingEngine(initial_capital=100000)
fetcher = DataFetcher()
analyzer = SentimentAnalyzer()

# Fetch data
data = fetcher.fetch_historical_data('AAPL')

# Analyze sentiment
sentiment = analyzer.analyze_headlines(['AAPL reports strong earnings'])

# Generate signal
signal = engine.generate_signal('AAPL', data, sentiment['avg_polarity'])

print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']:.2%}")
```

## Project Structure

```
Vanguard-Alpha/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration settings
├── utils.py                    # Utility functions
├── data_fetcher.py            # Market data fetching
├── sentiment_analyzer.py       # Sentiment analysis
├── risk_engine.py             # Risk management
├── trading_engine.py          # Trading logic
├── calibration.py             # Live calibration
├── main.py                    # Main application
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── logs/                      # Log files
├── results/                   # Output results
├── models/                    # Trained models
└── data/                      # Market data cache
```

## Configuration

Edit `config.py` to customize system parameters:

```python
# Trading Parameters
DEFAULT_SYMBOL = "AAPL"
DEFAULT_CAPITAL = 100000.0
RISK_PER_TRADE = 0.02  # 2% risk per trade
STOP_LOSS_PCT = 0.05   # 5% stop loss

# Market Data
HISTORICAL_PERIOD = "60d"
MARKET_DATA_INTERVAL = "15m"

# Risk Management
MONTE_CARLO_ITERATIONS = 10000
VAR_CONFIDENCE_LEVEL = 0.99

# API Configuration
ALPACA_API_KEY = "your_key"
ALPACA_SECRET_KEY = "your_secret"
```

## Usage Examples

### Example 1: Basic Analysis
```python
from main import VanguardAlpha

system = VanguardAlpha(initial_capital=100000)
analysis = system.run_analysis('AAPL')

print(f"Signal: {analysis['signal']['signal']}")
print(f"Sentiment: {analysis['sentiment']['overall_sentiment']}")
```

### Example 2: Risk Management
```python
from risk_engine import RiskEngine
import numpy as np

risk_engine = RiskEngine(initial_capital=100000)

# Calculate position size
position_size = risk_engine.calculate_position_size(
    entry_price=150,
    stop_loss_price=142.5
)

# Monte Carlo simulation
returns = np.random.normal(0.001, 0.02, 252)
simulation = risk_engine.monte_carlo_simulation(returns)

print(f"VaR 99%: ${simulation['var_99']:.2f}")
print(f"Position Size: {position_size} shares")
```

### Example 3: Live Calibration
```python
from calibration import LiveCalibration

calibration = LiveCalibration(use_paper_trading=True)
calibration.run_calibration_cycle(symbol='AAPL', iterations=5)
calibration.print_report()
```

## Performance Metrics

The system tracks and reports:

- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Calmar Ratio**: Return relative to maximum drawdown
- **Latency**: System response time
- **Slippage**: Price difference between expected and actual execution

## Calibration Results

After running calibration, the system provides:

```
Average Slippage: $0.0234 (0.0156%)
Average Network Latency: 45.23 ms
Average Execution Latency: 123.45 ms
Average Total Latency: 168.68 ms

Assessment: ✅ GOOD: Latency acceptable for swing trading
```

## API Integration

### Alpaca Markets (Paper Trading)
The system integrates with Alpaca Markets for paper trading:

1. Create account at https://alpaca.markets
2. Get API keys from dashboard
3. Set environment variables or update config.py
4. Run calibration to test connection

### Data Sources
- **yfinance**: Historical and real-time market data
- **Alpaca API**: Paper trading execution
- **News APIs**: Financial news (via yfinance)

## Advanced Features

### 1. Sentiment Analysis with FinBERT
Enable advanced NLP analysis:
```python
analyzer = SentimentAnalyzer(use_finbert=True)
```

### 2. Custom Technical Indicators
Add custom indicators in `data_fetcher.py`:
```python
def _calculate_indicators(self, data):
    # Add your custom indicators here
    data['Custom_Indicator'] = ...
    return data
```

### 3. Strategy Backtesting
Use the risk engine for historical analysis:
```python
from risk_engine import RiskEngine
import pandas as pd

risk_engine = RiskEngine(100000)
portfolio_returns = pd.Series([...])  # Your returns
metrics = risk_engine.calculate_portfolio_metrics(portfolio_returns)
```

## Troubleshooting

### Connection Issues
```
Error: Failed to connect to Alpaca
Solution: Check API keys and internet connection
```

### Data Fetch Errors
```
Error: No data found for symbol
Solution: Verify symbol is valid (e.g., AAPL, MSFT)
```

### High Latency
```
Warning: Latency too high
Solution: Check internet connection, disable VPN, reduce data frequency
```

## Performance Targets

The system aims for:

- **Sharpe Ratio**: > 1.5
- **Win Rate**: > 55%
- **Maximum Drawdown**: < 15%
- **Calmar Ratio**: > 1.0
- **Latency**: < 200ms for HFT, < 500ms for swing trading

## Limitations

- **Paper Trading Only**: Current implementation uses paper trading (no real money)
- **Single Symbol**: Designed for single-symbol analysis (multi-symbol support coming)
- **Sentiment Bias**: Relies on news availability and quality
- **Market Hours**: Optimized for US market hours

## Future Enhancements

- [ ] Multi-asset portfolio support
- [ ] Graph Neural Networks for correlation analysis
- [ ] Deep Reinforcement Learning agent training
- [ ] Backtesting engine with walk-forward analysis
- [ ] Real-time dashboard with Plotly/Dash
- [ ] C++/Rust execution layer for HFT
- [ ] Machine learning model persistence
- [ ] Advanced risk hedging strategies

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Disclaimer

**IMPORTANT**: This system is for educational and research purposes only. It is not financial advice. Trading involves substantial risk of loss. Past performance is not indicative of future results. Always conduct your own research and consult with a financial advisor before trading.

## Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Check existing documentation
- Review logs in `logs/` directory

## Authors

- **Manus AI** - Initial development and architecture

## Acknowledgments

- Alpaca Markets for trading API
- Yahoo Finance for market data
- Hugging Face for NLP models
- OpenAI for language models

## Version History

### v1.0.0 (Current)
- Initial release
- Core trading engine
- Risk management system
- Live calibration module
- Sentiment analysis
- Paper trading support

---

**Last Updated**: January 2026
**Status**: Active Development
