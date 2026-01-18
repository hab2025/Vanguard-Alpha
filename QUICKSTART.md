# Vanguard-Alpha Quick Start Guide

Get started with Vanguard-Alpha in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/hab2025/Vanguard-Alpha.git
cd Vanguard-Alpha

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Your First Analysis

```python
from main import VanguardAlpha

# Initialize system
system = VanguardAlpha(initial_capital=100000)

# Run analysis for Apple stock
analysis = system.run_analysis('AAPL')

# View results
print(f"Signal: {analysis['signal']['signal']}")
print(f"Confidence: {analysis['signal']['confidence']:.2%}")
print(f"Current Price: ${analysis['signal']['price']:.2f}")
```

## Run Examples

```bash
# Run interactive examples
python examples.py

# Or run specific example
python -c "from examples import example_1_basic_analysis; example_1_basic_analysis()"
```

## Quick Backtest

```python
from backtesting_engine import BacktestEngine, simple_ma_crossover_strategy

# Create backtest engine
engine = BacktestEngine(initial_cash=100000)

# Run backtest
results = engine.run_backtest(
    symbol='AAPL',
    strategy_func=simple_ma_crossover_strategy,
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# View results
engine.print_results(results)
```

## Paper Trading Calibration

```bash
# Set API keys (optional for paper trading)
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"

# Run calibration
python calibration.py
```

## Create Visualizations

```python
from visualization import TradingVisualizer
from data_fetcher import DataFetcher

# Fetch data
fetcher = DataFetcher()
data = fetcher.fetch_historical_data('AAPL', period='6mo')

# Create chart
visualizer = TradingVisualizer()
fig = visualizer.plot_price_with_indicators(data, 'AAPL')
fig.show()
```

## Next Steps

1. **Explore Examples**: Run `python examples.py` to see all features
2. **Read Documentation**: Check `README.md` for detailed information
3. **Customize Strategies**: Edit `backtesting_engine.py` to create your own
4. **Train RL Agent**: Use `rl_agent.py` for reinforcement learning
5. **Monitor Performance**: Use visualization tools to track results

## Common Commands

```bash
# Run main application
python main.py

# Run backtesting
python backtesting_engine.py

# Run calibration
python calibration.py

# Run examples
python examples.py

# Create visualizations
python visualization.py
```

## Troubleshooting

### Issue: Module not found
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: No data fetched
```bash
# Check internet connection
# Try different symbol (e.g., MSFT, GOOGL)
# Increase timeout in config.py
```

### Issue: API connection failed
```bash
# Verify API keys are correct
# Check Alpaca account status
# Try simulation mode instead
```

## Support

For more help:
- Check `README.md` for detailed documentation
- Review `examples.py` for usage patterns
- Open an issue on GitHub

Happy Trading! 🚀
