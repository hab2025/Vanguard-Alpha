# Vanguard-Alpha Architecture Documentation

## System Overview

Vanguard-Alpha is designed as a modular, scalable trading system with clear separation of concerns. The architecture follows a layered approach with distinct components for data processing, analysis, decision-making, and execution.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Vanguard-Alpha System                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  Data Layer   │  │ Analysis Layer│  │ Decision Layer│   │
│  ├───────────────┤  ├───────────────┤  ├───────────────┤   │
│  │ data_fetcher  │→ │ sentiment_    │→ │ trading_      │   │
│  │               │  │ analyzer      │  │ engine        │   │
│  │ NewsData      │→ │               │→ │               │   │
│  │ Fetcher       │  │               │  │ rl_agent      │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
│         ↓                   ↓                   ↓            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  Risk Layer   │  │ Execution     │  │ Visualization │   │
│  ├───────────────┤  ├───────────────┤  ├───────────────┤   │
│  │ risk_engine   │→ │ calibration   │  │ visualization │   │
│  │               │  │               │  │               │   │
│  │ backtesting_  │  │ main          │  │               │   │
│  │ engine        │  │               │  │               │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Layer

#### data_fetcher.py
**Purpose**: Fetch and process market data from multiple sources

**Key Classes**:
- `DataFetcher`: Main data fetching class
  - Fetches historical OHLCV data
  - Calculates technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
  - Implements caching mechanism
  - Provides data freshness checks

- `NewsDataFetcher`: Financial news retrieval
  - Fetches news headlines
  - Provides sample data for testing

**Data Flow**:
```
yfinance API → DataFetcher → Technical Indicators → Cache → Trading Engine
```

**Key Methods**:
- `fetch_historical_data()`: Get historical market data
- `fetch_latest_bar()`: Get real-time price data
- `_calculate_indicators()`: Compute technical indicators

### 2. Analysis Layer

#### sentiment_analyzer.py
**Purpose**: Analyze sentiment from financial news and text

**Key Classes**:
- `SentimentAnalyzer`: Sentiment analysis engine
  - Supports TextBlob (default)
  - Optional FinBERT integration
  - Headline aggregation
  - Confidence scoring

**Analysis Pipeline**:
```
News Headlines → TextBlob/FinBERT → Polarity Score → Signal Strength → Trading Decision
```

**Key Methods**:
- `analyze_text()`: Analyze single text
- `analyze_headlines()`: Aggregate multiple headlines
- `get_signal_strength()`: Convert score to signal strength

### 3. Decision Layer

#### trading_engine.py
**Purpose**: Core trading logic and signal generation

**Key Classes**:
- `TradingEngine`: Main trading decision engine
  - Combines technical and sentiment analysis
  - Generates trading signals (BUY/SELL/HOLD)
  - Manages positions
  - Tracks trade history

**Decision Process**:
```
Technical Score (60%) + Sentiment Score (40%) → Combined Score → Signal Generation
```

**Signal Thresholds**:
- BUY: Combined score > 0.5
- SELL: Combined score < -0.5
- HOLD: -0.5 ≤ score ≤ 0.5

**Key Methods**:
- `generate_signal()`: Create trading signal
- `execute_trade()`: Execute buy/sell order
- `close_trade()`: Close open position
- `get_performance_summary()`: Calculate metrics

#### rl_agent.py
**Purpose**: Reinforcement Learning for strategy optimization

**Key Classes**:
- `TradingEnvironment`: Gymnasium-compatible RL environment
  - State space: [balance, shares, price, indicators]
  - Action space: [Hold, Buy, Sell]
  - Reward: Portfolio value change

- `RLTradingAgent`: Deep RL agent using PPO
  - Trains on historical data
  - Learns optimal trading policy
  - Model persistence

- `SimpleQLearningAgent`: Fallback Q-Learning implementation

**RL Workflow**:
```
Environment → State → Agent → Action → Reward → Update Policy → Repeat
```

### 4. Risk Layer

#### risk_engine.py
**Purpose**: Risk management and position sizing

**Key Classes**:
- `RiskEngine`: Comprehensive risk management
  - Dynamic position sizing
  - Volatility adjustment
  - Monte Carlo simulation
  - VaR calculation
  - Correlation analysis

**Risk Metrics**:
- Value at Risk (VaR) at 95% and 99% confidence
- Conditional Value at Risk (CVaR)
- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

**Position Sizing Formula**:
```
Position Size = Risk Amount / (Entry Price - Stop Loss Price)
Adjusted Size = Position Size × Volatility Adjustment Factor
```

**Key Methods**:
- `calculate_position_size()`: Determine optimal position
- `monte_carlo_simulation()`: Run scenario analysis
- `calculate_var()`: Compute Value at Risk
- `check_correlation_breakdown()`: Detect dangerous correlations

#### backtesting_engine.py
**Purpose**: Historical strategy testing

**Key Classes**:
- `BacktestEngine`: Backtest trading strategies
  - Simulates historical trading
  - Applies slippage and commissions
  - Calculates performance metrics
  - Supports custom strategies

**Backtest Flow**:
```
Historical Data → Strategy Function → Signal → Execute Trade → Track P&L → Calculate Metrics
```

**Built-in Strategies**:
- `simple_ma_crossover_strategy()`: Moving average crossover
- `rsi_strategy()`: RSI-based trading

**Key Methods**:
- `run_backtest()`: Execute full backtest
- `_execute_buy()`: Simulate buy order
- `_execute_sell()`: Simulate sell order
- `_calculate_results()`: Compute metrics

### 5. Execution Layer

#### calibration.py
**Purpose**: Live system calibration and performance measurement

**Key Classes**:
- `LiveCalibration`: Paper trading calibration
  - Measures latency (network + execution)
  - Tracks slippage
  - Generates performance reports
  - Supports Alpaca API integration

**Calibration Metrics**:
- Network Latency: Time to fetch market data
- Execution Latency: Time to execute order
- Total Latency: End-to-end response time
- Slippage: Price difference (expected vs actual)

**Calibration Flow**:
```
Market Data Request → Measure Latency → Execute Test Trade → Measure Slippage → Generate Report
```

**Key Methods**:
- `execute_test_trade()`: Run single test trade
- `run_calibration_cycle()`: Execute multiple tests
- `generate_calibration_report()`: Create metrics report

#### main.py
**Purpose**: Main application entry point

**Key Classes**:
- `VanguardAlpha`: Main system orchestrator
  - Coordinates all components
  - Provides unified interface
  - Manages system lifecycle

**System Flow**:
```
Initialize → Load Config → Fetch Data → Analyze → Generate Signal → Execute → Monitor
```

### 6. Visualization Layer

#### visualization.py
**Purpose**: Create charts and dashboards

**Key Classes**:
- `TradingVisualizer`: Comprehensive visualization toolkit
  - Price charts with indicators
  - Backtest results
  - Risk heatmaps
  - Monte Carlo distributions
  - Performance dashboards

**Chart Types**:
- Candlestick charts with technical indicators
- Portfolio value over time
- Trade P&L visualization
- Correlation heatmaps
- Risk distribution histograms
- Performance gauges and metrics

**Key Methods**:
- `plot_price_with_indicators()`: Technical analysis chart
- `plot_backtest_results()`: Backtest visualization
- `plot_risk_heatmap()`: Correlation matrix
- `create_performance_dashboard()`: Comprehensive dashboard

### 7. Utility Layer

#### config.py
**Purpose**: Centralized configuration management

**Configuration Categories**:
- API credentials
- Trading parameters
- Market data settings
- Risk management parameters
- Backtesting configuration
- Logging and output paths

#### utils.py
**Purpose**: Common utility functions

**Key Functions**:
- `setup_logger()`: Configure logging
- `calculate_sharpe_ratio()`: Compute Sharpe ratio
- `calculate_max_drawdown()`: Calculate drawdown
- `format_currency()`: Format monetary values
- `MetricsTracker`: Track performance metrics

## Data Flow

### Complete Trading Cycle

```
1. Data Acquisition
   └→ yfinance API
   └→ DataFetcher
   └→ Technical Indicators

2. Sentiment Analysis
   └→ News Headlines
   └→ SentimentAnalyzer
   └→ Polarity Score

3. Signal Generation
   └→ Technical Score (60%)
   └→ Sentiment Score (40%)
   └→ Combined Score
   └→ Trading Signal

4. Risk Assessment
   └→ RiskEngine
   └→ Position Sizing
   └→ VaR Calculation

5. Trade Execution
   └→ TradingEngine
   └→ Order Placement
   └→ Position Management

6. Performance Tracking
   └→ Trade History
   └→ Metrics Calculation
   └→ Visualization
```

## Design Patterns

### 1. Strategy Pattern
Used in backtesting engine to allow custom strategies:
```python
def custom_strategy(data: pd.DataFrame) -> str:
    # Strategy logic
    return 'BUY' | 'SELL' | 'HOLD'

engine.run_backtest(symbol='AAPL', strategy_func=custom_strategy)
```

### 2. Factory Pattern
Used in sentiment analyzer for model selection:
```python
analyzer = SentimentAnalyzer(use_finbert=True)  # Creates FinBERT analyzer
analyzer = SentimentAnalyzer(use_finbert=False)  # Creates TextBlob analyzer
```

### 3. Observer Pattern
Implemented in metrics tracking:
```python
tracker = MetricsTracker()
tracker.add_trade(...)  # Automatically updates metrics
```

### 4. Singleton Pattern
Used for logger configuration:
```python
logger = setup_logger(__name__)  # Returns configured logger instance
```

## Extension Points

### Adding New Data Sources
```python
class CustomDataFetcher(DataFetcher):
    def fetch_from_custom_api(self):
        # Implementation
        pass
```

### Creating Custom Strategies
```python
def my_strategy(data: pd.DataFrame) -> str:
    # Your logic here
    return signal
```

### Adding New Risk Metrics
```python
class ExtendedRiskEngine(RiskEngine):
    def calculate_custom_metric(self):
        # Implementation
        pass
```

### Custom Visualization
```python
class CustomVisualizer(TradingVisualizer):
    def plot_custom_chart(self):
        # Implementation
        pass
```

## Performance Considerations

### Optimization Strategies

1. **Data Caching**: Reduce API calls by caching market data
2. **Indicator Calculation**: Compute once, reuse multiple times
3. **Vectorized Operations**: Use NumPy/Pandas for calculations
4. **Lazy Loading**: Load models only when needed
5. **Parallel Processing**: Future enhancement for multi-symbol analysis

### Scalability

Current system handles:
- Single symbol analysis: < 2 seconds
- Backtest (3 years): < 10 seconds
- Monte Carlo (10K iterations): < 1 second
- RL training (100K steps): 5-10 minutes

## Security Considerations

1. **API Keys**: Stored in environment variables
2. **Paper Trading**: Default mode to prevent real money loss
3. **Input Validation**: All user inputs validated
4. **Error Handling**: Comprehensive exception handling
5. **Logging**: Sensitive data excluded from logs

## Future Enhancements

1. **Multi-Asset Support**: Portfolio-level optimization
2. **Real-time Streaming**: WebSocket data feeds
3. **Advanced RL**: LSTM-based agents
4. **Graph Neural Networks**: Asset correlation modeling
5. **Distributed Backtesting**: Parallel strategy testing
6. **Web Dashboard**: Real-time monitoring interface
7. **Database Integration**: Persistent storage
8. **Alert System**: Email/SMS notifications

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock external dependencies
- Edge case validation

### Integration Tests
- Component interaction testing
- End-to-end workflows
- API integration verification

### Performance Tests
- Latency measurement
- Throughput testing
- Resource utilization

### Stress Tests
- High-frequency scenarios
- Market crash simulations
- Network failure handling

## Deployment

### Development
```bash
python main.py  # Local development
```

### Production (Future)
```bash
docker-compose up  # Containerized deployment
```

## Monitoring

### Key Metrics to Track
- System latency
- Trade execution success rate
- API call frequency
- Memory usage
- Error rates
- Performance metrics (Sharpe, drawdown, etc.)

---

**Last Updated**: January 2026
**Version**: 1.0.0
