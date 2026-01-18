"""
Usage Examples for Vanguard-Alpha Trading System
Demonstrates various features and capabilities
"""

import logging
from config import DEFAULT_SYMBOL, RESULTS_DIR
from utils import setup_logger
from main import VanguardAlpha
from backtesting_engine import BacktestEngine, simple_ma_crossover_strategy, rsi_strategy
from visualization import TradingVisualizer
from data_fetcher import DataFetcher

logger = setup_logger(__name__)

def example_1_basic_analysis():
    """Example 1: Run basic market analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Market Analysis")
    print("="*70 + "\n")
    
    system = VanguardAlpha(initial_capital=100000)
    analysis = system.run_analysis('AAPL')
    
    if analysis:
        print("\n✅ Analysis completed successfully!")
        print(f"Signal: {analysis['signal']['signal']}")
        print(f"Confidence: {analysis['signal']['confidence']:.2%}")
        print(f"Sentiment: {analysis['sentiment']['overall_sentiment']}")

def example_2_execute_strategy():
    """Example 2: Execute trading strategy"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Execute Trading Strategy")
    print("="*70 + "\n")
    
    system = VanguardAlpha(initial_capital=100000)
    result = system.execute_strategy('MSFT')
    
    if result.get('success'):
        print("\n✅ Strategy executed successfully!")
        if 'trade' in result:
            trade = result['trade']
            print(f"Trade: {trade['signal']} {trade['quantity']} shares @ ${trade['entry_price']:.2f}")
        else:
            print(f"Reason: {result.get('reason')}")

def example_3_backtesting():
    """Example 3: Run backtest with MA crossover strategy"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Backtesting with MA Crossover Strategy")
    print("="*70 + "\n")
    
    engine = BacktestEngine(initial_cash=100000)
    
    results = engine.run_backtest(
        symbol='AAPL',
        strategy_func=simple_ma_crossover_strategy,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    if results.get('success'):
        engine.print_results(results)
        
        # Visualize results
        visualizer = TradingVisualizer()
        fig = visualizer.plot_backtest_results(
            results['portfolio_history'],
            results['trades'],
            'AAPL',
            save_path=f'{RESULTS_DIR}/backtest_ma_crossover.html'
        )
        
        print(f"\n✅ Backtest chart saved to {RESULTS_DIR}/backtest_ma_crossover.html")

def example_4_rsi_strategy():
    """Example 4: Backtest with RSI strategy"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Backtesting with RSI Strategy")
    print("="*70 + "\n")
    
    engine = BacktestEngine(initial_cash=100000)
    
    results = engine.run_backtest(
        symbol='TSLA',
        strategy_func=rsi_strategy,
        start_date='2021-01-01',
        end_date='2023-12-31'
    )
    
    if results.get('success'):
        engine.print_results(results)
        
        # Visualize results
        visualizer = TradingVisualizer()
        fig = visualizer.plot_backtest_results(
            results['portfolio_history'],
            results['trades'],
            'TSLA',
            save_path=f'{RESULTS_DIR}/backtest_rsi_strategy.html'
        )
        
        print(f"\n✅ Backtest chart saved to {RESULTS_DIR}/backtest_rsi_strategy.html")

def example_5_visualization():
    """Example 5: Create technical analysis charts"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Technical Analysis Visualization")
    print("="*70 + "\n")
    
    fetcher = DataFetcher()
    visualizer = TradingVisualizer()
    
    # Fetch data
    data = fetcher.fetch_historical_data('NVDA', period='6mo', interval='1d')
    
    if not data.empty:
        # Create chart
        fig = visualizer.plot_price_with_indicators(
            data,
            'NVDA',
            save_path=f'{RESULTS_DIR}/nvda_technical_analysis.html'
        )
        
        print(f"✅ Technical analysis chart saved to {RESULTS_DIR}/nvda_technical_analysis.html")

def example_6_risk_analysis():
    """Example 6: Risk management and Monte Carlo simulation"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Risk Analysis with Monte Carlo Simulation")
    print("="*70 + "\n")
    
    from risk_engine import RiskEngine
    import numpy as np
    
    risk_engine = RiskEngine(initial_capital=100000)
    
    # Simulate historical returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    
    # Run Monte Carlo simulation
    simulation = risk_engine.monte_carlo_simulation(returns, iterations=10000)
    
    print("Monte Carlo Simulation Results:")
    print(f"Expected Return: {simulation['expected_return']:.4f}")
    print(f"VaR 95%: {simulation['var_95']:.4f}")
    print(f"VaR 99%: {simulation['var_99']:.4f}")
    print(f"CVaR 95%: {simulation['cvar_95']:.4f}")
    print(f"CVaR 99%: {simulation['cvar_99']:.4f}")
    print(f"Best Case: {simulation['best_case']:.4f}")
    print(f"Worst Case: {simulation['worst_case']:.4f}")
    
    # Calculate position size
    entry_price = 150.0
    stop_loss = 142.5
    position_size = risk_engine.calculate_position_size(entry_price, stop_loss)
    
    print(f"\nPosition Sizing:")
    print(f"Entry Price: ${entry_price:.2f}")
    print(f"Stop Loss: ${stop_loss:.2f}")
    print(f"Recommended Position Size: {position_size} shares")
    print(f"Total Investment: ${position_size * entry_price:,.2f}")

def example_7_calibration():
    """Example 7: Live calibration"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Live Calibration (Paper Trading)")
    print("="*70 + "\n")
    
    system = VanguardAlpha(initial_capital=100000)
    
    print("Running calibration cycle...")
    report = system.run_calibration(iterations=3)
    
    if report:
        print("\n✅ Calibration completed!")

def example_8_multi_symbol_analysis():
    """Example 8: Analyze multiple symbols"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Multi-Symbol Analysis")
    print("="*70 + "\n")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    system = VanguardAlpha(initial_capital=100000)
    
    results = {}
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        analysis = system.run_analysis(symbol)
        
        if analysis:
            results[symbol] = {
                'signal': analysis['signal']['signal'],
                'confidence': analysis['signal']['confidence'],
                'sentiment': analysis['sentiment']['overall_sentiment'],
                'price': analysis['signal']['price']
            }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'SYMBOL':<10} {'SIGNAL':<10} {'CONFIDENCE':<15} {'SENTIMENT':<15} {'PRICE':<10}")
    print("-"*70)
    
    for symbol, data in results.items():
        print(f"{symbol:<10} {data['signal']:<10} {data['confidence']:<15.2%} "
              f"{data['sentiment']:<15} ${data['price']:<10.2f}")

def example_9_custom_strategy():
    """Example 9: Create and backtest custom strategy"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Custom Strategy Backtesting")
    print("="*70 + "\n")
    
    def custom_momentum_strategy(data):
        """Custom momentum strategy combining multiple indicators"""
        if len(data) < 50:
            return 'HOLD'
        
        current = data.iloc[-1]
        
        # Get indicators
        price = current['Close']
        sma_20 = current.get('SMA_20', price)
        sma_50 = current.get('SMA_50', price)
        rsi = current.get('RSI', 50)
        macd = current.get('MACD', 0)
        signal_line = current.get('Signal_Line', 0)
        
        # Buy conditions: price above both MAs, RSI not overbought, MACD bullish
        if (price > sma_20 > sma_50 and 
            rsi < 70 and 
            macd > signal_line):
            return 'BUY'
        
        # Sell conditions: price below both MAs or RSI overbought
        elif (price < sma_20 < sma_50 or rsi > 75):
            return 'SELL'
        
        return 'HOLD'
    
    engine = BacktestEngine(initial_cash=100000)
    
    results = engine.run_backtest(
        symbol='AAPL',
        strategy_func=custom_momentum_strategy,
        start_date='2021-01-01',
        end_date='2023-12-31'
    )
    
    if results.get('success'):
        engine.print_results(results)
        
        print("\n✅ Custom strategy backtest completed!")

def example_10_performance_dashboard():
    """Example 10: Create performance dashboard"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Performance Dashboard")
    print("="*70 + "\n")
    
    # Run backtest first
    engine = BacktestEngine(initial_cash=100000)
    results = engine.run_backtest(
        symbol='AAPL',
        strategy_func=simple_ma_crossover_strategy,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    if results.get('success'):
        # Create dashboard
        visualizer = TradingVisualizer()
        fig = visualizer.create_performance_dashboard(
            results,
            save_path=f'{RESULTS_DIR}/performance_dashboard.html'
        )
        
        print(f"✅ Performance dashboard saved to {RESULTS_DIR}/performance_dashboard.html")

def run_all_examples():
    """Run all examples"""
    examples = [
        ("Basic Analysis", example_1_basic_analysis),
        ("Execute Strategy", example_2_execute_strategy),
        ("Backtesting MA Crossover", example_3_backtesting),
        ("Backtesting RSI Strategy", example_4_rsi_strategy),
        ("Technical Visualization", example_5_visualization),
        ("Risk Analysis", example_6_risk_analysis),
        ("Live Calibration", example_7_calibration),
        ("Multi-Symbol Analysis", example_8_multi_symbol_analysis),
        ("Custom Strategy", example_9_custom_strategy),
        ("Performance Dashboard", example_10_performance_dashboard)
    ]
    
    print("\n" + "="*70)
    print("VANGUARD-ALPHA EXAMPLES")
    print("="*70)
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{i}. {name}")
    
    print("\n0. Run All Examples")
    print("="*70 + "\n")
    
    choice = input("Select example to run (0-10): ").strip()
    
    if choice == '0':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        try:
            examples[int(choice)-1][1]()
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    run_all_examples()
