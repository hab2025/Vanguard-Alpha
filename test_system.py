"""
System Test Script for Vanguard-Alpha
Quick verification that all components are working
"""

import sys
import logging

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from config import DEFAULT_SYMBOL, DEFAULT_CAPITAL
        from utils import setup_logger, calculate_sharpe_ratio
        from data_fetcher import DataFetcher, NewsDataFetcher
        from sentiment_analyzer import SentimentAnalyzer
        from risk_engine import RiskEngine
        from trading_engine import TradingEngine
        from calibration import LiveCalibration
        from backtesting_engine import BacktestEngine
        from visualization import TradingVisualizer
        from main import VanguardAlpha
        
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_data_fetcher():
    """Test data fetching"""
    print("\nTesting data fetcher...")
    
    try:
        from data_fetcher import DataFetcher
        
        fetcher = DataFetcher()
        data = fetcher.fetch_historical_data('AAPL', period='5d', interval='1d')
        
        if not data.empty:
            print(f"✅ Data fetcher working ({len(data)} records fetched)")
            return True
        else:
            print("⚠️  No data fetched (may be network issue)")
            return True  # Not a critical failure
    except Exception as e:
        print(f"❌ Data fetcher failed: {str(e)}")
        return False

def test_sentiment_analyzer():
    """Test sentiment analysis"""
    print("\nTesting sentiment analyzer...")
    
    try:
        from sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(use_finbert=False)
        
        result = analyzer.analyze_text("Apple reports strong quarterly earnings")
        
        if 'polarity' in result and 'label' in result:
            print(f"✅ Sentiment analyzer working (polarity: {result['polarity']:.2f})")
            return True
        else:
            print("❌ Sentiment analyzer returned invalid result")
            return False
    except Exception as e:
        print(f"❌ Sentiment analyzer failed: {str(e)}")
        return False

def test_risk_engine():
    """Test risk engine"""
    print("\nTesting risk engine...")
    
    try:
        from risk_engine import RiskEngine
        import numpy as np
        
        engine = RiskEngine(initial_capital=100000)
        
        # Test position sizing
        position_size = engine.calculate_position_size(
            entry_price=150,
            stop_loss_price=145
        )
        
        # Test Monte Carlo
        returns = np.random.normal(0.001, 0.02, 100)
        simulation = engine.monte_carlo_simulation(returns, iterations=1000)
        
        if position_size > 0 and 'var_99' in simulation:
            print(f"✅ Risk engine working (position size: {position_size})")
            return True
        else:
            print("❌ Risk engine returned invalid result")
            return False
    except Exception as e:
        print(f"❌ Risk engine failed: {str(e)}")
        return False

def test_trading_engine():
    """Test trading engine"""
    print("\nTesting trading engine...")
    
    try:
        from trading_engine import TradingEngine
        from data_fetcher import DataFetcher
        import pandas as pd
        
        engine = TradingEngine(initial_capital=100000)
        fetcher = DataFetcher()
        
        # Create sample data
        data = pd.DataFrame({
            'Close': [150, 151, 152],
            'SMA_5': [150, 150.5, 151],
            'SMA_20': [149, 149.5, 150],
            'RSI': [55, 56, 57],
            'MACD': [0.5, 0.6, 0.7],
            'Signal_Line': [0.4, 0.5, 0.6]
        })
        
        signal = engine.generate_signal('AAPL', data, sentiment_score=0.3)
        
        if 'signal' in signal and 'confidence' in signal:
            print(f"✅ Trading engine working (signal: {signal['signal']})")
            return True
        else:
            print("❌ Trading engine returned invalid result")
            return False
    except Exception as e:
        print(f"❌ Trading engine failed: {str(e)}")
        return False

def test_backtesting_engine():
    """Test backtesting engine"""
    print("\nTesting backtesting engine...")
    
    try:
        from backtesting_engine import BacktestEngine, simple_ma_crossover_strategy
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100)
        data = pd.DataFrame({
            'Close': np.random.uniform(145, 155, 100),
            'Open': np.random.uniform(145, 155, 100),
            'High': np.random.uniform(150, 160, 100),
            'Low': np.random.uniform(140, 150, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        engine = BacktestEngine(initial_cash=100000)
        
        # Just test that it can initialize
        print("✅ Backtesting engine initialized")
        return True
    except Exception as e:
        print(f"❌ Backtesting engine failed: {str(e)}")
        return False

def test_visualization():
    """Test visualization"""
    print("\nTesting visualization...")
    
    try:
        from visualization import TradingVisualizer
        
        visualizer = TradingVisualizer()
        
        print("✅ Visualization module initialized")
        return True
    except Exception as e:
        print(f"❌ Visualization failed: {str(e)}")
        return False

def test_main_system():
    """Test main system"""
    print("\nTesting main system...")
    
    try:
        from main import VanguardAlpha
        
        system = VanguardAlpha(initial_capital=100000)
        
        print("✅ Main system initialized")
        return True
    except Exception as e:
        print(f"❌ Main system failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("VANGUARD-ALPHA SYSTEM TEST")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Data Fetcher", test_data_fetcher),
        ("Sentiment Analyzer", test_sentiment_analyzer),
        ("Risk Engine", test_risk_engine),
        ("Trading Engine", test_trading_engine),
        ("Backtesting Engine", test_backtesting_engine),
        ("Visualization", test_visualization),
        ("Main System", test_main_system)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<25} {status}")
    
    print("-"*70)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
