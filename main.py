"""
Main Application for Vanguard-Alpha Trading System
Entry point for the trading system
"""

import logging
import sys
from datetime import datetime
from config import DEFAULT_SYMBOL, DEFAULT_CAPITAL, DEBUG_MODE
from utils import setup_logger, format_currency, format_percentage
from data_fetcher import DataFetcher, NewsDataFetcher
from sentiment_analyzer import SentimentAnalyzer
from trading_engine import TradingEngine
from risk_engine import RiskEngine
from calibration import LiveCalibration

logger = setup_logger(__name__)

class VanguardAlpha:
    """Main Vanguard-Alpha Trading System"""
    
    def __init__(self, initial_capital: float = DEFAULT_CAPITAL):
        """
        Initialize Vanguard-Alpha system
        
        Args:
            initial_capital: Initial trading capital
        """
        logger.info("="*70)
        logger.info("Initializing Vanguard-Alpha Trading System")
        logger.info("="*70)
        
        self.initial_capital = initial_capital
        self.trading_engine = TradingEngine(initial_capital)
        self.data_fetcher = DataFetcher()
        self.news_fetcher = NewsDataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        logger.info(f"Initial Capital: {format_currency(initial_capital)}")
        logger.info("System initialized successfully")
    
    def run_analysis(self, symbol: str = DEFAULT_SYMBOL) -> dict:
        """
        Run complete analysis for a symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Analysis results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Running Analysis for {symbol}")
        logger.info(f"{'='*70}\n")
        
        # Fetch market data
        logger.info(f"[1/4] Fetching market data for {symbol}...")
        market_data = self.data_fetcher.fetch_historical_data(symbol)
        
        if market_data.empty:
            logger.error(f"Failed to fetch market data for {symbol}")
            return {}
        
        logger.info(f"✅ Fetched {len(market_data)} records")
        
        # Fetch news and analyze sentiment
        logger.info(f"[2/4] Fetching news for {symbol}...")
        headlines = self.news_fetcher.get_sample_headlines(symbol)
        
        logger.info(f"[3/4] Analyzing sentiment...")
        sentiment_result = self.sentiment_analyzer.analyze_headlines(headlines)
        
        logger.info(f"✅ Sentiment Analysis Complete")
        logger.info(f"   - Overall Sentiment: {sentiment_result['overall_sentiment'].upper()}")
        logger.info(f"   - Average Polarity: {sentiment_result['avg_polarity']:.4f}")
        logger.info(f"   - Positive: {sentiment_result['positive_count']} | \\\n     Negative: {sentiment_result['negative_count']} | Neutral: {sentiment_result['neutral_count']}")
        
        # Generate trading signal
        logger.info(f"[4/4] Generating trading signal...")
        signal = self.trading_engine.generate_signal(
            symbol,
            market_data,
            sentiment_result['avg_polarity']
        )
        
        logger.info(f"✅ Signal Generated: {signal['signal']}")
        logger.info(f"   - Confidence: {format_percentage(signal['confidence'])}")
        logger.info(f"   - Technical Score: {signal['technical_score']:.4f}")
        logger.info(f"   - Sentiment Score: {signal['sentiment_score']:.4f}")
        logger.info(f"   - Combined Score: {signal['combined_score']:.4f}")
        logger.info(f"   - Current Price: {format_currency(signal['price'])}")
        logger.info(f"   - Reason: {signal['reason']}")
        
        return {
            'symbol': symbol,
            'market_data': market_data,
            'sentiment': sentiment_result,
            'signal': signal,
            'headlines': headlines
        }
    
    def execute_strategy(self, symbol: str = DEFAULT_SYMBOL) -> dict:
        """
        Execute complete trading strategy
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Strategy execution results
        """
        # Run analysis
        analysis = self.run_analysis(symbol)
        
        if not analysis:
            return {'success': False, 'reason': 'Analysis failed'}
        
        signal = analysis['signal']
        
        # Execute trade if signal is strong enough
        if signal['signal'] != 'HOLD' and signal['confidence'] > 0.6:
            logger.info(f"\n{'='*70}")
            logger.info(f"Executing Trade")
            logger.info(f"{'='*70}\n")
            
            price = signal['price']
            
            # Calculate position size
            stop_loss = price * (1 - 0.05) if signal['signal'] == 'BUY' else price * (1 + 0.05)
            quantity = self.trading_engine.risk_engine.calculate_position_size(
                price, stop_loss
            )
            
            logger.info(f"Position Size: {quantity} shares")
            logger.info(f"Entry Price: {format_currency(price)}")
            
            # Execute trade
            result = self.trading_engine.execute_trade(
                symbol,
                signal['signal'],
                price,
                quantity
            )
            
            if result['success']:
                logger.info(f"✅ Trade executed successfully")
                return {'success': True, 'trade': result['trade']}
            else:
                logger.error(f"❌ Trade execution failed: {result['reason']}")
                return result
        else:
            logger.info(f"\n⏸️  Signal confidence too low ({format_percentage(signal['confidence'])}) or HOLD signal")
            logger.info(f"No trade executed")
            return {'success': True, 'reason': 'No trade executed (low confidence or HOLD)'}
    
    def run_calibration(self, iterations: int = 5) -> dict:
        """
        Run live calibration
        
        Args:
            iterations: Number of test trades
            
        Returns:
            Calibration report
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Running Live Calibration")
        logger.info(f"{'='*70}\n")
        
        calibration = LiveCalibration(use_paper_trading=True)
        
        if not calibration.connected:
            logger.warning("Falling back to simulation mode")
            calibration = LiveCalibration(use_paper_trading=False)
        
        calibration.run_calibration_cycle(iterations=iterations)
        calibration.print_report()
        
        return calibration.generate_calibration_report()
    
    def get_performance_summary(self) -> dict:
        """
        Get system performance summary
        
        Returns:
            Performance metrics
        """
        return self.trading_engine.get_performance_summary()
    
    def print_system_status(self):
        """Print current system status"""
        logger.info(f"\n{'='*70}")
        logger.info(f"System Status")
        logger.info(f"{'='*70}")
        
        risk_summary = self.trading_engine.risk_engine.get_risk_summary()
        perf_summary = self.trading_engine.get_performance_summary()
        
        logger.info(f"Initial Capital: {format_currency(risk_summary['initial_capital'])}")
        logger.info(f"Current Capital: {format_currency(risk_summary['current_capital'])}")
        logger.info(f"Unrealized P&L: {format_currency(risk_summary['unrealized_pnl'])}")
        logger.info(f"Open Positions: {risk_summary['total_positions']}")
        
        logger.info(f"\nPerformance:")
        logger.info(f"Total Trades: {perf_summary['total_trades']}")
        logger.info(f"Winning Trades: {perf_summary['winning_trades']}")
        logger.info(f"Losing Trades: {perf_summary['losing_trades']}")
        logger.info(f"Win Rate: {format_percentage(perf_summary['win_rate'])}")
        logger.info(f"Total P&L: {format_currency(perf_summary['total_pnl'])}")
        
        logger.info(f"{'='*70}\n")

def main():
    """Main entry point"""
    try:
        # Initialize system
        system = VanguardAlpha(initial_capital=100000)
        
        # Run analysis and strategy
        print("\n" + "="*70)
        print("VANGUARD-ALPHA TRADING SYSTEM")
        print("="*70)
        print("\nOptions:")
        print("1. Run Analysis")
        print("2. Execute Strategy")
        print("3. Run Calibration")
        print("4. Show System Status")
        print("5. Exit")
        print("="*70 + "\n")
        
        while True:
            choice = input("Select option (1-5): ").strip()
            
            if choice == '1':
                symbol = input("Enter symbol (default: AAPL): ").strip() or DEFAULT_SYMBOL
                system.run_analysis(symbol)
            
            elif choice == '2':
                symbol = input("Enter symbol (default: AAPL): ").strip() or DEFAULT_SYMBOL
                system.execute_strategy(symbol)
            
            elif choice == '3':
                iterations = input("Enter number of iterations (default: 5): ").strip()
                iterations = int(iterations) if iterations.isdigit() else 5
                system.run_calibration(iterations)
            
            elif choice == '4':
                system.print_system_status()
            
            elif choice == '5':
                logger.info("Exiting Vanguard-Alpha")
                break
            
            else:
                print("Invalid option. Please try again.")
    
    except KeyboardInterrupt:
        logger.info("\nSystem interrupted by user")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
