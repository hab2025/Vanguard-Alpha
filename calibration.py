"""
Live Calibration Module for Vanguard-Alpha
Tests system performance with live market data (paper trading)
"""

import time
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import uuid
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    DEFAULT_SYMBOL, RESULTS_DIR
)
from utils import setup_logger, format_currency, format_percentage

logger = setup_logger(__name__)

class LiveCalibration:
    """Calibrate system performance with live market data"""
    
    def __init__(self, use_paper_trading: bool = True):
        """
        Initialize calibration module
        
        Args:
            use_paper_trading: Use paper trading (True) or simulation (False)
        """
        self.use_paper_trading = use_paper_trading
        self.trade_logs = []
        self.metrics = []
        
        if use_paper_trading:
            try:
                import alpaca_trade_api as tradeapi
                self.api = tradeapi.REST(
                    ALPACA_API_KEY,
                    ALPACA_SECRET_KEY,
                    ALPACA_BASE_URL,
                    api_version='v2'
                )
                account = self.api.get_account()
                logger.info(f"✅ Connected to Alpaca! Cash: ${account.cash}")
                self.connected = True
            except Exception as e:
                logger.error(f"❌ Failed to connect to Alpaca: {str(e)}")
                self.connected = False
        else:
            self.connected = True
            logger.info("Running in simulation mode")
    
    def get_market_data(self, symbol: str = DEFAULT_SYMBOL) -> dict:
        """
        Fetch latest market data
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with market data
        """
        start_time = time.time()
        
        if self.use_paper_trading and self.connected:
            try:
                bar = self.api.get_latest_bar(symbol)
                latency = (time.time() - start_time) * 1000
                
                return {
                    'price': bar.c,
                    'bid': bar.c - 0.01,
                    'ask': bar.c + 0.01,
                    'timestamp': time.time(),
                    'net_latency_ms': latency,
                    'source': 'live'
                }
            except Exception as e:
                logger.error(f"Error fetching market data: {str(e)}")
                return self._get_simulated_data()
        else:
            return self._get_simulated_data()
    
    def _get_simulated_data(self) -> dict:
        """Get simulated market data for testing"""
        import random
        
        start_time = time.time()
        time.sleep(random.uniform(0.05, 0.3))  # Simulate network latency
        
        current_price = 150 + random.uniform(-5, 5)
        
        return {
            'price': current_price,
            'bid': current_price - 0.05,
            'ask': current_price + 0.05,
            'timestamp': time.time(),
            'net_latency_ms': (time.time() - start_time) * 1000,
            'source': 'simulated'
        }
    
    def execute_test_trade(self, symbol: str = DEFAULT_SYMBOL,
                          side: str = 'buy') -> dict:
        """
        Execute a test trade and measure metrics
        
        Args:
            symbol: Stock ticker symbol
            side: Trade side ('buy' or 'sell')
            
        Returns:
            Dictionary with trade metrics
        """
        # Get market data
        data = self.get_market_data(symbol)
        expected_price = data['price']
        net_latency = data['net_latency_ms']
        
        start_exec = time.time()
        
        if self.use_paper_trading and self.connected:
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=1,
                    side=side,
                    type='market',
                    time_in_force='gtc'
                )
                
                # Wait for order to fill
                while True:
                    order_status = self.api.get_order(order.id)
                    if order_status.status == 'filled':
                        filled_price = float(order_status.filled_avg_price)
                        break
                    time.sleep(0.5)
                    
                exec_latency = (time.time() - start_exec) * 1000
                
            except Exception as e:
                logger.error(f"Trade execution error: {str(e)}")
                filled_price = expected_price
                exec_latency = (time.time() - start_exec) * 1000
        else:
            # Simulate trade execution
            import random
            exec_delay = random.uniform(0.1, 0.5)
            time.sleep(exec_delay)
            
            market_move = random.uniform(-10, 10) / 1000  # Small price move
            filled_price = expected_price + market_move if side == 'buy' else expected_price - market_move
            exec_latency = (time.time() - start_exec) * 1000
        
        # Calculate metrics
        slippage = abs(filled_price - expected_price)
        total_latency = net_latency + exec_latency
        
        return {
            'id': str(uuid.uuid4())[:8],
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'expected_price': round(expected_price, 4),
            'filled_price': round(filled_price, 4),
            'slippage_abs': round(slippage, 4),
            'slippage_pct': round((slippage / expected_price) * 100, 4),
            'network_latency_ms': round(net_latency, 2),
            'execution_latency_ms': round(exec_latency, 2),
            'total_latency_ms': round(total_latency, 2)
        }
    
    def run_calibration_cycle(self, symbol: str = DEFAULT_SYMBOL,
                             iterations: int = 5) -> list:
        """
        Run calibration cycle with multiple test trades
        
        Args:
            symbol: Stock ticker symbol
            iterations: Number of test trades
            
        Returns:
            List of trade metrics
        """
        logger.info(f"\\n{'='*70}")
        logger.info(f"Starting Live Calibration Cycle")
        logger.info(f"Symbol: {symbol}, Iterations: {iterations}")
        logger.info(f"{'='*70}\\n")
        
        print(f"{'SIDE':<6} | {'EXP PRICE':<12} | {'FILL PRICE':<12} | {'SLIPPAGE':<10} | {'LATENCY':<10}")
        print("-" * 70)
        
        for i in range(iterations):
            side = 'buy' if i % 2 == 0 else 'sell'
            result = self.execute_test_trade(symbol, side)
            
            self.trade_logs.append(result)
            
            print(f"{side.upper():<6} | {result['expected_price']:<12} | {result['filled_price']:<12} | \\\n                {result['slippage_abs']:<10} | {result['total_latency_ms']:<10}")
            
            time.sleep(1)  # Wait between trades
        
        return self.trade_logs
    
    def generate_calibration_report(self) -> dict:
        """
        Generate comprehensive calibration report
        
        Returns:
            Dictionary with report data
        """
        if not self.trade_logs:
            logger.warning("No trade logs available")
            return {}
        
        df = pd.DataFrame(self.trade_logs)
        
        avg_slippage = df['slippage_abs'].mean()
        avg_slippage_pct = df['slippage_pct'].mean()
        avg_net_latency = df['network_latency_ms'].mean()
        avg_exec_latency = df['execution_latency_ms'].mean()
        avg_total_latency = df['total_latency_ms'].mean()
        
        max_latency = df['total_latency_ms'].max()
        min_latency = df['total_latency_ms'].min()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_trades': len(self.trade_logs),
            'avg_slippage_abs': round(avg_slippage, 4),
            'avg_slippage_pct': round(avg_slippage_pct, 4),
            'avg_network_latency_ms': round(avg_net_latency, 2),
            'avg_execution_latency_ms': round(avg_exec_latency, 2),
            'avg_total_latency_ms': round(avg_total_latency, 2),
            'max_latency_ms': round(max_latency, 2),
            'min_latency_ms': round(min_latency, 2),
            'source': 'paper_trading' if self.use_paper_trading else 'simulation'
        }
        
        # Assessment
        if avg_total_latency < 200:
            assessment = "✅ EXCELLENT: Latency within HFT acceptable range"
        elif avg_total_latency < 500:
            assessment = "✅ GOOD: Latency acceptable for swing trading"
        elif avg_total_latency < 1000:
            assessment = "⚠️ WARNING: Latency acceptable for day trading"
        else:
            assessment = "❌ CRITICAL: Latency too high, consider optimization"
        
        if avg_slippage_pct < 0.05:
            slippage_assessment = "✅ EXCELLENT: Minimal slippage"
        elif avg_slippage_pct < 0.1:
            slippage_assessment = "✅ GOOD: Acceptable slippage"
        elif avg_slippage_pct < 0.2:
            slippage_assessment = "⚠️ WARNING: Moderate slippage"
        else:
            slippage_assessment = "❌ CRITICAL: High slippage, consider limit orders"
        
        report['assessment'] = assessment
        report['slippage_assessment'] = slippage_assessment
        
        return report
    
    def export_report(self, filename: str = None) -> str:
        """
        Export calibration report to CSV
        
        Args:
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if not self.trade_logs:
            logger.warning("No data to export")
            return ""
        
        if filename is None:
            filename = f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = f"{RESULTS_DIR}/{filename}"
        
        df = pd.DataFrame(self.trade_logs)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Report exported to {filepath}")
        
        return filepath
    
    def print_report(self):
        """Print calibration report to console"""
        report = self.generate_calibration_report()
        
        if not report:
            return
        
        print("\\n" + "="*70)
        print("📊 VANGUARD-ALPHA CALIBRATION REPORT")
        print("="*70)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total Trades: {report['total_trades']}")
        print(f"Source: {report['source']}")
        print("-" * 70)
        print(f"Average Slippage: {format_currency(report['avg_slippage_abs'])} ({format_percentage(report['avg_slippage_pct']/100)})")
        print(f"Average Network Latency: {report['avg_network_latency_ms']:.2f} ms")
        print(f"Average Execution Latency: {report['avg_execution_latency_ms']:.2f} ms")
        print(f"Average Total Latency: {report['avg_total_latency_ms']:.2f} ms")
        print(f"Max Latency: {report['max_latency_ms']:.2f} ms")
        print(f"Min Latency: {report['min_latency_ms']:.2f} ms")
        print("-" * 70)
        print(f"Assessment: {report['assessment']}")
        print(f"Slippage Assessment: {report['slippage_assessment']}")
        print("="*70 + "\\n")

def main():
    """Main calibration routine"""
    calibration = LiveCalibration(use_paper_trading=True)
    
    if not calibration.connected:
        logger.warning("Falling back to simulation mode")
        calibration = LiveCalibration(use_paper_trading=False)
    
    calibration.run_calibration_cycle(symbol='AAPL', iterations=5)
    calibration.print_report()
    calibration.export_report()

if __name__ == "__main__":
    main()
