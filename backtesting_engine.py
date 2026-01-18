"""
Backtesting Engine for Vanguard-Alpha
Test trading strategies against historical data
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from config import (
    BACKTEST_START_DATE, BACKTEST_END_DATE,
    BACKTEST_INITIAL_CASH, BACKTEST_COMMISSION,
    BACKTEST_SLIPPAGE_PCT
)
from utils import setup_logger, calculate_sharpe_ratio, calculate_max_drawdown
from data_fetcher import DataFetcher
from trading_engine import TradingEngine

logger = setup_logger(__name__)

class BacktestEngine:
    """Backtest trading strategies on historical data"""
    
    def __init__(self, initial_cash: float = BACKTEST_INITIAL_CASH):
        """
        Initialize backtesting engine
        
        Args:
            initial_cash: Initial capital for backtesting
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.portfolio_value = initial_cash
        
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        self.commission = BACKTEST_COMMISSION
        self.slippage = BACKTEST_SLIPPAGE_PCT
        
        self.data_fetcher = DataFetcher()
        
    def load_data(self, symbol: str, start_date: str = BACKTEST_START_DATE,
                  end_date: str = BACKTEST_END_DATE) -> pd.DataFrame:
        """
        Load historical data for backtesting
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Historical data DataFrame
        """
        logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                logger.error(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Calculate indicators
            data = self._calculate_indicators(data)
            
            logger.info(f"Loaded {len(data)} days of data")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        return data
    
    def run_backtest(self, symbol: str, strategy_func, start_date: str = None,
                     end_date: str = None) -> dict:
        """
        Run backtest with a given strategy
        
        Args:
            symbol: Stock ticker symbol
            strategy_func: Strategy function that returns signals
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting Backtest for {symbol}")
        logger.info(f"{'='*70}\n")
        
        # Load data
        data = self.load_data(
            symbol,
            start_date or BACKTEST_START_DATE,
            end_date or BACKTEST_END_DATE
        )
        
        if data.empty:
            return {'success': False, 'reason': 'No data available'}
        
        # Reset state
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        # Iterate through data
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            
            if len(current_data) < 50:  # Need minimum data for indicators
                continue
            
            current_row = current_data.iloc[-1]
            current_price = current_row['Close']
            current_date = current_data.index[-1]
            
            # Get signal from strategy
            signal = strategy_func(current_data)
            
            # Execute trades based on signal
            if signal == 'BUY' and symbol not in self.positions:
                self._execute_buy(symbol, current_price, current_date)
            
            elif signal == 'SELL' and symbol in self.positions:
                self._execute_sell(symbol, current_price, current_date)
            
            # Update portfolio value
            self._update_portfolio_value(symbol, current_price, current_date)
        
        # Close any open positions at end
        if symbol in self.positions:
            final_price = data.iloc[-1]['Close']
            final_date = data.index[-1]
            self._execute_sell(symbol, final_price, final_date)
        
        # Calculate results
        results = self._calculate_results(data)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Backtest Complete")
        logger.info(f"{'='*70}\n")
        
        return results
    
    def _execute_buy(self, symbol: str, price: float, date):
        """Execute buy order"""
        # Apply slippage
        execution_price = price * (1 + self.slippage)
        
        # Calculate position size (use 95% of cash to leave buffer)
        available_cash = self.cash * 0.95
        quantity = int(available_cash / execution_price)
        
        if quantity <= 0:
            return
        
        cost = quantity * execution_price
        commission_cost = cost * self.commission
        total_cost = cost + commission_cost
        
        if total_cost > self.cash:
            return
        
        # Execute trade
        self.cash -= total_cost
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': execution_price,
            'entry_date': date
        }
        
        self.trades.append({
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': execution_price,
            'date': date,
            'commission': commission_cost
        })
        
        logger.debug(f"BUY: {quantity} {symbol} @ ${execution_price:.2f} on {date}")
    
    def _execute_sell(self, symbol: str, price: float, date):
        """Execute sell order"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        quantity = position['quantity']
        
        # Apply slippage
        execution_price = price * (1 - self.slippage)
        
        proceeds = quantity * execution_price
        commission_cost = proceeds * self.commission
        net_proceeds = proceeds - commission_cost
        
        # Calculate P&L
        entry_cost = quantity * position['entry_price']
        pnl = net_proceeds - entry_cost
        pnl_pct = (pnl / entry_cost) * 100
        
        # Execute trade
        self.cash += net_proceeds
        
        self.trades.append({
            'symbol': symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': execution_price,
            'date': date,
            'commission': commission_cost,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_price': position['entry_price'],
            'entry_date': position['entry_date']
        })
        
        logger.debug(f"SELL: {quantity} {symbol} @ ${execution_price:.2f} on {date}, P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        
        del self.positions[symbol]
    
    def _update_portfolio_value(self, symbol: str, current_price: float, date):
        """Update portfolio value"""
        position_value = 0
        
        if symbol in self.positions:
            quantity = self.positions[symbol]['quantity']
            position_value = quantity * current_price
        
        self.portfolio_value = self.cash + position_value
        
        self.portfolio_history.append({
            'date': date,
            'cash': self.cash,
            'position_value': position_value,
            'total_value': self.portfolio_value
        })
    
    def _calculate_results(self, data: pd.DataFrame) -> dict:
        """Calculate backtest results and metrics"""
        if not self.portfolio_history:
            return {'success': False, 'reason': 'No portfolio history'}
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
        
        # Buy and hold benchmark
        initial_price = data.iloc[0]['Close']
        final_price = data.iloc[-1]['Close']
        buy_hold_return = (final_price - initial_price) / initial_price
        
        # Strategy performance
        final_value = self.portfolio_value
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        # Calculate metrics
        returns = portfolio_df['returns'].dropna()
        sharpe_ratio = calculate_sharpe_ratio(returns) if len(returns) > 0 else 0
        max_drawdown = calculate_max_drawdown(returns) if len(returns) > 0 else 0
        
        # Trade statistics
        completed_trades = [t for t in self.trades if 'pnl' in t]
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Calmar ratio
        annual_return = total_return * (252 / len(data))
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        results = {
            'success': True,
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'buy_hold_return': buy_hold_return,
            'buy_hold_return_pct': buy_hold_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'portfolio_history': portfolio_df,
            'trades': completed_trades
        }
        
        return results
    
    def print_results(self, results: dict):
        """Print backtest results"""
        if not results.get('success'):
            logger.error(f"Backtest failed: {results.get('reason')}")
            return
        
        print(f"\n{'='*70}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"Initial Capital:        ${results['initial_cash']:,.2f}")
        print(f"Final Value:            ${results['final_value']:,.2f}")
        print(f"Total Return:           {results['total_return_pct']:.2f}%")
        print(f"Buy & Hold Return:      {results['buy_hold_return_pct']:.2f}%")
        print(f"-" * 70)
        print(f"Sharpe Ratio:           {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:           {results['max_drawdown_pct']:.2f}%")
        print(f"Calmar Ratio:           {results['calmar_ratio']:.2f}")
        print(f"-" * 70)
        print(f"Total Trades:           {results['total_trades']}")
        print(f"Winning Trades:         {results['winning_trades']}")
        print(f"Losing Trades:          {results['losing_trades']}")
        print(f"Win Rate:               {results['win_rate_pct']:.2f}%")
        print(f"Average Win:            ${results['avg_win']:.2f}")
        print(f"Average Loss:           ${results['avg_loss']:.2f}")
        print(f"Profit Factor:          {results['profit_factor']:.2f}")
        print(f"{'='*70}\n")

def simple_ma_crossover_strategy(data: pd.DataFrame) -> str:
    """
    Simple moving average crossover strategy
    
    Args:
        data: Historical data with indicators
        
    Returns:
        Signal: 'BUY', 'SELL', or 'HOLD'
    """
    if len(data) < 50:
        return 'HOLD'
    
    current = data.iloc[-1]
    previous = data.iloc[-2]
    
    sma_20 = current['SMA_20']
    sma_50 = current['SMA_50']
    prev_sma_20 = previous['SMA_20']
    prev_sma_50 = previous['SMA_50']
    
    # Golden cross (bullish)
    if prev_sma_20 <= prev_sma_50 and sma_20 > sma_50:
        return 'BUY'
    
    # Death cross (bearish)
    elif prev_sma_20 >= prev_sma_50 and sma_20 < sma_50:
        return 'SELL'
    
    return 'HOLD'

def rsi_strategy(data: pd.DataFrame) -> str:
    """
    RSI-based strategy
    
    Args:
        data: Historical data with indicators
        
    Returns:
        Signal: 'BUY', 'SELL', or 'HOLD'
    """
    if len(data) < 20:
        return 'HOLD'
    
    current = data.iloc[-1]
    rsi = current['RSI']
    
    # Oversold - buy signal
    if rsi < 30:
        return 'BUY'
    
    # Overbought - sell signal
    elif rsi > 70:
        return 'SELL'
    
    return 'HOLD'

def main():
    """Test backtesting engine"""
    engine = BacktestEngine(initial_cash=100000)
    
    # Test with MA crossover strategy
    results = engine.run_backtest(
        symbol='AAPL',
        strategy_func=simple_ma_crossover_strategy,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    engine.print_results(results)

if __name__ == "__main__":
    main()
