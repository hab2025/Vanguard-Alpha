"""
Trading Engine for Vanguard-Alpha
Core trading logic and decision making
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from config import (
    SENTIMENT_THRESHOLD_BUY, SENTIMENT_THRESHOLD_SELL,
    RISK_PER_TRADE, STOP_LOSS_PCT, TAKE_PROFIT_PCT
)
from utils import setup_logger
from data_fetcher import DataFetcher
from sentiment_analyzer import SentimentAnalyzer
from risk_engine import RiskEngine

logger = setup_logger(__name__)

class TradingEngine:
    """Core trading engine with decision logic"""
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize trading engine
        
        Args:
            initial_capital: Initial trading capital
        """
        self.initial_capital = initial_capital
        self.data_fetcher = DataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer(use_finbert=False)
        self.risk_engine = RiskEngine(initial_capital)
        
        self.positions = {}
        self.trade_history = []
        self.signals = []
        
    def generate_signal(self, symbol: str, market_data: pd.DataFrame,
                       sentiment_score: float) -> dict:
        """
        Generate trading signal based on technical and sentiment analysis
        
        Args:
            symbol: Stock ticker symbol
            market_data: DataFrame with OHLCV and indicators
            sentiment_score: Sentiment score (-1 to 1)
            
        Returns:
            Dictionary with signal information
        """
        if market_data.empty:
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': 0,
                'reason': 'No market data available'
            }
        
        latest = market_data.iloc[-1]
        
        # Technical Analysis
        price = latest['Close']
        sma_5 = latest.get('SMA_5', price)
        sma_20 = latest.get('SMA_20', price)
        rsi = latest.get('RSI', 50)
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('Signal_Line', 0)
        
        # Calculate technical score
        technical_score = 0
        
        # Price above moving averages (bullish)
        if price > sma_5 > sma_20:
            technical_score += 0.3
        elif price < sma_5 < sma_20:
            technical_score -= 0.3
        
        # RSI signals
        if rsi < 30:
            technical_score += 0.2  # Oversold, potential bounce
        elif rsi > 70:
            technical_score -= 0.2  # Overbought, potential pullback
        
        # MACD signals
        if macd > macd_signal:
            technical_score += 0.2  # Bullish crossover
        elif macd < macd_signal:
            technical_score -= 0.2  # Bearish crossover
        
        # Combine technical and sentiment scores
        combined_score = (technical_score * 0.6) + (sentiment_score * 0.4)
        
        # Generate signal
        if combined_score > SENTIMENT_THRESHOLD_BUY:
            signal = 'BUY'
            confidence = min(combined_score, 1.0)
        elif combined_score < SENTIMENT_THRESHOLD_SELL:
            signal = 'SELL'
            confidence = min(abs(combined_score), 1.0)
        else:
            signal = 'HOLD'
            confidence = 0
        
        signal_info = {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'technical_score': technical_score,
            'sentiment_score': sentiment_score,
            'combined_score': combined_score,
            'price': price,
            'rsi': rsi,
            'timestamp': datetime.now().isoformat(),
            'reason': self._get_signal_reason(signal, technical_score, sentiment_score)
        }
        
        self.signals.append(signal_info)
        return signal_info
    
    def _get_signal_reason(self, signal: str, technical_score: float,
                          sentiment_score: float) -> str:
        """
        Generate human-readable reason for signal
        
        Args:
            signal: Trading signal (BUY/SELL/HOLD)
            technical_score: Technical analysis score
            sentiment_score: Sentiment analysis score
            
        Returns:
            Reason string
        """
        if signal == 'BUY':
            reasons = []
            if technical_score > 0:
                reasons.append("positive technical setup")
            if sentiment_score > 0:
                reasons.append("positive sentiment")
            return f"Strong buy signal: {', '.join(reasons)}"
        
        elif signal == 'SELL':
            reasons = []
            if technical_score < 0:
                reasons.append("negative technical setup")
            if sentiment_score < 0:
                reasons.append("negative sentiment")
            return f"Strong sell signal: {', '.join(reasons)}"
        
        else:
            return "Market conditions neutral, holding position"
    
    def execute_trade(self, symbol: str, signal: str, price: float,
                     quantity: int) -> dict:
        """
        Execute a trade
        
        Args:
            symbol: Stock ticker symbol
            signal: Trade signal (BUY/SELL)
            price: Execution price
            quantity: Number of shares
            
        Returns:
            Trade execution result
        """
        if signal not in ['BUY', 'SELL']:
            return {'success': False, 'reason': 'Invalid signal'}
        
        try:
            # Calculate stop loss and take profit
            if signal == 'BUY':
                stop_loss = price * (1 - STOP_LOSS_PCT)
                take_profit = price * (1 + TAKE_PROFIT_PCT)
            else:
                stop_loss = price * (1 + STOP_LOSS_PCT)
                take_profit = price * (1 - TAKE_PROFIT_PCT)
            
            trade = {
                'symbol': symbol,
                'signal': signal,
                'entry_price': price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now().isoformat(),
                'status': 'open'
            }
            
            self.positions[symbol] = trade
            self.trade_history.append(trade)
            
            logger.info(f"Trade executed: {signal} {quantity} {symbol} @ ${price:.2f}")
            
            return {
                'success': True,
                'trade': trade
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return {'success': False, 'reason': str(e)}
    
    def close_trade(self, symbol: str, exit_price: float) -> dict:
        """
        Close an open trade
        
        Args:
            symbol: Stock ticker symbol
            exit_price: Exit price
            
        Returns:
            Trade closure result
        """
        if symbol not in self.positions:
            return {'success': False, 'reason': f'No open position for {symbol}'}
        
        try:
            trade = self.positions[symbol]
            
            # Calculate P&L
            if trade['signal'] == 'BUY':
                pnl = (exit_price - trade['entry_price']) * trade['quantity']
            else:
                pnl = (trade['entry_price'] - exit_price) * trade['quantity']
            
            trade['exit_price'] = exit_price
            trade['exit_time'] = datetime.now().isoformat()
            trade['pnl'] = pnl
            trade['status'] = 'closed'
            
            # Update capital
            self.risk_engine.update_capital(pnl)
            
            logger.info(f"Trade closed: {symbol} @ ${exit_price:.2f}, P&L: ${pnl:.2f}")
            
            del self.positions[symbol]
            
            return {
                'success': True,
                'pnl': pnl,
                'trade': trade
            }
            
        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
            return {'success': False, 'reason': str(e)}
    
    def get_open_positions(self) -> list:
        """
        Get list of open positions
        
        Returns:
            List of open positions
        """
        return list(self.positions.values())
    
    def get_trade_history(self) -> list:
        """
        Get trade history
        
        Returns:
            List of all trades
        """
        return self.trade_history
    
    def get_performance_summary(self) -> dict:
        """
        Get performance summary
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_trade_pnl': 0
            }
        
        closed_trades = [t for t in self.trade_history if t['status'] == 'closed']
        
        if not closed_trades:
            return {
                'total_trades': len(self.trade_history),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_trade_pnl': 0
            }
        
        pnls = [t['pnl'] for t in closed_trades]
        winning_trades = sum(1 for p in pnls if p > 0)
        losing_trades = sum(1 for p in pnls if p < 0)
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / len(closed_trades),
            'total_pnl': sum(pnls),
            'avg_trade_pnl': np.mean(pnls),
            'max_win': max(pnls),
            'max_loss': min(pnls)
        }
