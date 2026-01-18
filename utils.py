"""
Utility functions for Vanguard-Alpha Trading System
Helper functions for logging, data processing, and calculations
"""

import logging
import os
from datetime import datetime
from config import LOG_FILE, VERBOSE

def setup_logger(name: str) -> logging.Logger:
    """
    Setup a logger instance with file and console handlers
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if VERBOSE else logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe Ratio for a series of returns
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Sharpe ratio value
    """
    import numpy as np
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    if len(excess_returns) == 0 or excess_returns.std() == 0:
        return 0
    
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return sharpe

def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown from a series of returns
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown percentage
    """
    import numpy as np
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()

def calculate_win_rate(trades):
    """
    Calculate win rate from a list of trades
    
    Args:
        trades: List of trade results (positive for wins, negative for losses)
        
    Returns:
        Win rate as percentage
    """
    if len(trades) == 0:
        return 0
    
    wins = sum(1 for trade in trades if trade > 0)
    return wins / len(trades)

def format_currency(value, decimals=2):
    """
    Format value as currency string
    
    Args:
        value: Numeric value
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    return f"${value:,.{decimals}f}"

def format_percentage(value, decimals=2):
    """
    Format value as percentage string
    
    Args:
        value: Numeric value (0-1 range)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value*100:.{decimals}f}%"

def get_timestamp():
    """
    Get current timestamp in ISO format
    
    Returns:
        ISO format timestamp string
    """
    return datetime.now().isoformat()

def ensure_directory(directory):
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)

class MetricsTracker:
    """Track and manage performance metrics"""
    
    def __init__(self):
        self.trades = []
        self.returns = []
        self.timestamps = []
        
    def add_trade(self, entry_price, exit_price, quantity, timestamp):
        """Record a completed trade"""
        pnl = (exit_price - entry_price) * quantity
        self.trades.append(pnl)
        self.timestamps.append(timestamp)
        
    def add_return(self, daily_return, timestamp):
        """Record daily return"""
        self.returns.append(daily_return)
        self.timestamps.append(timestamp)
        
    def get_summary(self):
        """Get summary statistics"""
        import numpy as np
        
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_trade": 0
            }
        
        return {
            "total_trades": len(self.trades),
            "win_rate": calculate_win_rate(self.trades),
            "total_pnl": sum(self.trades),
            "avg_trade": np.mean(self.trades),
            "max_win": max(self.trades),
            "max_loss": min(self.trades)
        }
