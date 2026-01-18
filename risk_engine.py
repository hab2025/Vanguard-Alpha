"""
Risk Management Engine for Vanguard-Alpha
Handles risk calculations, position sizing, and stress testing
"""

import numpy as np
import pandas as pd
import logging
from config import (
    MONTE_CARLO_ITERATIONS, VAR_CONFIDENCE_LEVEL,
    MAX_POSITION_SIZE, CORRELATION_THRESHOLD,
    RISK_PER_TRADE, STOP_LOSS_PCT
)
from utils import setup_logger

logger = setup_logger(__name__)

class RiskEngine:
    """Manage trading risks and position sizing"""
    
    def __init__(self, initial_capital: float):
        """
        Initialize risk engine
        
        Args:
            initial_capital: Initial portfolio capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        
    def calculate_position_size(self, entry_price: float, stop_loss_price: float,
                               risk_amount: float = None) -> float:
        """
        Calculate optimal position size based on risk management
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            risk_amount: Amount willing to risk (default: RISK_PER_TRADE * capital)
            
        Returns:
            Position size (number of shares)
        """
        if risk_amount is None:
            risk_amount = self.current_capital * RISK_PER_TRADE
        
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            logger.warning("Risk per share is zero, cannot calculate position size")
            return 0
        
        position_size = risk_amount / risk_per_share
        
        # Apply maximum position size constraint
        max_position = (self.current_capital * MAX_POSITION_SIZE) / entry_price
        position_size = min(position_size, max_position)
        
        return int(position_size)
    
    def calculate_volatility_adjusted_size(self, position_size: float, 
                                          volatility: float) -> float:
        """
        Adjust position size based on market volatility
        
        Args:
            position_size: Base position size
            volatility: Current market volatility
            
        Returns:
            Volatility-adjusted position size
        """
        if volatility == 0:
            return position_size
        
        # Reduce position size when volatility is high
        adjustment_factor = 1 / (1 + volatility * 10)
        adjusted_size = position_size * adjustment_factor
        
        return adjusted_size
    
    def monte_carlo_simulation(self, returns: np.ndarray, 
                              iterations: int = MONTE_CARLO_ITERATIONS) -> dict:
        """
        Perform Monte Carlo simulation for future scenarios
        
        Args:
            returns: Historical returns array
            iterations: Number of simulations
            
        Returns:
            Dictionary with simulation results
        """
        if len(returns) == 0:
            return {
                'var_95': 0,
                'var_99': 0,
                'cvar_95': 0,
                'cvar_99': 0,
                'expected_return': 0,
                'worst_case': 0,
                'best_case': 0
            }
        
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Generate random returns
        simulated_returns = np.random.normal(mu, sigma, iterations)
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(simulated_returns, 5)
        var_99 = np.percentile(simulated_returns, 1)
        
        # Calculate Conditional Value at Risk (CVaR)
        cvar_95 = simulated_returns[simulated_returns <= var_95].mean()
        cvar_99 = simulated_returns[simulated_returns <= var_99].mean()
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'expected_return': mu,
            'worst_case': simulated_returns.min(),
            'best_case': simulated_returns.max(),
            'std_dev': sigma
        }
    
    def calculate_var(self, portfolio_value: float, returns: np.ndarray,
                     confidence_level: float = VAR_CONFIDENCE_LEVEL) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            portfolio_value: Current portfolio value
            returns: Historical returns
            confidence_level: Confidence level (e.g., 0.99 for 99%)
            
        Returns:
            VaR in currency units
        """
        if len(returns) == 0:
            return 0
        
        percentile = (1 - confidence_level) * 100
        var_return = np.percentile(returns, percentile)
        var_amount = portfolio_value * var_return
        
        return var_amount
    
    def check_correlation_breakdown(self, asset_returns_dict: dict) -> list:
        """
        Check for dangerous correlations between assets
        
        Args:
            asset_returns_dict: Dictionary of asset returns {symbol: returns_array}
            
        Returns:
            List of alerts for high correlations
        """
        alerts = []
        
        if len(asset_returns_dict) < 2:
            return alerts
        
        symbols = list(asset_returns_dict.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol_a = symbols[i]
                symbol_b = symbols[j]
                
                returns_a = asset_returns_dict[symbol_a]
                returns_b = asset_returns_dict[symbol_b]
                
                # Ensure same length
                min_len = min(len(returns_a), len(returns_b))
                returns_a = returns_a[:min_len]
                returns_b = returns_b[:min_len]
                
                if len(returns_a) == 0 or len(returns_b) == 0:
                    continue
                
                correlation = np.corrcoef(returns_a, returns_b)[0, 1]
                
                if abs(correlation) > CORRELATION_THRESHOLD:
                    alerts.append({
                        'asset_a': symbol_a,
                        'asset_b': symbol_b,
                        'correlation': correlation,
                        'severity': 'critical' if abs(correlation) > 0.95 else 'warning'
                    })
        
        return alerts
    
    def calculate_portfolio_metrics(self, portfolio_returns: np.ndarray) -> dict:
        """
        Calculate comprehensive portfolio metrics
        
        Args:
            portfolio_returns: Array of portfolio returns
            
        Returns:
            Dictionary with portfolio metrics
        """
        if len(portfolio_returns) == 0:
            return {
                'total_return': 0,
                'annual_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0
            }
        
        # Basic metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Sharpe Ratio
        risk_free_rate = 0.02
        excess_return = annual_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino Ratio (only downside volatility)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
    
    def update_capital(self, pnl: float):
        """
        Update current capital based on P&L
        
        Args:
            pnl: Profit/Loss amount
        """
        self.current_capital += pnl
        logger.info(f"Capital updated: ${self.current_capital:.2f} (P&L: ${pnl:.2f})")
    
    def get_risk_summary(self) -> dict:
        """
        Get summary of current risk exposure
        
        Returns:
            Dictionary with risk summary
        """
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'unrealized_pnl': self.current_capital - self.initial_capital,
            'total_positions': len(self.positions),
            'max_position_size': self.initial_capital * MAX_POSITION_SIZE
        }
