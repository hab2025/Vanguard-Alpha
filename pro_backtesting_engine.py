"""
Vanguard-Alpha Professional Backtesting Engine v2.0
===================================================
محرك اختبار احترافي مع Walk-Forward Analysis و Monte Carlo Simulation

الميزات:
1. Walk-Forward Analysis - اختبار متحرك لمنع Overfitting
2. Monte Carlo Simulation - محاكاة 10,000 سيناريو
3. Realistic Modeling - Slippage, Commission, Spread
4. Advanced Metrics - Sharpe, Sortino, Calmar, Recovery Factor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# =============== CONFIGURATION ===============

@dataclass
class BacktestConfig:
    """إعدادات Backtesting"""
    initial_capital: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    risk_free_rate: float = 0.02  # 2% سنوياً
    
    # Walk-Forward settings
    train_period_days: int = 180  # 6 أشهر
    test_period_days: int = 30  # شهر واحد
    
    # Monte Carlo settings
    mc_simulations: int = 10000
    mc_confidence_levels: List[float] = None
    
    def __post_init__(self):
        if self.mc_confidence_levels is None:
            self.mc_confidence_levels = [0.95, 0.99]

@dataclass
class Trade:
    """بيانات الصفقة"""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0.0
    direction: str = "LONG"  # LONG or SHORT
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

# =============== PROFESSIONAL BACKTESTING ENGINE ===============

class ProBacktestEngine:
    """محرك Backtesting الاحترافي"""
    
    def __init__(self, strategy, config: BacktestConfig = None):
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.logger = logging.getLogger("ProBacktestEngine")
        
        # Results
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.metrics: Dict = {}
        
        self.logger.info("📊 Professional Backtesting Engine initialized")
    
    def run_backtest(self, 
                     price_data: pd.DataFrame,
                     features_data: pd.DataFrame) -> Dict:
        """تشغيل Backtesting الأساسي"""
        
        self.logger.info("🔄 Running backtest...")
        
        capital = self.config.initial_capital
        position = None
        
        self.equity_curve = [capital]
        self.trades = []
        
        for i in range(len(price_data)):
            current_price = price_data.iloc[i]['Close']
            current_time = price_data.index[i]
            
            # Get features for this timestep
            if i < len(features_data):
                current_features = features_data.iloc[i].to_dict()
            else:
                continue
            
            # Get signal from strategy
            signal = self.strategy.predict(current_features)
            
            if signal is None:
                self.equity_curve.append(capital)
                continue
            
            action = signal.get('action', 'HOLD')
            
            # Execute trades
            if action == 'BUY' and position is None:
                # Open LONG position
                quantity = (capital * 0.95) / current_price  # 95% of capital
                entry_price = current_price * (1 + self.config.slippage_rate)
                commission = quantity * entry_price * self.config.commission_rate
                
                position = Trade(
                    entry_time=current_time,
                    entry_price=entry_price,
                    quantity=quantity,
                    direction="LONG",
                    commission=commission,
                    slippage=quantity * current_price * self.config.slippage_rate
                )
                
                capital -= (quantity * entry_price + commission)
                
            elif action == 'SELL' and position is not None and position.direction == "LONG":
                # Close LONG position
                exit_price = current_price * (1 - self.config.slippage_rate)
                commission = position.quantity * exit_price * self.config.commission_rate
                
                position.exit_time = current_time
                position.exit_price = exit_price
                position.pnl = (exit_price - position.entry_price) * position.quantity - position.commission - commission
                
                capital += (position.quantity * exit_price - commission)
                
                self.trades.append(position)
                position = None
            
            # Update equity
            if position is not None:
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
                self.equity_curve.append(capital + unrealized_pnl)
            else:
                self.equity_curve.append(capital)
        
        # Close any open position
        if position is not None:
            exit_price = price_data.iloc[-1]['Close'] * (1 - self.config.slippage_rate)
            commission = position.quantity * exit_price * self.config.commission_rate
            
            position.exit_time = price_data.index[-1]
            position.exit_price = exit_price
            position.pnl = (exit_price - position.entry_price) * position.quantity - position.commission - commission
            
            self.trades.append(position)
        
        # Calculate metrics
        self.metrics = self._calculate_metrics()
        
        self.logger.info(f"✅ Backtest complete: {len(self.trades)} trades")
        
        return self.metrics
    
    def walk_forward_analysis(self,
                               price_data: pd.DataFrame,
                               features_data: pd.DataFrame) -> Dict:
        """Walk-Forward Analysis - اختبار متحرك"""
        
        self.logger.info("🚶 Running Walk-Forward Analysis...")
        
        train_days = self.config.train_period_days
        test_days = self.config.test_period_days
        
        results = []
        
        # Split data into windows
        total_days = len(price_data)
        current_start = 0
        
        while current_start + train_days + test_days <= total_days:
            # Train period
            train_end = current_start + train_days
            train_price = price_data.iloc[current_start:train_end]
            train_features = features_data.iloc[current_start:train_end]
            
            # Test period
            test_end = train_end + test_days
            test_price = price_data.iloc[train_end:test_end]
            test_features = features_data.iloc[train_end:test_end]
            
            # Train strategy (if applicable)
            if hasattr(self.strategy, 'train'):
                self.strategy.train(train_features)
            
            # Test on out-of-sample data
            test_metrics = self.run_backtest(test_price, test_features)
            
            results.append({
                'train_start': train_price.index[0],
                'train_end': train_price.index[-1],
                'test_start': test_price.index[0],
                'test_end': test_price.index[-1],
                'metrics': test_metrics
            })
            
            # Move window
            current_start += test_days
        
        # Aggregate results
        avg_metrics = self._aggregate_walk_forward_results(results)
        
        self.logger.info(f"✅ Walk-Forward Analysis complete: {len(results)} windows")
        
        return {
            'windows': results,
            'average_metrics': avg_metrics
        }
    
    def monte_carlo_simulation(self) -> Dict:
        """Monte Carlo Simulation - محاكاة 10,000 سيناريو"""
        
        if len(self.trades) == 0:
            self.logger.warning("⚠️ No trades to simulate")
            return {}
        
        self.logger.info(f"🎲 Running Monte Carlo Simulation ({self.config.mc_simulations:,} iterations)...")
        
        # Extract trade returns
        trade_returns = [t.pnl / self.config.initial_capital for t in self.trades if t.pnl != 0]
        
        if len(trade_returns) == 0:
            return {}
        
        # Run simulations
        final_equities = []
        
        for _ in range(self.config.mc_simulations):
            # Randomly sample trades with replacement
            simulated_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate cumulative return
            equity = self.config.initial_capital
            for ret in simulated_returns:
                equity *= (1 + ret)
            
            final_equities.append(equity)
        
        final_equities = np.array(final_equities)
        
        # Calculate statistics
        results = {
            'mean_final_equity': np.mean(final_equities),
            'median_final_equity': np.median(final_equities),
            'std_final_equity': np.std(final_equities),
            'min_final_equity': np.min(final_equities),
            'max_final_equity': np.max(final_equities),
        }
        
        # Confidence intervals
        for confidence in self.config.mc_confidence_levels:
            percentile = (1 - confidence) * 100
            var = np.percentile(final_equities, percentile)
            results[f'VaR_{int(confidence*100)}'] = self.config.initial_capital - var
        
        # Probability of profit
        results['prob_profit'] = np.sum(final_equities > self.config.initial_capital) / len(final_equities)
        
        self.logger.info(f"✅ Monte Carlo Simulation complete")
        
        return results
    
    def _calculate_metrics(self) -> Dict:
        """حساب المقاييس المتقدمة"""
        
        if len(self.equity_curve) == 0:
            return {}
        
        equity = pd.Series(self.equity_curve)
        
        # Basic metrics
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        
        # Trade statistics
        completed_trades = [t for t in self.trades if t.pnl != 0]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]
        
        # Win rate
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        
        # Average win/loss
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum([t.pnl for t in winning_trades])
        gross_loss = abs(sum([t.pnl for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Returns
        returns = equity.pct_change().dropna()
        
        # Sharpe Ratio (annualized)
        excess_returns = returns - (self.config.risk_free_rate / 252)
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Maximum Drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        # Recovery Factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Total commission and slippage
        total_commission = sum([t.commission for t in completed_trades])
        total_slippage = sum([t.slippage for t in completed_trades])
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Total Trades': len(completed_trades),
            'Winning Trades': len(winning_trades),
            'Losing Trades': len(losing_trades),
            'Win Rate': f"{win_rate:.2%}",
            'Average Win': f"${avg_win:.2f}",
            'Average Loss': f"${avg_loss:.2f}",
            'Profit Factor': f"{profit_factor:.2f}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Sortino Ratio': f"{sortino:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Calmar Ratio': f"{calmar:.2f}",
            'Recovery Factor': f"{recovery_factor:.2f}",
            'Total Commission': f"${total_commission:.2f}",
            'Total Slippage': f"${total_slippage:.2f}",
            'Final Equity': f"${equity.iloc[-1]:.2f}"
        }
        
        return metrics
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """تجميع نتائج Walk-Forward"""
        
        # Extract metrics from all windows
        all_returns = []
        all_sharpe = []
        all_win_rates = []
        
        for window in results:
            metrics = window['metrics']
            
            # Parse string metrics back to numbers
            total_return_str = metrics.get('Total Return', '0%')
            total_return = float(total_return_str.strip('%').strip('$')) / 100
            all_returns.append(total_return)
            
            sharpe_str = metrics.get('Sharpe Ratio', '0.00')
            sharpe = float(sharpe_str)
            all_sharpe.append(sharpe)
            
            win_rate_str = metrics.get('Win Rate', '0%')
            win_rate = float(win_rate_str.strip('%')) / 100
            all_win_rates.append(win_rate)
        
        return {
            'Average Return': f"{np.mean(all_returns):.2%}",
            'Average Sharpe': f"{np.mean(all_sharpe):.2f}",
            'Average Win Rate': f"{np.mean(all_win_rates):.2%}",
            'Consistency': f"{np.std(all_returns):.2%}"
        }
    
    def plot_results(self):
        """رسم النتائج"""
        
        if not self.equity_curve:
            self.logger.warning("❌ No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Professional Backtest Results', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        axes[0, 0].plot(self.equity_curve, linewidth=2, color='blue')
        axes[0, 0].axhline(y=self.config.initial_capital, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Equity ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        equity = pd.Series(self.equity_curve)
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        axes[0, 1].fill_between(range(len(drawdown)), drawdown * 100, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Returns Distribution
        returns = equity.pct_change().dropna() * 100
        axes[1, 0].hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Trade PnL
        if self.trades:
            trade_pnls = [t.pnl for t in self.trades if t.pnl != 0]
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
            axes[1, 1].bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.7)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            axes[1, 1].set_title('Trade PnL')
            axes[1, 1].set_xlabel('Trade #')
            axes[1, 1].set_ylabel('PnL ($)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/Vanguard-Alpha/backtest_results.png', dpi=150)
        self.logger.info("📊 Results plotted and saved")
        plt.close()
    
    def print_report(self):
        """طباعة التقرير النهائي"""
        
        print("\n" + "="*80)
        print("📊 PROFESSIONAL BACKTEST REPORT")
        print("="*80)
        
        for key, value in self.metrics.items():
            print(f"{key:<25}: {value}")
        
        print("="*80)

# =============== EXAMPLE USAGE ===============

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    price_data = pd.DataFrame({
        'Close': 42000 + np.cumsum(np.random.randn(1000) * 100)
    }, index=dates)
    
    features_data = pd.DataFrame(np.random.randn(1000, 20))
    
    # Dummy strategy
    class DummyStrategy:
        def predict(self, features):
            action = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
            return {'action': action, 'confidence': 0.75}
    
    strategy = DummyStrategy()
    
    # Create backtest engine
    config = BacktestConfig(initial_capital=10000, commission_rate=0.001)
    engine = ProBacktestEngine(strategy, config)
    
    # Run backtest
    metrics = engine.run_backtest(price_data, features_data)
    
    # Print report
    engine.print_report()
    
    # Monte Carlo
    mc_results = engine.monte_carlo_simulation()
    if mc_results:
        print("\n" + "="*80)
        print("🎲 MONTE CARLO SIMULATION RESULTS")
        print("="*80)
        for key, value in mc_results.items():
            if isinstance(value, float):
                print(f"{key:<30}: ${value:,.2f}")
            else:
                print(f"{key:<30}: {value}")
        print("="*80)
    
    # Plot results
    engine.plot_results()
    
    print("\n✅ Backtest complete!")
