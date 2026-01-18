"""
Visualization Module for Vanguard-Alpha
Create charts and dashboards for trading analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from config import RESULTS_DIR
from utils import setup_logger

logger = setup_logger(__name__)

class TradingVisualizer:
    """Create visualizations for trading analysis"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.figures = []
    
    def plot_price_with_indicators(self, data: pd.DataFrame, symbol: str,
                                   save_path: str = None) -> go.Figure:
        """
        Plot price chart with technical indicators
        
        Args:
            data: DataFrame with OHLCV and indicators
            symbol: Stock ticker symbol
            save_path: Path to save figure (optional)
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price & Moving Averages', 'MACD', 'RSI'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_20'],
                          name='SMA 20', line=dict(color='orange', width=1)),
                row=1, col=1
            )
        
        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_50'],
                          name='SMA 50', line=dict(color='blue', width=1)),
                row=1, col=1
            )
        
        # MACD
        if 'MACD' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'],
                          name='MACD', line=dict(color='blue', width=1)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Signal_Line'],
                          name='Signal', line=dict(color='red', width=1)),
                row=2, col=1
            )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'],
                          name='RSI', line=dict(color='purple', width=1)),
                row=3, col=1
            )
            
            # Overbought/Oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                         annotation_text="Overbought", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                         annotation_text="Oversold", row=3, col=1)
        
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_title='Date',
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Chart saved to {save_path}")
        
        return fig
    
    def plot_backtest_results(self, portfolio_history: pd.DataFrame,
                             trades: list, symbol: str,
                             save_path: str = None) -> go.Figure:
        """
        Plot backtest results
        
        Args:
            portfolio_history: Portfolio value over time
            trades: List of executed trades
            symbol: Stock ticker symbol
            save_path: Path to save figure (optional)
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Portfolio Value', 'Trade P&L'),
            row_heights=[0.7, 0.3]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=portfolio_history.index,
                y=portfolio_history['total_value'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Mark buy/sell points
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        if buy_trades:
            buy_dates = [t['date'] for t in buy_trades]
            buy_values = [portfolio_history.loc[t['date'], 'total_value'] 
                         if t['date'] in portfolio_history.index else None 
                         for t in buy_trades]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_values,
                    mode='markers',
                    name='Buy',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        if sell_trades:
            sell_dates = [t['date'] for t in sell_trades]
            sell_values = [portfolio_history.loc[t['date'], 'total_value']
                          if t['date'] in portfolio_history.index else None
                          for t in sell_trades]
            
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_values,
                    mode='markers',
                    name='Sell',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=1, col=1
            )
        
        # Trade P&L
        if sell_trades:
            pnl_values = [t['pnl'] for t in sell_trades]
            colors = ['green' if p > 0 else 'red' for p in pnl_values]
            
            fig.add_trace(
                go.Bar(
                    x=sell_dates,
                    y=pnl_values,
                    name='Trade P&L',
                    marker_color=colors
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f'{symbol} Backtest Results',
            xaxis_title='Date',
            height=700,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Backtest chart saved to {save_path}")
        
        return fig
    
    def plot_risk_heatmap(self, correlation_matrix: pd.DataFrame,
                         save_path: str = None) -> go.Figure:
        """
        Plot correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            save_path: Path to save figure (optional)
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Asset Correlation Heatmap',
            xaxis_title='Assets',
            yaxis_title='Assets',
            height=600,
            width=700
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Heatmap saved to {save_path}")
        
        return fig
    
    def plot_monte_carlo_simulation(self, simulated_returns: np.ndarray,
                                   var_95: float, var_99: float,
                                   save_path: str = None) -> go.Figure:
        """
        Plot Monte Carlo simulation results
        
        Args:
            simulated_returns: Array of simulated returns
            var_95: Value at Risk (95%)
            var_99: Value at Risk (99%)
            save_path: Path to save figure (optional)
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=simulated_returns,
            nbinsx=50,
            name='Simulated Returns',
            marker_color='lightblue'
        ))
        
        # VaR lines
        fig.add_vline(x=var_95, line_dash="dash", line_color="orange",
                     annotation_text=f"VaR 95%: {var_95:.4f}")
        fig.add_vline(x=var_99, line_dash="dash", line_color="red",
                     annotation_text=f"VaR 99%: {var_99:.4f}")
        
        fig.update_layout(
            title='Monte Carlo Simulation - Return Distribution',
            xaxis_title='Returns',
            yaxis_title='Frequency',
            height=500,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Monte Carlo chart saved to {save_path}")
        
        return fig
    
    def create_performance_dashboard(self, results: dict,
                                    save_path: str = None) -> go.Figure:
        """
        Create comprehensive performance dashboard
        
        Args:
            results: Dictionary with performance metrics
            save_path: Path to save figure (optional)
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Returns', 'Win Rate', 'Risk Metrics', 'Trade Statistics'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Total Return
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=results.get('total_return_pct', 0),
                title={"text": "Total Return (%)"},
                delta={'reference': 0, 'relative': False},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=1
        )
        
        # Win Rate
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=results.get('win_rate_pct', 0),
                title={"text": "Win Rate (%)"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "darkgreen"},
                      'threshold': {
                          'line': {'color': "red", 'width': 4},
                          'thickness': 0.75,
                          'value': 50
                      }},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=2
        )
        
        # Risk Metrics
        risk_metrics = ['Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio']
        risk_values = [
            results.get('sharpe_ratio', 0),
            results.get('max_drawdown_pct', 0),
            results.get('calmar_ratio', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=risk_metrics, y=risk_values, name='Risk Metrics'),
            row=2, col=1
        )
        
        # Trade Statistics
        trade_stats = ['Total', 'Winning', 'Losing']
        trade_values = [
            results.get('total_trades', 0),
            results.get('winning_trades', 0),
            results.get('losing_trades', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=trade_stats, y=trade_values, name='Trades',
                  marker_color=['blue', 'green', 'red']),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Performance Dashboard',
            height=700,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig

def main():
    """Test visualization"""
    from data_fetcher import DataFetcher
    
    visualizer = TradingVisualizer()
    fetcher = DataFetcher()
    
    # Fetch data
    data = fetcher.fetch_historical_data('AAPL', period='6mo', interval='1d')
    
    if not data.empty:
        # Create price chart
        fig = visualizer.plot_price_with_indicators(
            data, 'AAPL',
            save_path=f'{RESULTS_DIR}/aapl_technical_analysis.html'
        )
        
        print(f"Chart created and saved to {RESULTS_DIR}/aapl_technical_analysis.html")

if __name__ == "__main__":
    main()
