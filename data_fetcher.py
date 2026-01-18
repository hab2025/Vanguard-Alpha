"""
Data Fetcher Module for Vanguard-Alpha
Handles fetching market data from various sources
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from config import (
    HISTORICAL_PERIOD, MARKET_DATA_INTERVAL, 
    LOOKBACK_WINDOW, DEFAULT_SYMBOL
)
from utils import setup_logger

logger = setup_logger(__name__)

class DataFetcher:
    """Fetch and process market data"""
    
    def __init__(self):
        self.cache = {}
        self.last_update = {}
        
    def fetch_historical_data(self, symbol: str, period: str = HISTORICAL_PERIOD, 
                             interval: str = MARKET_DATA_INTERVAL) -> pd.DataFrame:
        """
        Fetch historical market data using yfinance
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (e.g., '60d', '1y')
            interval: Data interval (e.g., '1m', '15m', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching historical data for {symbol} (period: {period}, interval: {interval})")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Calculate technical indicators
            data = self._calculate_indicators(data)
            
            # Cache the data
            self.cache[symbol] = data
            self.last_update[symbol] = datetime.now()
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_latest_bar(self, symbol: str) -> dict:
        """
        Fetch the latest bar/candle data
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with latest OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if data.empty:
                return {}
            
            latest = data.iloc[-1]
            return {
                'timestamp': data.index[-1],
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'close': latest['Close'],
                'volume': latest['Volume']
            }
            
        except Exception as e:
            logger.error(f"Error fetching latest bar for {symbol}: {str(e)}")
            return {}
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with added indicators
        """
        # Simple Moving Averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
        
        # RSI (Relative Strength Index)
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        
        # Volatility
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=LOOKBACK_WINDOW).std()
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        
        return data
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Series of closing prices
            period: RSI period (default 14)
            
        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_cached_data(self, symbol: str) -> pd.DataFrame:
        """
        Get cached data for a symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Cached DataFrame or empty DataFrame if not found
        """
        return self.cache.get(symbol, pd.DataFrame())
    
    def is_cache_fresh(self, symbol: str, max_age_minutes: int = 5) -> bool:
        """
        Check if cached data is fresh
        
        Args:
            symbol: Stock ticker symbol
            max_age_minutes: Maximum cache age in minutes
            
        Returns:
            True if cache is fresh, False otherwise
        """
        if symbol not in self.last_update:
            return False
        
        age = (datetime.now() - self.last_update[symbol]).total_seconds() / 60
        return age < max_age_minutes

class NewsDataFetcher:
    """Fetch news and sentiment data"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
    def fetch_financial_news(self, symbol: str, limit: int = 10) -> list:
        """
        Fetch financial news headlines for a symbol
        
        Args:
            symbol: Stock ticker symbol
            limit: Number of headlines to fetch
            
        Returns:
            List of news headlines
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news[:limit]
            
            headlines = []
            for item in news:
                headlines.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'timestamp': item.get('providerPublishTime', 0)
                })
            
            self.logger.info(f"Fetched {len(headlines)} news items for {symbol}")
            return headlines
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []
    
    def get_sample_headlines(self, symbol: str) -> list:
        """
        Get sample headlines for testing
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            List of sample headlines
        """
        return [
            f"{symbol} reports strong quarterly earnings",
            f"Market rally boosts {symbol} stock price",
            f"Analyst upgrades {symbol} to buy rating",
            f"Economic uncertainty affects {symbol} trading",
            f"{symbol} announces new product launch"
        ]
