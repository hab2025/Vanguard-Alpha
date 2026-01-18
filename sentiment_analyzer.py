"""
Sentiment Analysis Module for Vanguard-Alpha
Analyzes financial news and sentiment using NLP
"""

import numpy as np
import logging
from textblob import TextBlob
from config import (
    SENTIMENT_MODEL, SENTIMENT_THRESHOLD_BUY, 
    SENTIMENT_THRESHOLD_SELL
)
from utils import setup_logger

logger = setup_logger(__name__)

class SentimentAnalyzer:
    """Analyze sentiment from financial news and text"""
    
    def __init__(self, use_finbert: bool = False):
        """
        Initialize sentiment analyzer
        
        Args:
            use_finbert: Whether to use FinBERT model (requires transformers library)
        """
        self.use_finbert = use_finbert
        self.sentiment_cache = {}
        
        if use_finbert:
            try:
                from transformers import pipeline
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=SENTIMENT_MODEL
                )
                logger.info(f"Loaded FinBERT model: {SENTIMENT_MODEL}")
            except ImportError:
                logger.warning("Transformers library not available, using TextBlob")
                self.use_finbert = False
            except Exception as e:
                logger.error(f"Error loading FinBERT: {str(e)}")
                self.use_finbert = False
    
    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment of a text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'label': 'neutral',
                'confidence': 0
            }
        
        # Check cache
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]
        
        if self.use_finbert:
            result = self._analyze_with_finbert(text)
        else:
            result = self._analyze_with_textblob(text)
        
        # Cache result
        self.sentiment_cache[text] = result
        
        return result
    
    def _analyze_with_textblob(self, text: str) -> dict:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Determine label
            if polarity > SENTIMENT_THRESHOLD_BUY:
                label = 'positive'
            elif polarity < SENTIMENT_THRESHOLD_SELL:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Confidence based on subjectivity
            confidence = abs(polarity)
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'label': label,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text with TextBlob: {str(e)}")
            return {
                'polarity': 0,
                'subjectivity': 0,
                'label': 'neutral',
                'confidence': 0
            }
    
    def _analyze_with_finbert(self, text: str) -> dict:
        """
        Analyze sentiment using FinBERT
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            result = self.sentiment_pipeline(text[:512])[0]  # Limit to 512 tokens
            
            label = result['label'].lower()
            score = result['score']
            
            # Convert to polarity scale (-1 to 1)
            if label == 'positive':
                polarity = score
            elif label == 'negative':
                polarity = -score
            else:
                polarity = 0
            
            return {
                'polarity': polarity,
                'subjectivity': 0.5,  # FinBERT doesn't provide subjectivity
                'label': label,
                'confidence': score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text with FinBERT: {str(e)}")
            return {
                'polarity': 0,
                'subjectivity': 0,
                'label': 'neutral',
                'confidence': 0
            }
    
    def analyze_headlines(self, headlines: list) -> dict:
        """
        Analyze sentiment from multiple headlines
        
        Args:
            headlines: List of headline strings or dictionaries
            
        Returns:
            Aggregated sentiment scores
        """
        if not headlines:
            return {
                'avg_polarity': 0,
                'avg_confidence': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'overall_sentiment': 'neutral'
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for headline in headlines:
            # Handle both string and dictionary formats
            if isinstance(headline, dict):
                text = headline.get('title', '') or headline.get('summary', '')
            else:
                text = headline
            
            sentiment = self.analyze_text(text)
            sentiments.append(sentiment)
            
            if sentiment['label'] == 'positive':
                positive_count += 1
            elif sentiment['label'] == 'negative':
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate aggregates
        polarities = [s['polarity'] for s in sentiments]
        confidences = [s['confidence'] for s in sentiments]
        
        avg_polarity = np.mean(polarities)
        avg_confidence = np.mean(confidences)
        
        # Determine overall sentiment
        if avg_polarity > SENTIMENT_THRESHOLD_BUY:
            overall_sentiment = 'positive'
        elif avg_polarity < SENTIMENT_THRESHOLD_SELL:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'avg_polarity': avg_polarity,
            'avg_confidence': avg_confidence,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'overall_sentiment': overall_sentiment,
            'total_headlines': len(headlines)
        }
    
    def get_signal_strength(self, sentiment_score: float) -> str:
        """
        Get signal strength based on sentiment score
        
        Args:
            sentiment_score: Sentiment polarity score (-1 to 1)
            
        Returns:
            Signal strength label
        """
        abs_score = abs(sentiment_score)
        
        if abs_score > 0.8:
            return 'very_strong'
        elif abs_score > 0.6:
            return 'strong'
        elif abs_score > 0.4:
            return 'moderate'
        elif abs_score > 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def clear_cache(self):
        """Clear sentiment cache"""
        self.sentiment_cache.clear()
        logger.info("Sentiment cache cleared")
