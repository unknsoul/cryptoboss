"""
Sentiment Analysis Module
Integrates market sentiment into trading decisions

Features:
- Fear & Greed Index
- News sentiment scoring
- Social media sentiment (simulated)
- Sentiment-adjusted signal confidence
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import random

try:
    from src.core.monitoring.logger import get_logger
except ImportError:
    from core.monitoring.logger import get_logger

logger = get_logger()


class SentimentLevel(Enum):
    """Market sentiment levels"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class SentimentData:
    """Container for sentiment analysis results"""
    fear_greed_index: int          # 0-100 scale
    sentiment_level: SentimentLevel
    news_sentiment: float          # -1 to 1
    social_sentiment: float        # -1 to 1
    timestamp: datetime
    confidence: float              # 0-1, how reliable the data is


class SentimentAnalyzer:
    """
    Market Sentiment Analysis Engine
    
    Combines multiple sentiment sources:
    - Fear & Greed Index
    - News headline sentiment
    - Social media trends
    """
    
    def __init__(self, 
                 use_mock_data: bool = True,
                 fear_greed_api_key: Optional[str] = None):
        """
        Args:
            use_mock_data: Use simulated data (for testing)
            fear_greed_api_key: API key for fear & greed data (optional)
        """
        self.use_mock_data = use_mock_data
        self.api_key = fear_greed_api_key
        
        # Cache for sentiment data
        self.cache: Dict[str, Any] = {}
        self.cache_duration = timedelta(minutes=15)
        
        # Historical sentiment for trend analysis
        self.sentiment_history: List[SentimentData] = []
        
        logger.info(
            "SentimentAnalyzer initialized",
            use_mock=use_mock_data
        )
    
    def get_current_sentiment(self) -> SentimentData:
        """
        Get current market sentiment from all sources.
        
        Returns:
            SentimentData with aggregated sentiment
        """
        # Check cache
        if self._is_cache_valid():
            return self.cache['sentiment']
        
        if self.use_mock_data:
            sentiment = self._generate_mock_sentiment()
        else:
            sentiment = self._fetch_real_sentiment()
        
        # Cache and store history
        self.cache = {
            'sentiment': sentiment,
            'timestamp': datetime.now()
        }
        self.sentiment_history.append(sentiment)
        
        # Keep only last 100 readings
        if len(self.sentiment_history) > 100:
            self.sentiment_history = self.sentiment_history[-100:]
        
        return sentiment
    
    def get_fear_greed_index(self) -> int:
        """
        Get Fear & Greed Index (0-100)
        
        0-25: Extreme Fear
        25-45: Fear
        45-55: Neutral
        55-75: Greed
        75-100: Extreme Greed
        
        Returns:
            Fear & Greed Index value
        """
        sentiment = self.get_current_sentiment()
        return sentiment.fear_greed_index
    
    def get_news_sentiment(self) -> Dict[str, Any]:
        """
        Get sentiment from recent crypto news
        
        Returns:
            Aggregated news sentiment
        """
        if self.use_mock_data:
            return self._mock_news_sentiment()
        else:
            return self._fetch_news_sentiment()
    
    def adjust_signal_confidence(self, 
                                  base_confidence: float,
                                  signal_direction: str) -> float:
        """
        Adjust signal confidence based on sentiment.
        
        - In extreme fear, increase confidence for LONG signals (buy the fear)
        - In extreme greed, increase confidence for SHORT signals (sell the greed)
        
        Args:
            base_confidence: Original signal confidence (0-1)
            signal_direction: 'LONG' or 'SHORT'
        
        Returns:
            Adjusted confidence (0-1)
        """
        sentiment = self.get_current_sentiment()
        fg_index = sentiment.fear_greed_index
        
        adjustment = 0.0
        
        if signal_direction == 'LONG':
            if fg_index < 25:  # Extreme fear - good for buying
                adjustment = 0.15
                logger.info("📈 Extreme fear detected - boosting LONG confidence")
            elif fg_index < 45:  # Fear
                adjustment = 0.05
            elif fg_index > 75:  # Extreme greed - bad for buying
                adjustment = -0.15
                logger.info("⚠️ Extreme greed detected - reducing LONG confidence")
            elif fg_index > 55:  # Greed
                adjustment = -0.05
        
        elif signal_direction == 'SHORT':
            if fg_index > 75:  # Extreme greed - good for shorting
                adjustment = 0.15
                logger.info("📉 Extreme greed detected - boosting SHORT confidence")
            elif fg_index > 55:  # Greed
                adjustment = 0.05
            elif fg_index < 25:  # Extreme fear - bad for shorting
                adjustment = -0.15
                logger.info("⚠️ Extreme fear detected - reducing SHORT confidence")
            elif fg_index < 45:  # Fear
                adjustment = -0.05
        
        # Also consider news sentiment
        news_adjustment = sentiment.news_sentiment * 0.05
        if signal_direction == 'LONG':
            adjustment += news_adjustment
        else:
            adjustment -= news_adjustment
        
        adjusted = base_confidence + adjustment
        return max(0.0, min(1.0, adjusted))  # Clamp to [0, 1]
    
    def should_filter_signal(self, signal_direction: str) -> bool:
        """
        Check if sentiment suggests filtering out the signal.
        
        Extreme contrarian signals (e.g., LONG in extreme greed) 
        should be filtered unless very high confidence.
        
        Args:
            signal_direction: 'LONG' or 'SHORT'
        
        Returns:
            True if signal should be filtered out
        """
        sentiment = self.get_current_sentiment()
        
        # Filter LONG signals in extreme greed
        if signal_direction == 'LONG' and sentiment.fear_greed_index > 85:
            logger.warning("🚫 Signal filtered: LONG during extreme greed (F&G > 85)")
            return True
        
        # Filter SHORT signals in extreme fear
        if signal_direction == 'SHORT' and sentiment.fear_greed_index < 15:
            logger.warning("🚫 Signal filtered: SHORT during extreme fear (F&G < 15)")
            return True
        
        return False
    
    def get_sentiment_trend(self) -> str:
        """
        Analyze trend in sentiment over recent history.
        
        Returns:
            'improving', 'worsening', or 'stable'
        """
        if len(self.sentiment_history) < 5:
            return 'stable'
        
        recent = [s.fear_greed_index for s in self.sentiment_history[-5:]]
        older = [s.fear_greed_index for s in self.sentiment_history[-10:-5]] if len(self.sentiment_history) >= 10 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        diff = recent_avg - older_avg
        
        if diff > 5:
            return 'improving'
        elif diff < -5:
            return 'worsening'
        else:
            return 'stable'
    
    def _is_cache_valid(self) -> bool:
        """Check if cached sentiment is still valid"""
        if 'timestamp' not in self.cache:
            return False
        return datetime.now() - self.cache['timestamp'] < self.cache_duration
    
    def _generate_mock_sentiment(self) -> SentimentData:
        """Generate realistic mock sentiment data"""
        # Simulate Fear & Greed with some momentum
        if self.sentiment_history:
            last_fg = self.sentiment_history[-1].fear_greed_index
            change = np.random.normal(0, 5)
            fg_index = int(np.clip(last_fg + change, 0, 100))
        else:
            fg_index = random.randint(30, 70)
        
        # Determine level
        if fg_index < 25:
            level = SentimentLevel.EXTREME_FEAR
        elif fg_index < 45:
            level = SentimentLevel.FEAR
        elif fg_index < 55:
            level = SentimentLevel.NEUTRAL
        elif fg_index < 75:
            level = SentimentLevel.GREED
        else:
            level = SentimentLevel.EXTREME_GREED
        
        # Mock news and social sentiment
        news_sentiment = np.random.uniform(-0.5, 0.5)
        social_sentiment = np.random.uniform(-0.3, 0.3)
        
        return SentimentData(
            fear_greed_index=fg_index,
            sentiment_level=level,
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            timestamp=datetime.now(),
            confidence=0.8  # Mock data confidence
        )
    
    def _fetch_real_sentiment(self) -> SentimentData:
        """
        Fetch real sentiment data from APIs.
        
        In production, this would call:
        - alternative.me Fear & Greed API
        - News APIs (Cryptopanic, etc.)
        - Social APIs (Twitter/X, Reddit)
        """
        # Placeholder - implement actual API calls
        logger.warning("Real sentiment API not implemented, using mock data")
        return self._generate_mock_sentiment()
    
    def _mock_news_sentiment(self) -> Dict[str, Any]:
        """Generate mock news sentiment"""
        headlines = [
            {"headline": "Bitcoin rallies on ETF approval hopes", "sentiment": 0.7},
            {"headline": "Crypto market sees increased institutional interest", "sentiment": 0.5},
            {"headline": "Regulatory concerns weigh on crypto prices", "sentiment": -0.3},
            {"headline": "Major exchange reports record trading volume", "sentiment": 0.4},
            {"headline": "Whale wallet moves large Bitcoin holdings", "sentiment": -0.2}
        ]
        
        # Return random subset
        selected = random.sample(headlines, 3)
        avg_sentiment = np.mean([h['sentiment'] for h in selected])
        
        return {
            'headlines': selected,
            'average_sentiment': avg_sentiment,
            'timestamp': datetime.now().isoformat()
        }
    
    def _fetch_news_sentiment(self) -> Dict[str, Any]:
        """Fetch real news sentiment - placeholder"""
        return self._mock_news_sentiment()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get sentiment analysis summary"""
        current = self.get_current_sentiment()
        
        return {
            'fear_greed_index': current.fear_greed_index,
            'sentiment_level': current.sentiment_level.value,
            'news_sentiment': round(current.news_sentiment, 2),
            'social_sentiment': round(current.social_sentiment, 2),
            'trend': self.get_sentiment_trend(),
            'data_points': len(self.sentiment_history),
            'confidence': current.confidence
        }


if __name__ == "__main__":
    print("=" * 70)
    print("📊 SENTIMENT ANALYZER TEST")
    print("=" * 70)
    
    analyzer = SentimentAnalyzer(use_mock_data=True)
    
    # Test basic sentiment
    print("\n1. Current Sentiment:")
    sentiment = analyzer.get_current_sentiment()
    print(f"   Fear & Greed Index: {sentiment.fear_greed_index}")
    print(f"   Level: {sentiment.sentiment_level.value}")
    print(f"   News Sentiment: {sentiment.news_sentiment:.2f}")
    
    # Test signal adjustment
    print("\n2. Signal Confidence Adjustment:")
    base_conf = 0.70
    
    long_adj = analyzer.adjust_signal_confidence(base_conf, 'LONG')
    short_adj = analyzer.adjust_signal_confidence(base_conf, 'SHORT')
    
    print(f"   Base Confidence: {base_conf:.2f}")
    print(f"   LONG Adjusted: {long_adj:.2f}")
    print(f"   SHORT Adjusted: {short_adj:.2f}")
    
    # Test news sentiment
    print("\n3. News Sentiment:")
    news = analyzer.get_news_sentiment()
    for h in news['headlines']:
        emoji = "📈" if h['sentiment'] > 0 else "📉" if h['sentiment'] < 0 else "➡️"
        print(f"   {emoji} {h['headline'][:50]}... ({h['sentiment']:+.1f})")
    
    # Generate history for trend
    print("\n4. Generating sentiment history...")
    for _ in range(10):
        analyzer.get_current_sentiment()
        analyzer.cache = {}  # Force new data
    
    print(f"   Trend: {analyzer.get_sentiment_trend()}")
    
    # Summary
    print("\n5. Summary:")
    summary = analyzer.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ Sentiment Analyzer Test Complete")
    print("=" * 70)
