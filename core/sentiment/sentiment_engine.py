"""
Sentiment Analysis & News - Enterprise Features #250-260
News Sentiment, Social Media, Fear/Greed Index, Funding Rates.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import re

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """Feature #250: News Sentiment Analyzer"""
    
    POSITIVE_WORDS = ['bullish', 'surge', 'rally', 'gain', 'profit', 'growth', 'adoption', 'approve']
    NEGATIVE_WORDS = ['bearish', 'crash', 'drop', 'loss', 'ban', 'hack', 'fear', 'sell']
    
    def __init__(self):
        self.news_history: List[Dict] = []
        logger.info("News Sentiment Analyzer initialized")
    
    def analyze_headline(self, headline: str) -> Dict:
        words = headline.lower().split()
        pos = sum(1 for w in words if any(p in w for p in self.POSITIVE_WORDS))
        neg = sum(1 for w in words if any(n in w for n in self.NEGATIVE_WORDS))
        
        score = (pos - neg) / max(len(words), 1)
        sentiment = 'BULLISH' if score > 0.1 else 'BEARISH' if score < -0.1 else 'NEUTRAL'
        
        return {'headline': headline[:50], 'sentiment': sentiment, 'score': round(score, 3)}
    
    def get_aggregate_sentiment(self, hours: int = 24) -> Dict:
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [n for n in self.news_history if datetime.fromisoformat(n['timestamp']) > cutoff]
        
        if not recent:
            return {'sentiment': 'NEUTRAL', 'score': 0}
        
        avg_score = sum(n['score'] for n in recent) / len(recent)
        return {'sentiment': 'BULLISH' if avg_score > 0 else 'BEARISH', 'score': round(avg_score, 3), 'count': len(recent)}


class SocialMediaSentiment:
    """Feature #251: Social Media Sentiment Tracker"""
    
    def __init__(self):
        self.mentions: Dict[str, List[Dict]] = {}
        logger.info("Social Media Sentiment initialized")
    
    def add_mention(self, symbol: str, sentiment: float, platform: str):
        if symbol not in self.mentions:
            self.mentions[symbol] = []
        self.mentions[symbol].append({'sentiment': sentiment, 'platform': platform, 'time': datetime.now().isoformat()})
        self.mentions[symbol] = self.mentions[symbol][-1000:]
    
    def get_sentiment(self, symbol: str) -> Dict:
        if symbol not in self.mentions or not self.mentions[symbol]:
            return {'sentiment': 0, 'volume': 0}
        recent = self.mentions[symbol][-100:]
        avg = sum(m['sentiment'] for m in recent) / len(recent)
        return {'sentiment': round(avg, 3), 'volume': len(recent), 'trend': 'UP' if avg > 0 else 'DOWN'}


class FearGreedIndex:
    """Feature #252: Crypto Fear & Greed Index"""
    
    def __init__(self):
        self.history: List[Dict] = []
        logger.info("Fear & Greed Index initialized")
    
    def calculate(self, volatility: float, momentum: float, social_volume: float, dominance: float) -> Dict:
        vol_score = max(0, 50 - volatility * 1000)
        mom_score = 50 + momentum * 100
        social_score = min(100, social_volume * 10)
        dom_score = dominance
        
        index = (vol_score * 0.25 + mom_score * 0.25 + social_score * 0.25 + dom_score * 0.25)
        index = max(0, min(100, index))
        
        if index >= 75:
            label = 'EXTREME_GREED'
        elif index >= 55:
            label = 'GREED'
        elif index >= 45:
            label = 'NEUTRAL'
        elif index >= 25:
            label = 'FEAR'
        else:
            label = 'EXTREME_FEAR'
        
        self.history.append({'index': index, 'label': label, 'timestamp': datetime.now().isoformat()})
        return {'index': round(index), 'label': label}


class FundingRateAnalyzer:
    """Feature #253: Funding Rate Analyzer"""
    
    def __init__(self):
        self.rates: Dict[str, List[float]] = {}
        logger.info("Funding Rate Analyzer initialized")
    
    def add_rate(self, symbol: str, rate: float):
        if symbol not in self.rates:
            self.rates[symbol] = []
        self.rates[symbol].append(rate)
        self.rates[symbol] = self.rates[symbol][-100:]
    
    def analyze(self, symbol: str) -> Dict:
        if symbol not in self.rates or not self.rates[symbol]:
            return {'signal': 'NEUTRAL'}
        
        current = self.rates[symbol][-1]
        avg = sum(self.rates[symbol]) / len(self.rates[symbol])
        
        if current > 0.05:
            signal = 'BEARISH'  # Longs paying heavily
        elif current < -0.05:
            signal = 'BULLISH'  # Shorts paying heavily
        else:
            signal = 'NEUTRAL'
        
        return {'current': round(current, 4), 'avg': round(avg, 4), 'signal': signal}


class OpenInterestTracker:
    """Feature #254: Open Interest Tracker"""
    
    def __init__(self):
        self.oi_history: Dict[str, List[float]] = {}
        logger.info("Open Interest Tracker initialized")
    
    def update(self, symbol: str, oi: float):
        if symbol not in self.oi_history:
            self.oi_history[symbol] = []
        self.oi_history[symbol].append(oi)
        self.oi_history[symbol] = self.oi_history[symbol][-200:]
    
    def get_trend(self, symbol: str) -> Dict:
        if symbol not in self.oi_history or len(self.oi_history[symbol]) < 10:
            return {'trend': 'UNKNOWN'}
        
        oi = self.oi_history[symbol]
        recent = sum(oi[-10:]) / 10
        older = sum(oi[-20:-10]) / 10 if len(oi) >= 20 else recent
        
        change = (recent - older) / older * 100 if older > 0 else 0
        trend = 'INCREASING' if change > 5 else 'DECREASING' if change < -5 else 'STABLE'
        
        return {'current': oi[-1], 'change_pct': round(change, 2), 'trend': trend}


class LiquidationTracker:
    """Feature #255: Liquidation Tracker"""
    
    def __init__(self):
        self.liquidations: List[Dict] = []
        logger.info("Liquidation Tracker initialized")
    
    def add_liquidation(self, symbol: str, side: str, size: float, price: float):
        self.liquidations.append({
            'symbol': symbol, 'side': side, 'size': size, 'price': price,
            'timestamp': datetime.now().isoformat()
        })
        self.liquidations = self.liquidations[-1000:]
    
    def get_recent_liquidations(self, hours: int = 1) -> Dict:
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [l for l in self.liquidations if datetime.fromisoformat(l['timestamp']) > cutoff]
        
        long_liqs = sum(l['size'] for l in recent if l['side'] == 'LONG')
        short_liqs = sum(l['size'] for l in recent if l['side'] == 'SHORT')
        
        return {'long_liquidations': long_liqs, 'short_liquidations': short_liqs, 'count': len(recent)}


class WhaleWatcher:
    """Feature #256: Whale Activity Watcher"""
    
    def __init__(self, whale_threshold: float = 100000):
        self.threshold = whale_threshold
        self.whale_txs: List[Dict] = []
        logger.info(f"Whale Watcher initialized - Threshold: ${whale_threshold:,.0f}")
    
    def add_transaction(self, symbol: str, side: str, amount_usd: float):
        if amount_usd >= self.threshold:
            self.whale_txs.append({
                'symbol': symbol, 'side': side, 'amount_usd': amount_usd,
                'timestamp': datetime.now().isoformat()
            })
            self.whale_txs = self.whale_txs[-500:]
    
    def get_whale_activity(self, hours: int = 24) -> Dict:
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [t for t in self.whale_txs if datetime.fromisoformat(t['timestamp']) > cutoff]
        
        buy_volume = sum(t['amount_usd'] for t in recent if t['side'] == 'BUY')
        sell_volume = sum(t['amount_usd'] for t in recent if t['side'] == 'SELL')
        
        return {
            'buy_volume': buy_volume, 'sell_volume': sell_volume,
            'net_flow': buy_volume - sell_volume,
            'signal': 'BULLISH' if buy_volume > sell_volume * 1.5 else 'BEARISH' if sell_volume > buy_volume * 1.5 else 'NEUTRAL'
        }


class MarketMomentumIndex:
    """Feature #257: Market Momentum Index"""
    
    def __init__(self):
        self.momentum_history: List[float] = []
        logger.info("Market Momentum Index initialized")
    
    def calculate(self, prices: List[float]) -> Dict:
        if len(prices) < 20:
            return {'momentum': 0, 'signal': 'NEUTRAL'}
        
        roc_10 = (prices[-1] - prices[-10]) / prices[-10] * 100 if prices[-10] != 0 else 0
        roc_20 = (prices[-1] - prices[-20]) / prices[-20] * 100 if prices[-20] != 0 else 0
        
        momentum = (roc_10 * 0.6 + roc_20 * 0.4)
        self.momentum_history.append(momentum)
        
        signal = 'BULLISH' if momentum > 2 else 'BEARISH' if momentum < -2 else 'NEUTRAL'
        return {'momentum': round(momentum, 2), 'roc_10': round(roc_10, 2), 'signal': signal}


class ExchangeFlowTracker:
    """Feature #258: Exchange Flow Tracker"""
    
    def __init__(self):
        self.flows: List[Dict] = []
        logger.info("Exchange Flow Tracker initialized")
    
    def add_flow(self, direction: str, amount: float, exchange: str):
        self.flows.append({'direction': direction, 'amount': amount, 'exchange': exchange, 'timestamp': datetime.now().isoformat()})
        self.flows = self.flows[-1000:]
    
    def get_net_flow(self, hours: int = 24) -> Dict:
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [f for f in self.flows if datetime.fromisoformat(f['timestamp']) > cutoff]
        
        inflows = sum(f['amount'] for f in recent if f['direction'] == 'IN')
        outflows = sum(f['amount'] for f in recent if f['direction'] == 'OUT')
        
        net = inflows - outflows
        return {'inflows': inflows, 'outflows': outflows, 'net_flow': net, 'signal': 'BEARISH' if net > 0 else 'BULLISH'}


class OnChainAnalyzer:
    """Feature #259: On-Chain Data Analyzer"""
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        logger.info("On-Chain Analyzer initialized")
    
    def update_metric(self, name: str, value: float):
        self.metrics[name] = value
    
    def get_health_score(self) -> Dict:
        if not self.metrics:
            return {'score': 50, 'signal': 'NEUTRAL'}
        
        score = 50
        if self.metrics.get('active_addresses', 0) > 100000:
            score += 10
        if self.metrics.get('hash_rate_change', 0) > 0:
            score += 10
        if self.metrics.get('exchange_reserve_change', 0) < 0:
            score += 10
        
        return {'score': min(100, score), 'signal': 'BULLISH' if score > 60 else 'BEARISH' if score < 40 else 'NEUTRAL'}


class SentimentAggregator:
    """Feature #260: Multi-Source Sentiment Aggregator"""
    
    def __init__(self):
        self.sources: Dict[str, Dict] = {}
        logger.info("Sentiment Aggregator initialized")
    
    def add_source(self, name: str, sentiment: float, weight: float = 1.0):
        self.sources[name] = {'sentiment': sentiment, 'weight': weight, 'updated': datetime.now().isoformat()}
    
    def get_aggregate(self) -> Dict:
        if not self.sources:
            return {'sentiment': 0, 'signal': 'NEUTRAL'}
        
        total_weight = sum(s['weight'] for s in self.sources.values())
        weighted_sum = sum(s['sentiment'] * s['weight'] for s in self.sources.values())
        
        agg = weighted_sum / total_weight if total_weight > 0 else 0
        return {'sentiment': round(agg, 3), 'signal': 'BULLISH' if agg > 0.2 else 'BEARISH' if agg < -0.2 else 'NEUTRAL', 'sources': len(self.sources)}


# Singletons
def get_news_sentiment(): return NewsSentimentAnalyzer()
def get_social_sentiment(): return SocialMediaSentiment()
def get_fear_greed(): return FearGreedIndex()
def get_funding_analyzer(): return FundingRateAnalyzer()
def get_oi_tracker(): return OpenInterestTracker()
def get_liquidation_tracker(): return LiquidationTracker()
def get_whale_watcher(): return WhaleWatcher()
def get_momentum_index(): return MarketMomentumIndex()
def get_exchange_flow(): return ExchangeFlowTracker()
def get_onchain(): return OnChainAnalyzer()
def get_sentiment_aggregator(): return SentimentAggregator()
