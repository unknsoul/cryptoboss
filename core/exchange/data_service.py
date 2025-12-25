"""
Exchange Data Service - Unified API for Market Data

This provides a clean interface for fetching public market data
from Binance without requiring API keys. The bot should use THIS
instead of raw requests.get() calls.
"""
import requests
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants
BINANCE_BASE_URL = "https://api.binance.com/api/v3"
REQUEST_TIMEOUT = 10  # seconds


class ExchangeDataService:
    """
    Unified service for fetching public market data.
    
    Usage:
        from core.exchange.data_service import get_data_service
        service = get_data_service()
        klines = service.fetch_klines("BTCUSDT", "5m", 300)
    """
    
    def __init__(self, base_url: str = BINANCE_BASE_URL):
        self.base_url = base_url
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def fetch_klines(
        self, 
        symbol: str = "BTCUSDT", 
        interval: str = "5m", 
        limit: int = 300
    ) -> Optional[pd.DataFrame]:
        """
        Fetch candlestick data from Binance.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe (e.g., "1m", "5m", "1h", "1d")
            limit: Number of candles (max 1000)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            Or None if request fails.
        """
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Raise for 4xx/5xx errors
            
            data = response.json()
            
            if not data:
                logger.warning(f"No klines data returned for {symbol}")
                return None
            
            # Parse into DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Return only essential columns
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching klines for {symbol}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching klines: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for klines: {e}")
            return None
        except ValueError as e:
            logger.error(f"Failed to parse klines response: {e}")
            return None
    
    def fetch_current_price(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """
        Fetch current price for a symbol.
        
        Returns:
            Current price as float, or None if request fails.
        """
        try:
            url = f"{self.base_url}/ticker/price"
            params = {'symbol': symbol}
            
            response = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            return float(data['price'])
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching price for {symbol}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching price: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse price response: {e}")
            return None
    
    def fetch_orderbook(
        self, 
        symbol: str = "BTCUSDT", 
        limit: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch order book for a symbol.
        
        Returns:
            Dict with 'bids', 'asks', 'bid_total', 'ask_total', 'imbalance'
            Or None if request fails.
        """
        try:
            url = f"{self.base_url}/depth"
            params = {'symbol': symbol, 'limit': limit}
            
            response = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            bids = [[float(p), float(q)] for p, q in data.get('bids', [])]
            asks = [[float(p), float(q)] for p, q in data.get('asks', [])]
            
            bid_total = sum(q for _, q in bids)
            ask_total = sum(q for _, q in asks)
            
            # Imbalance: positive = more bids (bullish), negative = more asks (bearish)
            imbalance = (bid_total - ask_total) / (bid_total + ask_total + 1e-10)
            
            return {
                'bids': bids,
                'asks': asks,
                'bid_total': bid_total,
                'ask_total': ask_total,
                'imbalance': imbalance,
                'spread': asks[0][0] - bids[0][0] if bids and asks else 0
            }
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching orderbook for {symbol}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching orderbook: {e}")
            return None
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Failed to parse orderbook response: {e}")
            return None
    
    def fetch_24hr_ticker(self, symbol: str = "BTCUSDT") -> Optional[Dict[str, Any]]:
        """Fetch 24-hour ticker statistics"""
        try:
            url = f"{self.base_url}/ticker/24hr"
            params = {'symbol': symbol}
            
            response = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'price_change': float(data['priceChange']),
                'price_change_pct': float(data['priceChangePercent']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'volume_24h': float(data['volume']),
                'quote_volume_24h': float(data['quoteVolume'])
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch 24hr ticker: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse 24hr ticker: {e}")
            return None


# Singleton instance
_data_service: Optional[ExchangeDataService] = None


def get_data_service() -> ExchangeDataService:
    """Get or create the singleton ExchangeDataService instance"""
    global _data_service
    if _data_service is None:
        _data_service = ExchangeDataService()
    return _data_service
