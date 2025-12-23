"""
Data Manager for BTC Trading Bot
Automated data downloading, validation, and caching
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import requests
import time


class DataManager:
    """
    Professional data management system:
    - Download historical data from Binance
    - Validate data quality
    - Cache for performance
    - Incremental updates
    """
    
    def __init__(self, data_dir="data", cache_enabled=True):
        self.data_dir = data_dir
        self.cache_enabled = cache_enabled
        
        # Create data directory if needed
        os.makedirs(data_dir, exist_ok=True)
    
    def download_binance_data(self, 
                             symbol="BTCUSDT",
                             interval="1h",
                             start_date=None,
                             end_date=None,
                             limit=1000):
        """
        Download historical data from Binance API
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Start date (datetime or string)
            end_date: End date (datetime or string)
            limit: Number of candles per request
        
        Returns:
            DataFrame with OHLCV data
        """
        
        base_url = "https://api.binance.com/api/v3/klines"
        
        # Convert dates to timestamps
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            start_ms = int(start_date.timestamp() * 1000)
        else:
            # Default to 1 year ago
            start_ms = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
        
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            end_ms = int(end_date.timestamp() * 1000)
        else:
            end_ms = int(datetime.now().timestamp() * 1000)
        
        all_data = []
        current_start = start_ms
        
        print(f"Downloading {symbol} {interval} data...")
        
        while current_start < end_ms:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ms,
                'limit': limit
            }
            
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # Update start time for next batch
                current_start = data[-1][0] + 1  # Next ms after last candle
                
                print(f"  Downloaded {len(all_data)} candles...", end='\r')
                
                # Rate limiting
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"\n❌ Error downloading data: {e}")
                break
        
        print(f"\n✅ Downloaded {len(all_data)} total candles")
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Keep only essential columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
        else:
            return pd.DataFrame()
    
    def validate_data(self, df):
        """
        Validate data quality
        
        Checks for:
        - Missing values
        - Duplicate timestamps
        - Price anomalies
        - Gaps in data
        
        Returns:
            (is_valid: bool, issues: list)
        """
        
        issues = []
        
        # Check for missing values
        if df.isnull().any().any():
            issues.append("Missing values detected")
        
        # Check for duplicates
        if df.duplicated(subset=['timestamp']).any():
            issues.append("Duplicate timestamps detected")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        if (df[price_cols] <= 0).any().any():
            issues.append("Negative or zero prices detected")
        
        # Check OHLC consistency
        if ((df['high'] < df['low']) | 
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])).any():
            issues.append("OHLC consistency violations detected")
        
        # Check for large gaps (for hourly data, gap > 2 hours is suspicious)
        if 'timestamp' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff()
            # For hourly data, expect ~1 hour between candles
            # Allow up to 2 hours for occasional gaps
            expected_diff = pd.Timedelta(hours=1)
            max_allowed_diff = pd.Timedelta(hours=2)
            
            large_gaps = time_diffs[time_diffs > max_allowed_diff]
            if len(large_gaps) > 0:
                issues.append(f"Data gaps detected: {len(large_gaps)} gaps larger than 2 hours")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
    
    def save_data(self, df, filename):
        """Save data to CSV"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"💾 Data saved to: {filepath}")
    
    def load_data(self, filename):
        """Load data from CSV"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        df = pd.read_csv(filepath)
        
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def update_data(self, filename, symbol="BTCUSDT", interval="1h"):
        """
        Update existing data file with new candles
        
        Args:
            filename: CSV filename in data directory
            symbol: Trading pair
            interval: Timeframe
        
        Returns:
            Updated DataFrame
        """
        
        # Load existing data
        existing_df = self.load_data(filename)
        
        if existing_df is None or len(existing_df) == 0:
            print("No existing data found, downloading full history...")
            df = self.download_binance_data(symbol, interval)
        else:
            # Get last timestamp
            last_timestamp = pd.to_datetime(existing_df['timestamp'].max())
            print(f"Last data point: {last_timestamp}")
            
            # Download new data
            new_df = self.download_binance_data(
                symbol, interval,
                start_date=last_timestamp,
                end_date=datetime.now()
            )
            
            if len(new_df) > 0:
                # Combine and remove duplicates
                df = pd.concat([existing_df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=['timestamp'], keep='last')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                print(f"✅ Added {len(new_df)} new candles")
            else:
                df = existing_df
                print("ℹ️  No new data available")
        
        # Validate
        is_valid, issues = self.validate_data(df)
        
        if not is_valid:
            print("⚠️  Data validation warnings:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Save
        self.save_data(df, filename)
        
        return df


if __name__ == "__main__":
    # Example usage
    dm = DataManager()
    
    # Download BTC data
    df = dm.download_binance_data(
        symbol="BTCUSDT",
        interval="1h",
        start_date="2023-01-01",
        end_date=datetime.now()
    )
    
    # Validate
    is_valid, issues = dm.validate_data(df)
    print(f"\nData validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # Save
    dm.save_data(df, "btc_1h.csv")
