"""
Download BTC Historical Data for Backtesting
Uses ccxt to download OHLCV data from Binance
"""

import pandas as pd
import ccxt
from datetime import datetime, timedelta
import time
import os

def download_btc_data(
    symbol='BTC/USDT',
    timeframe='1h',
    days_back=730,  # 2 years
    output_file='data/btc_1h.csv'
):
    """
    Download BTC historical data from Binance.
    
    Args:
        symbol: Trading pair
        timeframe: Candle timeframe (1h, 4h, 1d)  
        days_back: How many days of history
        output_file: Where to save CSV
    """
    print(f"Downloading {days_back} days of {symbol} {timeframe} data...")
    
    # Initialize Binance exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    # Calculate start time
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    since = int(start_time.timestamp() * 1000)
    
    all_candles = []
    
    while True:
        try:
            # Fetch OHLCV data
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            
            if not candles:
                break
            
            all_candles.extend(candles)
            
            # Update since to last candle time
            since = candles[-1][0] + 1
            
            print(f"Downloaded {len(all_candles)} candles... (latest: {datetime.fromtimestamp(candles[-1][0]/1000)})")
            
            # Check if we've reached current time
            if candles[-1][0] >= int(end_time.timestamp() * 1000):
                break
            
            # Rate limiting
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Downloaded {len(df)} candles")
    print(f"📅 Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"💾 Saved to: {output_file}")
    print(f"📊 First price: ${df['close'].iloc[0]:,.2f}")
    print(f"📊 Last price: ${df['close'].iloc[-1]:,.2f}")
    print(f"📊 Return: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.2f}%")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download BTC historical data')
    parser.add_argument('--days', type=int, default=730, help='Days of history')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (1h, 4h, 1d)')
    parser.add_argument('--output', type=str, default='data/btc_1h.csv', help='Output file')
    
    args = parser.parse_args()
    
    df = download_btc_data(
        days_back=args.days,
        timeframe=args.timeframe,
        output_file=args.output
    )
