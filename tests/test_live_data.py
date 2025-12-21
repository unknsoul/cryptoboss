
import time
from core.exchange.binance_client import BinanceClient

def test_live_data():
    print("Initializing Binance Client...")
    client = BinanceClient()
    
    print("Connecting to WebSocket...")
    client.connect()
    
    # Wait for connection
    time.sleep(2)
    
    print("Subscribing to BTC/USDT ticker...")
    client.subscribe_ticker("BTCUSDT")
    
    def on_ticker(data):
        print(f"Update: {data['symbol']} Price: ${data['price']:,.2f}")
        
    client.on('ticker', on_ticker)
    
    print("Listening for 10 seconds...")
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass
        
    print("Disconnecting...")
    client.disconnect()
    print("Done.")

if __name__ == "__main__":
    test_live_data()
