"""
Binance API Setup and Test Utility
Helps configure and test Binance API connection
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.exchange.binance_client import AdvancedBinanceClient
from core.monitoring.logger import get_logger


logger = get_logger()


def check_env_file():
    """Check if .env file exists"""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("❌ .env file not found!")
        print("\n📝 Creating .env file from .env.example...")
        
        example_path = Path('.env.example')
        if example_path.exists():
            with open(example_path, 'r') as f:
                content = f.read()
            
            with open('.env', 'w') as f:
                f.write(content)
            
            print("✅ Created .env file")
            print("\n⚠️  IMPORTANT: Edit .env and add your Binance API credentials!")
            print("   1. Open .env in a text editor")
            print("   2. Replace 'your_api_key_here' with your actual API key")
            print("   3. Replace 'your_api_secret_here' with your actual API secret")
            print("   4. Make sure USE_TESTNET=true for testing")
            return False
        else:
            print("❌ .env.example not found either!")
            return False
    
    return True


def test_binance_connection(use_testnet: bool = True):
    """
    Test Binance API connection
    
    Args:
        use_testnet: Test with testnet if True
    """
    print("=" * 70)
    print("🚀 BINANCE API CONNECTION TEST")
    print("=" * 70)
    
    if use_testnet:
        print("\n🧪 Testing with TESTNET (safe)")
    else:
        print("\n⚠️  Testing with MAINNET (REAL MONEY!)")
    
    print("\n" + "-" * 70)
    
    try:
        # Create client
        print("\n1. Initializing Binance client...")
        client = AdvancedBinanceClient(use_testnet=use_testnet)
        print("   ✅ Client initialized")
        
        # Test connection
        print("\n2. Testing API connection...")
        if not client.test_connection():
            print("   ❌ Connection test failed!")
            print("\n📝 Troubleshooting:")
            print("   1. Check your API keys in .env file")
            print("   2. Verify API key permissions on Binance")
            print("   3. Check if you're using correct network (testnet/mainnet)")
            print("   4. Ensure API key IP restrictions allow your IP")
            return False
        
        print("   ✅ Connection successful!")
        
        # Get account info
        print("\n3. Fetching account information...")
        account = client.get_account_info()
        
        if account:
            print("   ✅ Account access successful")
            print(f"\n   💰 Wallet Balance: ${account.get('total_wallet_balance', 0):.2f} USDT")
            print(f"   💵 Available: ${account.get('available_balance', 0):.2f} USDT")
            print(f"   🔒 In Use: ${account.get('used_balance', 0):.2f} USDT")
            
            positions = account.get('positions', [])
            if positions:
                print(f"\n   📊 Open Positions: {len(positions)}")
                for pos in positions:
                    print(f"      {pos.get('symbol')}: {pos.get('positionAmt')} ")
            else:
                print("\n   📊 No open positions")
        
        # Test rate limiter
        print("\n4. Checking rate limiter...")
        stats = client.get_rate_limit_stats()
        print(f"   ✅ Rate limiter active")
        print(f"   📊 Requests this minute: {stats['requests_last_minute']}/{stats['rpm_limit']}")
        print(f"   ⚖️  Current weight: {stats['current_weight']}/{stats['max_weight']}")
        
        # Test market data
        print("\n5. Testing market data access...")
        try:
            ticker = client.exchange.fetch_ticker('BTC/USDT')
            print(f"   ✅ BTC/USDT Price: ${ticker['last']:,.2f}")
            print(f"   📈 24h Change: {ticker['percentage']:.2f}%")
            print(f"   📊 24h Volume: ${ticker['quoteVolume']:,.0f}")
        except Exception as e:
            print(f"   ⚠️  Market data access: {e}")
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        
        print("\n🎉 Your Binance API is configured correctly!")
        
        if use_testnet:
            print("\n📝 Next Steps:")
            print("   1. Test trading strategies on testnet")
            print("   2. Run backtests: python run_backtest.py")
            print("   3. Launch dashboard: streamlit run dashboard/app.py")
            print("   4. When ready, switch  to mainnet in .env (USE_TESTNET=false)")
        else:
            print("\n⚠️  CAUTION: You're connected to MAINNET")
            print("   - Start with small capital")
            print("   - Monitor all trades closely")
            print("   - Use stop-losses on every position")
            print("   - Enable all alerts")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📝 Common Issues:")
        print("   1. Invalid API keys - check .env file")
        print("   2. API key permissions - enable 'Enable Futures' on Binance")
        print("   3. IP restriction - whitelist your IP or disable restriction")
        print("   4. Wrong network - testnet keys don't work on mainnet (and vice versa)")
        
        import traceback
        print("\n🔍 Full error trace:")
        traceback.print_exc()
        
        return False


def setup_binance_api():
    """Interactive setup wizard"""
    print("=" * 70)
    print("🔧 BINANCE API SETUP WIZARD")
    print("=" * 70)
    
    print("\n📝 This wizard will help you set up your Binance API access.")
    
    # Step 1: Check .env
    print("\nStep 1: Checking environment configuration...")
    if not check_env_file():
        print("\n⏸️  Setup paused. Please configure .env file and run again.")
        return
    
    print("✅ .env file found")
    
    # Step 2: Choose network
    print("\nStep 2: Choose network")
    print("   1. Testnet (recommended for testing)")
    print("   2. Mainnet (REAL trading with REAL money)")
    
    choice = input("\nYour choice (1 or 2): ").strip()
    
    use_testnet = choice == "1"
    
    if not use_testnet:
        confirm = input("\n⚠️  WARNING: You selected MAINNET. Type 'YES' to confirm: ").strip()
        if confirm != "YES":
            print("Setup cancelled.")
            return
    
    # Step 3: Test connection
    print("\nStep 3: Testing API connection...")
    success = test_binance_connection(use_testnet=use_testnet)
    
    if success:
        print("\n" + "🎉 " * 10)
        print("\nSETUP COMPLETE! Your trading bot is ready to use.")
        print("\n" + "🎉 " * 10)
    else:
        print("\n" + "❌ " * 10)
        print("\nSetup failed. Please fix the errors and try again.")
        print("\n" + "❌ " * 10)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Binance API Setup and Test')
    parser.add_argument('--setup', action='store_true', help='Run setup wizard')
    parser.add_argument('--test', action='store_true', help='Test API connection')
    parser.add_argument('--mainnet', action='store_true', help='Use mainnet instead of testnet')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_binance_api()
    elif args.test:
        use_testnet = not args.mainnet
        test_binance_connection(use_testnet=use_testnet)
    else:
        # Default: run setup
        setup_binance_api()
