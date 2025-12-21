"""
Python script to install all dependencies
"""

import subprocess
import sys

def install_requirements():
    """Install all required packages"""
    print("=" * 70)
    print("Installing Trading Bot Dependencies")
    print("=" * 70)
    print()
    
    try:
        # Upgrade pip first
        print("📦 Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("\n📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("\n" + "=" * 70)
        print("✅ Installation Complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Copy .env.example to .env")
        print("2. Add your API keys to .env")
        print("3. Run: python run_backtest.py")
        print("4. Launch dashboard: streamlit run dashboard/app.py")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
