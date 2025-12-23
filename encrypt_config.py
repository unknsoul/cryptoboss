"""
API Key Encryption Utility
Securely encrypt your Binance API keys
Run this script to convert plain text API keys to encrypted format
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.security.secure_config import SecureConfigManager


def encrypt_api_keys():
    """
    Interactive script to encrypt API keys
    Updates .env file with encrypted keys
    """
    print("=" * 70)
    print("API KEY ENCRYPTION UTILITY")
    print("=" * 70)
    print()
    print("This utility will encrypt your Binance API keys for secure storage.")
    print("The encryption key will be stored in .keyfile (keep this safe!)")
    print()
    
    # Check if .env exists
    env_path = Path('.env')
    if not env_path.exists():
        print("⚠️  .env file not found. Please create it first.")
        print("You can copy from .env.example if available.")
        return
    
    # Load current .env
    from dotenv import load_dotenv, set_key
    load_dotenv()
    
    # Get current API keys
    current_api_key = os.getenv('BINANCE_API_KEY', '')
    current_api_secret = os.getenv('BINANCE_API_SECRET', '')
    
    # Check if already encrypted
    encrypted_key_exists = os.getenv('BINANCE_API_KEY_ENCRYPTED', '')
    
    if encrypted_key_exists:
        print("✓ Encrypted keys already exist in .env")
        response = input("Do you want to re-encrypt with new keys? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Prompt for API keys (or use existing)
    print("\nEnter your API credentials:")
    print("(Press Enter to use existing values from .env)")
    print()
    
    api_key = input(f"API Key [{current_api_key[:10]}...]: ").strip()
    if not api_key:
        api_key = current_api_key
    
    api_secret = input(f"API Secret [{current_api_secret[:10]}...]: ").strip()
    if not api_secret:
        api_secret = current_api_secret
    
    if not api_key or not api_secret:
        print("\n✗ Error: API key and secret are required.")
        return
    
    # Create secure config manager
    print("\n🔐 Encrypting API keys...")
    manager = SecureConfigManager()
    
    # Encrypt keys
    encrypted_key, encrypted_secret = manager.encrypt_api_keys(api_key, api_secret)
    
    print("✓ Encryption complete!")
    print()
    print(f"Encrypted API Key: {encrypted_key[:60]}...")
    print(f"Encrypted API Secret: {encrypted_secret[:60]}...")
    print()
    
    # Update .env file
    print("📝 Updating .env file...")
    
    # Add encrypted keys
    set_key(env_path, 'BINANCE_API_KEY_ENCRYPTED', encrypted_key)
    set_key(env_path, 'BINANCE_API_SECRET_ENCRYPTED', encrypted_secret)
    
    # Comment out plain text keys (don't delete for safety)
    env_content = env_path.read_text()
    
    if 'BINANCE_API_KEY=' in env_content and 'BINANCE_API_KEY_ENCRYPTED' not in env_content:
        env_content = env_content.replace(
            f'BINANCE_API_KEY={current_api_key}',
            f'# BINANCE_API_KEY={current_api_key}  # ENCRYPTED - Use BINANCE_API_KEY_ENCRYPTED instead'
        )
    
    if 'BINANCE_API_SECRET=' in env_content:
        env_content = env_content.replace(
            f'BINANCE_API_SECRET={current_api_secret}',
            f'# BINANCE_API_SECRET={current_api_secret}  # ENCRYPTED - Use BINANCE_API_SECRET_ENCRYPTED instead'
        )
    
    env_path.write_text(env_content)
    
    print("✓ .env file updated!")
    print()
    print("=" * 70)
    print("ENCRYPTION SUCCESSFUL")
    print("=" * 70)
    print()
    print("✓ Encryption key saved to: .keyfile")
    print("✓ Encrypted keys saved to: .env")
    print()
    print("📌 IMPORTANT:")
    print("  1. Keep .keyfile safe - it's needed to decrypt your keys")
    print("  2. Add .keyfile to .gitignore (should already be there)")
    print("  3. Never commit your .env or .keyfile to version control")
    print("  4. Backup .keyfile securely if deploying to production")
    print()
    print("Your API keys are now encrypted and ready to use!")
    print("The trading bot will automatically decrypt them when loaded.")
    print()


if __name__ == '__main__':
    try:
        encrypt_api_keys()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
