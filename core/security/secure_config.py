"""
Secure Configuration Management with API Key Encryption
Prevents storing sensitive credentials in plain text
"""

from cryptography.fernet import Fernet
from pathlib import Path
from typing import Optional
import os


class SecureConfigManager:
    """
    Manages encryption and decryption of sensitive configuration data
    Uses Fernet symmetric encryption for API keys and secrets
    """
    
    def __init__(self, key_file: Path = Path('.keyfile')):
        """
        Initialize secure config manager
        
        Args:
            key_file: Path to encryption key file (default: .keyfile in project root)
        """
        self.key_file = key_file
        self.cipher = self._get_or_create_cipher()
    
    def _get_or_create_cipher(self) -> Fernet:
        """Get existing cipher or create new encryption key"""
        if not self.key_file.exists():
            # Generate new encryption key
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
            # Secure permissions (owner read/write only)
            self.key_file.chmod(0o600)
            print(f"✓ Created new encryption key: {self.key_file}")
        else:
            key = self.key_file.read_bytes()
        
        return Fernet(key)
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt sensitive string
        
        Args:
            plaintext: String to encrypt (e.g., API key)
            
        Returns:
            Encrypted string (base64 encoded)
        """
        if not plaintext:
            return ""
        
        encrypted_bytes = self.cipher.encrypt(plaintext.encode())
        return encrypted_bytes.decode()
    
    def decrypt(self, encrypted: str) -> str:
        """
        Decrypt encrypted string
        
        Args:
            encrypted: Encrypted string to decrypt
            
        Returns:
            Original plaintext string
        """
        if not encrypted:
            return ""
        
        try:
            decrypted_bytes = self.cipher.decrypt(encrypted.encode())
            return decrypted_bytes.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt: {e}. Key file may be corrupted.")
    
    def encrypt_api_keys(self, api_key: str, api_secret: str) -> tuple[str, str]:
        """
        Encrypt Binance API credentials
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            
        Returns:
            Tuple of (encrypted_key, encrypted_secret)
        """
        return (
            self.encrypt(api_key),
            self.encrypt(api_secret)
        )
    
    def decrypt_api_keys(self, encrypted_key: str, encrypted_secret: str) -> tuple[str, str]:
        """
        Decrypt Binance API credentials
        
        Args:
            encrypted_key: Encrypted API key
            encrypted_secret: Encrypted API secret
            
        Returns:
            Tuple of (api_key, api_secret)
        """
        return (
            self.decrypt(encrypted_key),
            self.decrypt(encrypted_secret)
        )
    
    @staticmethod
    def load_encrypted_env() -> dict[str, str]:
        """
        Load environment variables and decrypt if they're encrypted
        
        Returns:
            Dictionary with decrypted values
        """
        from dotenv import load_dotenv
        load_dotenv()
        
        config_manager = SecureConfigManager()
        
        # Get encrypted keys from environment
        encrypted_api_key = os.getenv('BINANCE_API_KEY_ENCRYPTED', '')
        encrypted_api_secret = os.getenv('BINANCE_API_SECRET_ENCRYPTED', '')
        
        # Fallback to plain text if encrypted not found (backward compatibility)
        if not encrypted_api_key:
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_API_SECRET', '')
            print("⚠️  WARNING: Using plain text API keys. Run encrypt_config.py to secure them.")
        else:
            api_key, api_secret = config_manager.decrypt_api_keys(
                encrypted_api_key, encrypted_api_secret
            )
        
        return {
            'BINANCE_API_KEY': api_key,
            'BINANCE_API_SECRET': api_secret,
            'BINANCE_TESTNET': os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        }


if __name__ == '__main__':
    # Example usage
    manager = SecureConfigManager()
    
    # Encrypt example API keys
    api_key = "your_api_key_here"
    api_secret = "your_api_secret_here"
    
    encrypted_key, encrypted_secret = manager.encrypt_api_keys(api_key, api_secret)
    
    print(f"Encrypted API Key: {encrypted_key[:50]}...")
    print(f"Encrypted API Secret: {encrypted_secret[:50]}...")
    
    # Decrypt
    decrypted_key, decrypted_secret = manager.decrypt_api_keys(encrypted_key, encrypted_secret)
    print(f"✓ Decryption successful: {decrypted_key == api_key}")
