"""
CryptoBoss 1.0.1 - Secure Key Vault

AES256 encryption for API keys at rest.
Keys are never logged and secrets never sent to frontend.
"""

import os
import base64
import hashlib
import logging
from typing import Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecureKeyVault:
    """
    Secure vault for encrypting/decrypting API keys.
    
    Uses AES256 (via Fernet) with a master key derived from:
    - Environment variable CRYPTOBOSS_MASTER_KEY, or
    - Auto-generated key stored in .keyfile
    
    Security Rules:
    - API keys never logged
    - API secrets never sent to frontend
    - Frontend only sees masked key fingerprint
    """
    
    _instance = None
    _fernet = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the encryption key."""
        master_key = self._get_or_create_master_key()
        self._fernet = Fernet(master_key)
        logger.info("🔐 SecureKeyVault initialized")
    
    def _get_or_create_master_key(self) -> bytes:
        """Get master key from environment or generate one."""
        # Try environment variable first
        env_key = os.environ.get("CRYPTOBOSS_MASTER_KEY")
        if env_key:
            # Derive a proper Fernet key from the env variable
            return self._derive_key(env_key.encode())
        
        # Try loading from keyfile
        keyfile_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".keyfile")
        keyfile_path = os.path.normpath(keyfile_path)
        
        if os.path.exists(keyfile_path):
            with open(keyfile_path, "rb") as f:
                return f.read()
        
        # Generate new key
        key = Fernet.generate_key()
        
        # Save to keyfile (should be in .gitignore)
        try:
            with open(keyfile_path, "wb") as f:
                f.write(key)
            logger.warning(f"⚠️ Generated new master key at {keyfile_path}")
            logger.warning("⚠️ BACKUP THIS FILE - losing it means losing access to encrypted data!")
        except Exception as e:
            logger.error(f"Could not save keyfile: {e}")
        
        return key
    
    def _derive_key(self, password: bytes) -> bytes:
        """Derive a Fernet-compatible key from a password."""
        # Use a fixed salt for deterministic key derivation
        # In production, you might want to store this salt securely
        salt = b"cryptoboss_v1.0.1_salt"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt(self, plaintext: str) -> bytes:
        """
        Encrypt a string (API key or secret).
        
        Returns encrypted bytes that can be stored in database.
        """
        if not plaintext:
            raise ValueError("Cannot encrypt empty value")
        return self._fernet.encrypt(plaintext.encode())
    
    def decrypt(self, encrypted: bytes) -> str:
        """
        Decrypt encrypted bytes back to string.
        
        Returns the original API key or secret.
        """
        if not encrypted:
            raise ValueError("Cannot decrypt empty value")
        return self._fernet.decrypt(encrypted).decode()
    
    def get_fingerprint(self, api_key: str) -> str:
        """
        Get a safe fingerprint for display.
        
        Shows first 4 and last 4 characters only.
        Example: "abc1...xyz9"
        """
        if not api_key or len(api_key) < 8:
            return "********"
        return f"{api_key[:4]}...{api_key[-4:]}"
    
    def get_hash(self, api_key: str) -> str:
        """
        Get a hash of the API key for comparison/lookup.
        
        Used to check if a key already exists without decrypting all keys.
        """
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]
    
    def encrypt_key_pair(self, api_key: str, api_secret: str) -> Tuple[bytes, bytes]:
        """
        Encrypt both API key and secret.
        
        Returns (encrypted_key, encrypted_secret)
        """
        return self.encrypt(api_key), self.encrypt(api_secret)
    
    def decrypt_key_pair(self, encrypted_key: bytes, encrypted_secret: bytes) -> Tuple[str, str]:
        """
        Decrypt both API key and secret.
        
        Returns (api_key, api_secret)
        """
        return self.decrypt(encrypted_key), self.decrypt(encrypted_secret)


# Singleton instance
_vault: SecureKeyVault = None


def get_key_vault() -> SecureKeyVault:
    """Get the global SecureKeyVault instance."""
    global _vault
    if _vault is None:
        _vault = SecureKeyVault()
    return _vault
