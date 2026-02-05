"""
CryptoBoss 1.0.1 - User & Exchange Account Models

Core identity models for user authentication and exchange account management.
All state must be scoped by exchange_account_id.
"""

import uuid
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class Environment(str, Enum):
    """Exchange environment types."""
    TESTNET = "TESTNET"
    LIVE = "LIVE"


class ExchangeName(str, Enum):
    """Supported exchanges."""
    BINANCE = "binance"


@dataclass
class User:
    """
    User identity model.
    
    Primary identity for the system. All data is scoped to user_id.
    """
    user_id: str
    email: str
    password_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    @classmethod
    def create(cls, email: str, password: str) -> "User":
        """Create a new user with hashed password."""
        user_id = str(uuid.uuid4())
        password_hash = cls._hash_password(password)
        return cls(
            user_id=user_id,
            email=email.lower().strip(),
            password_hash=password_hash
        )
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash password using bcrypt-style salted hash."""
        import secrets
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        ).hex()
        return f"{salt}${hashed}"
    
    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt, stored_hash = self.password_hash.split('$')
            computed_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt.encode(),
                100000
            ).hex()
            return computed_hash == stored_hash
        except (ValueError, AttributeError):
            return False
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert to dictionary for API responses."""
        data = {
            "user_id": self.user_id,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active
        }
        if include_sensitive:
            data["password_hash"] = self.password_hash
        return data


@dataclass
class ExchangeAccount:
    """
    Exchange account model.
    
    Each API key pair creates a NEW exchange_account_id.
    All state (trades, risk, replay) is scoped to this ID.
    """
    exchange_account_id: str
    user_id: str
    exchange_name: str
    environment: str  # TESTNET or LIVE
    api_key_encrypted: bytes
    api_secret_encrypted: bytes
    created_at: datetime = field(default_factory=datetime.now)
    last_validated_at: Optional[datetime] = None
    label: str = ""  # User-friendly name like "Main Trading Account"
    is_active: bool = True
    
    @classmethod
    def create(
        cls,
        user_id: str,
        exchange_name: str,
        environment: str,
        api_key_encrypted: bytes,
        api_secret_encrypted: bytes,
        label: str = ""
    ) -> "ExchangeAccount":
        """Create a new exchange account."""
        return cls(
            exchange_account_id=str(uuid.uuid4()),
            user_id=user_id,
            exchange_name=exchange_name,
            environment=environment,
            api_key_encrypted=api_key_encrypted,
            api_secret_encrypted=api_secret_encrypted,
            label=label or f"{exchange_name} {environment}"
        )
    
    def get_fingerprint(self, decrypted_key: str) -> str:
        """Get masked fingerprint for frontend display (e.g., 'abc...xyz')."""
        if len(decrypted_key) < 8:
            return "***"
        return f"{decrypted_key[:4]}...{decrypted_key[-4:]}"
    
    def mark_validated(self):
        """Mark account as recently validated."""
        self.last_validated_at = datetime.now()
    
    def to_dict(self, include_keys: bool = False) -> dict:
        """Convert to dictionary for API responses."""
        data = {
            "exchange_account_id": self.exchange_account_id,
            "user_id": self.user_id,
            "exchange_name": self.exchange_name,
            "environment": self.environment,
            "label": self.label,
            "created_at": self.created_at.isoformat(),
            "last_validated_at": self.last_validated_at.isoformat() if self.last_validated_at else None,
            "is_active": self.is_active
        }
        # Never include actual keys in API response
        return data


@dataclass
class UserSession:
    """
    User session for authentication.
    
    JWT-based session management.
    """
    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    active_exchange_account_id: Optional[str] = None
    
    @classmethod
    def create(cls, user_id: str, expires_in_hours: int = 24) -> "UserSession":
        """Create a new session."""
        from datetime import timedelta
        return cls(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            expires_at=datetime.now() + timedelta(hours=expires_in_hours)
        )
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def set_active_account(self, exchange_account_id: str):
        """Set the active exchange account for this session."""
        self.active_exchange_account_id = exchange_account_id
