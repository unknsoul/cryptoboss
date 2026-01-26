"""
Secrets Management - Upgrade J

Secure handling of API keys and sensitive configuration:
- Environment variable loading with validation
- Support for AWS Secrets Manager / HashiCorp Vault
- Key rotation support
- Audit logging

Never store secrets in code or git!
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Secret:
    """Represents a secret value."""
    name: str
    value: str
    source: str  # 'env', 'vault', 'aws', 'file'
    loaded_at: datetime
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def __repr__(self):
        # Never print the actual value!
        return f"Secret(name='{self.name}', source='{self.source}', loaded_at='{self.loaded_at}')"


class SecretsBackend(ABC):
    """Abstract secrets backend."""
    
    @abstractmethod
    def get_secret(self, name: str) -> Optional[str]:
        pass
    
    @abstractmethod
    def list_secrets(self) -> list:
        pass


class EnvSecretsBackend(SecretsBackend):
    """Load secrets from environment variables."""
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        
        # Load .env file if dotenv is available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
    
    def get_secret(self, name: str) -> Optional[str]:
        env_name = f"{self.prefix}{name}" if self.prefix else name
        return os.getenv(env_name)
    
    def list_secrets(self) -> list:
        if not self.prefix:
            return []
        return [k for k in os.environ.keys() if k.startswith(self.prefix)]


class VaultSecretsBackend(SecretsBackend):
    """HashiCorp Vault secrets backend."""
    
    def __init__(self, url: str, token: str, mount_point: str = "secret"):
        self.url = url
        self.token = token
        self.mount_point = mount_point
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import hvac
                self._client = hvac.Client(url=self.url, token=self.token)
            except ImportError:
                raise ImportError("hvac library required for Vault backend: pip install hvac")
        return self._client
    
    def get_secret(self, name: str) -> Optional[str]:
        try:
            client = self._get_client()
            secret = client.secrets.kv.read_secret_version(
                path=name,
                mount_point=self.mount_point
            )
            return secret['data']['data'].get('value')
        except Exception as e:
            logger.error(f"Failed to get secret from Vault: {e}")
            return None
    
    def list_secrets(self) -> list:
        try:
            client = self._get_client()
            secrets = client.secrets.kv.list_secrets(
                path="",
                mount_point=self.mount_point
            )
            return secrets['data']['keys']
        except Exception as e:
            logger.error(f"Failed to list secrets from Vault: {e}")
            return []


class AWSSecretsBackend(SecretsBackend):
    """AWS Secrets Manager backend."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('secretsmanager', region_name=self.region)
            except ImportError:
                raise ImportError("boto3 library required for AWS backend: pip install boto3")
        return self._client
    
    def get_secret(self, name: str) -> Optional[str]:
        try:
            client = self._get_client()
            response = client.get_secret_value(SecretId=name)
            return response.get('SecretString')
        except Exception as e:
            logger.error(f"Failed to get secret from AWS: {e}")
            return None
    
    def list_secrets(self) -> list:
        try:
            client = self._get_client()
            response = client.list_secrets()
            return [s['Name'] for s in response.get('SecretList', [])]
        except Exception as e:
            logger.error(f"Failed to list secrets from AWS: {e}")
            return []


class SecretsManager:
    """
    Central secrets management.
    
    Usage:
        secrets = SecretsManager()
        
        # Get exchange API keys
        api_key = secrets.get("BINANCE_API_KEY")
        api_secret = secrets.get("BINANCE_API_SECRET")
        
        # Validate required secrets
        secrets.require(["BINANCE_API_KEY", "BINANCE_API_SECRET", "TELEGRAM_BOT_TOKEN"])
        
        # Get all exchange credentials
        creds = secrets.get_exchange_credentials("binance")
    """
    
    # Required secrets for the trading bot
    REQUIRED_SECRETS = [
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
    ]
    
    OPTIONAL_SECRETS = [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_USER_ID",
        "EMAIL_SMTP_HOST",
        "EMAIL_SMTP_PORT",
        "EMAIL_USERNAME",
        "EMAIL_PASSWORD",
        "DISCORD_WEBHOOK_URL",
        "SLACK_WEBHOOK_URL",
        "GLASSNODE_API_KEY",
        "CRYPTOQUANT_API_KEY",
    ]
    
    def __init__(self, backend: SecretsBackend = None):
        self.backend = backend or EnvSecretsBackend()
        self._cache: Dict[str, Secret] = {}
        self._audit_log: list = []
    
    def get(self, name: str, default: str = None) -> Optional[str]:
        """Get a secret value."""
        # Check cache
        if name in self._cache:
            secret = self._cache[name]
            if not secret.is_expired():
                self._audit("get", name, "cache_hit")
                return secret.value
        
        # Load from backend
        value = self.backend.get_secret(name)
        
        if value:
            self._cache[name] = Secret(
                name=name,
                value=value,
                source=type(self.backend).__name__,
                loaded_at=datetime.now()
            )
            self._audit("get", name, "loaded")
            return value
        
        self._audit("get", name, "not_found")
        return default
    
    def require(self, names: list) -> Dict[str, str]:
        """
        Require multiple secrets, raise if any missing.
        
        Returns dict of name -> value
        """
        result = {}
        missing = []
        
        for name in names:
            value = self.get(name)
            if value is None:
                missing.append(name)
            else:
                result[name] = value
        
        if missing:
            raise ValueError(f"Missing required secrets: {missing}")
        
        return result
    
    def validate_required(self) -> tuple[bool, list]:
        """
        Validate all required secrets are present.
        
        Returns (valid, missing_list)
        """
        missing = []
        for name in self.REQUIRED_SECRETS:
            if not self.get(name):
                missing.append(name)
        
        return len(missing) == 0, missing
    
    def get_exchange_credentials(self, exchange: str = "binance") -> Dict[str, str]:
        """Get credentials for an exchange."""
        prefix = exchange.upper()
        
        return {
            "api_key": self.get(f"{prefix}_API_KEY"),
            "api_secret": self.get(f"{prefix}_API_SECRET"),
            "passphrase": self.get(f"{prefix}_PASSPHRASE"),  # For some exchanges
        }
    
    def get_telegram_config(self) -> Dict:
        """Get Telegram bot configuration."""
        token = self.get("TELEGRAM_BOT_TOKEN")
        user_id = self.get("TELEGRAM_USER_ID")
        
        return {
            "token": token,
            "user_ids": [int(user_id)] if user_id else [],
            "enabled": bool(token and user_id)
        }
    
    def get_email_config(self) -> Dict:
        """Get email notification configuration."""
        return {
            "smtp_host": self.get("EMAIL_SMTP_HOST", "smtp.gmail.com"),
            "smtp_port": int(self.get("EMAIL_SMTP_PORT", "587")),
            "username": self.get("EMAIL_USERNAME"),
            "password": self.get("EMAIL_PASSWORD"),
            "from_addr": self.get("EMAIL_FROM_ADDR"),
            "to_addrs": self.get("EMAIL_TO_ADDRS", "").split(","),
            "enabled": bool(self.get("EMAIL_USERNAME") and self.get("EMAIL_PASSWORD"))
        }
    
    def _audit(self, action: str, secret_name: str, result: str):
        """Log secret access for audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "secret": secret_name,
            "result": result
        }
        self._audit_log.append(entry)
        
        # Keep only last 1000 entries
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]
    
    def get_audit_log(self, limit: int = 100) -> list:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]
    
    def clear_cache(self):
        """Clear the secrets cache (force reload)."""
        self._cache.clear()
        self._audit("clear_cache", "*", "done")
    
    def rotate_secret(self, name: str, new_value: str) -> bool:
        """
        Rotate a secret (requires backend that supports writes).
        
        For now, just clears from cache so it will be reloaded.
        """
        if name in self._cache:
            del self._cache[name]
        
        self._audit("rotate", name, "cache_cleared")
        logger.warning(f"Secret {name} rotated - update source and restart")
        return True


# Singleton
_secrets: Optional[SecretsManager] = None

def get_secrets() -> SecretsManager:
    global _secrets
    if _secrets is None:
        _secrets = SecretsManager()
    return _secrets


def require_secret(name: str) -> str:
    """Convenience function to get a required secret."""
    value = get_secrets().get(name)
    if value is None:
        raise ValueError(f"Required secret not found: {name}")
    return value
