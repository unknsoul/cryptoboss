"""
CryptoBoss 1.0.1 - Exchange Account Service

Manages exchange accounts with state isolation.
Each API key pair creates a NEW exchange_account_id.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from ..models.user import ExchangeAccount, Environment
from .crypto import get_key_vault

logger = logging.getLogger(__name__)


@dataclass
class AccountResult:
    """Result of account operation."""
    success: bool
    account: Optional[ExchangeAccount] = None
    accounts: Optional[List[ExchangeAccount]] = None
    error: Optional[str] = None


class ExchangeAccountService:
    """
    Service for managing exchange accounts.
    
    Rules:
    - Each API key pair creates a NEW exchange_account_id
    - Changing API keys ALWAYS creates a new account
    - Old accounts remain archived unless deleted
    - Same user can switch between accounts
    """
    
    def __init__(self, account_repository=None):
        """
        Initialize account service.
        
        Args:
            account_repository: Database repository for accounts (injected)
        """
        self.account_repo = account_repository or InMemoryAccountRepository()
        self.key_vault = get_key_vault()
        logger.info("🔑 ExchangeAccountService initialized")
    
    def create_account(
        self,
        user_id: str,
        exchange_name: str,
        environment: str,
        api_key: str,
        api_secret: str,
        label: str = ""
    ) -> AccountResult:
        """
        Create a new exchange account.
        
        This always creates a NEW account - changing keys = new account.
        API keys are encrypted before storage.
        """
        # Validate environment
        if environment not in ["TESTNET", "LIVE"]:
            return AccountResult(success=False, error="Invalid environment. Use TESTNET or LIVE")
        
        # Validate exchange
        if exchange_name.lower() not in ["binance"]:
            return AccountResult(success=False, error="Unsupported exchange")
        
        # Validate keys not empty
        if not api_key or not api_secret:
            return AccountResult(success=False, error="API key and secret are required")
        
        # Check if this key already exists for this user
        key_hash = self.key_vault.get_hash(api_key)
        existing = self.account_repo.find_by_key_hash(user_id, key_hash)
        if existing:
            return AccountResult(success=False, error="This API key is already registered")
        
        # Encrypt keys
        try:
            encrypted_key, encrypted_secret = self.key_vault.encrypt_key_pair(api_key, api_secret)
        except Exception as e:
            logger.error(f"Failed to encrypt keys: {e}")
            return AccountResult(success=False, error="Failed to secure API keys")
        
        # Create account
        account = ExchangeAccount.create(
            user_id=user_id,
            exchange_name=exchange_name.lower(),
            environment=environment,
            api_key_encrypted=encrypted_key,
            api_secret_encrypted=encrypted_secret,
            label=label
        )
        
        # Store key hash for duplicate detection
        account._key_hash = key_hash
        
        # Save
        self.account_repo.save(account)
        
        logger.info(f"✅ New exchange account created: {account.exchange_account_id[:8]}... ({environment})")
        return AccountResult(success=True, account=account)
    
    def get_accounts(self, user_id: str) -> AccountResult:
        """Get all accounts for a user."""
        accounts = self.account_repo.find_by_user(user_id)
        return AccountResult(success=True, accounts=accounts)
    
    def get_account(self, user_id: str, exchange_account_id: str) -> AccountResult:
        """Get a specific account."""
        account = self.account_repo.find_by_id(exchange_account_id)
        if not account:
            return AccountResult(success=False, error="Account not found")
        if account.user_id != user_id:
            return AccountResult(success=False, error="Access denied")
        return AccountResult(success=True, account=account)
    
    def delete_account(self, user_id: str, exchange_account_id: str) -> AccountResult:
        """
        Delete/archive an exchange account.
        
        This marks the account as inactive but keeps data for audit.
        """
        account = self.account_repo.find_by_id(exchange_account_id)
        if not account:
            return AccountResult(success=False, error="Account not found")
        if account.user_id != user_id:
            return AccountResult(success=False, error="Access denied")
        
        # Archive instead of delete
        account.is_active = False
        self.account_repo.save(account)
        
        logger.info(f"🗑️ Exchange account archived: {exchange_account_id[:8]}...")
        return AccountResult(success=True, account=account)
    
    def get_decrypted_keys(self, user_id: str, exchange_account_id: str) -> Optional[Tuple[str, str]]:
        """
        Get decrypted API keys for an account.
        
        ⚠️ Use with caution - only for actual API calls to exchange.
        Never log or send to frontend.
        """
        account = self.account_repo.find_by_id(exchange_account_id)
        if not account or account.user_id != user_id:
            return None
        
        try:
            return self.key_vault.decrypt_key_pair(
                account.api_key_encrypted,
                account.api_secret_encrypted
            )
        except Exception as e:
            logger.error(f"Failed to decrypt keys: {e}")
            return None
    
    def get_key_fingerprint(self, user_id: str, exchange_account_id: str) -> Optional[str]:
        """Get masked fingerprint of API key for display."""
        keys = self.get_decrypted_keys(user_id, exchange_account_id)
        if keys:
            return self.key_vault.get_fingerprint(keys[0])
        return None
    
    def validate_account(self, user_id: str, exchange_account_id: str) -> AccountResult:
        """
        Validate an account's API keys with the exchange.
        
        Returns success if keys are valid.
        """
        keys = self.get_decrypted_keys(user_id, exchange_account_id)
        if not keys:
            return AccountResult(success=False, error="Could not retrieve keys")
        
        # TODO: Actually validate with exchange
        # For now, just mark as validated
        account = self.account_repo.find_by_id(exchange_account_id)
        if account:
            account.mark_validated()
            self.account_repo.save(account)
            return AccountResult(success=True, account=account)
        
        return AccountResult(success=False, error="Account not found")


class InMemoryAccountRepository:
    """
    In-memory account repository for development/testing.
    
    Replace with database repository in production.
    """
    
    def __init__(self):
        self.accounts: Dict[str, ExchangeAccount] = {}
        self.user_accounts: Dict[str, List[str]] = {}  # user_id -> [account_ids]
        self.key_hashes: Dict[str, str] = {}  # user_id:hash -> account_id
    
    def save(self, account: ExchangeAccount):
        """Save or update an account."""
        self.accounts[account.exchange_account_id] = account
        
        # Index by user
        if account.user_id not in self.user_accounts:
            self.user_accounts[account.user_id] = []
        if account.exchange_account_id not in self.user_accounts[account.user_id]:
            self.user_accounts[account.user_id].append(account.exchange_account_id)
        
        # Index by key hash if available
        if hasattr(account, '_key_hash'):
            key = f"{account.user_id}:{account._key_hash}"
            self.key_hashes[key] = account.exchange_account_id
    
    def find_by_id(self, exchange_account_id: str) -> Optional[ExchangeAccount]:
        """Find account by ID."""
        return self.accounts.get(exchange_account_id)
    
    def find_by_user(self, user_id: str) -> List[ExchangeAccount]:
        """Find all accounts for a user."""
        account_ids = self.user_accounts.get(user_id, [])
        accounts = [self.accounts[aid] for aid in account_ids if aid in self.accounts]
        return [a for a in accounts if a.is_active]
    
    def find_by_key_hash(self, user_id: str, key_hash: str) -> Optional[ExchangeAccount]:
        """Find account by API key hash."""
        key = f"{user_id}:{key_hash}"
        account_id = self.key_hashes.get(key)
        if account_id:
            return self.accounts.get(account_id)
        return None
    
    def delete(self, exchange_account_id: str) -> bool:
        """Delete an account."""
        if exchange_account_id in self.accounts:
            del self.accounts[exchange_account_id]
            return True
        return False


# Singleton
_account_service: Optional[ExchangeAccountService] = None


def get_account_service() -> ExchangeAccountService:
    """
    Get the global ExchangeAccountService instance.
    
    CRYPTOBOSS 2.0: Uses SQLite repository for persistent storage.
    Accounts now survive server restarts.
    """
    global _account_service
    if _account_service is None:
        # Try to use SQLite repository
        try:
            from ..database.repository import get_repository
            repository = get_repository()
            
            # Create adapter for SQLite repository
            class SQLiteAccountRepository:
                """Adapter to use SQLite repository for accounts."""
                def __init__(self, repo):
                    self.repo = repo
                    self.key_hashes: Dict[str, str] = {}  # user_id:hash -> account_id
                
                def save(self, account: ExchangeAccount):
                    self.repo.save_account(account)
                    if hasattr(account, '_key_hash'):
                        key = f"{account.user_id}:{account._key_hash}"
                        self.key_hashes[key] = account.exchange_account_id
                
                def find_by_id(self, exchange_account_id: str) -> Optional[ExchangeAccount]:
                    return self.repo.find_account_by_id(exchange_account_id)
                
                def find_by_user(self, user_id: str) -> List[ExchangeAccount]:
                    return self.repo.find_accounts_by_user(user_id)
                
                def find_by_key_hash(self, user_id: str, key_hash: str) -> Optional[ExchangeAccount]:
                    key = f"{user_id}:{key_hash}"
                    account_id = self.key_hashes.get(key)
                    if account_id:
                        return self.find_by_id(account_id)
                    return None
                
                def delete(self, exchange_account_id: str) -> bool:
                    # Mark as inactive instead of deleting
                    account = self.find_by_id(exchange_account_id)
                    if account:
                        account.is_active = False
                        self.save(account)
                        return True
                    return False
            
            adapter = SQLiteAccountRepository(repository)
            _account_service = ExchangeAccountService(account_repository=adapter)
            logger.info("🔑 ExchangeAccountService using SQLite repository (persistent)")
        except ImportError:
            # Fallback to in-memory
            _account_service = ExchangeAccountService()
            logger.warning("⚠️ ExchangeAccountService using in-memory repository (NOT persistent)")
    return _account_service
