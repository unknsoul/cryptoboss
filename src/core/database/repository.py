"""
CryptoBoss 2.0 - SQLite Database Repository

Persistent storage for users, exchange accounts, and trades.
All data is scoped by user_id and exchange_account_id.

CRITICAL RULE: Every query MUST filter by ownership keys.
"""

import sqlite3
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from pathlib import Path

from ..models.user import User, ExchangeAccount

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path("data/cryptoboss.db")


@contextmanager
def get_connection():
    """Get database connection with context manager."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class SQLiteUserRepository:
    """
    SQLite-based user repository.
    
    Persists users across server restarts.
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._ensure_db_exists()
        self._init_tables()
        logger.info(f"📁 SQLite UserRepository initialized: {self.db_path}")
    
    def _ensure_db_exists(self):
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_tables(self):
        """Initialize database tables."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exchange_accounts (
                    exchange_account_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    exchange_name TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    api_key_encrypted BLOB,
                    api_secret_encrypted BLOB,
                    label TEXT,
                    created_at TEXT NOT NULL,
                    last_validated_at TEXT,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    exchange_account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    pnl REAL DEFAULT 0,
                    fees REAL DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (exchange_account_id) REFERENCES exchange_accounts(exchange_account_id)
                )
            """)
            
            # Index for fast trade lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_ownership 
                ON trades(user_id, exchange_account_id)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_sessions (
                    user_id TEXT PRIMARY KEY,
                    active_exchange_account_id TEXT,
                    last_activity TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            conn.commit()
            logger.info("✅ Database tables initialized")
    
    # === User Operations ===
    
    def save(self, user: User) -> bool:
        """Save or update a user."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO users 
                (user_id, email, password_hash, created_at, is_active)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user.user_id,
                user.email.lower(),
                user.password_hash,
                user.created_at.isoformat(),
                1 if user.is_active else 0
            ))
            conn.commit()
            logger.info(f"✅ User saved: {user.email}")
            return True
    
    def find_by_id(self, user_id: str) -> Optional[User]:
        """Find user by ID."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", 
                (user_id,)
            ).fetchone()
            
            if row:
                return self._row_to_user(row)
            return None
    
    def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", 
                (email.lower().strip(),)
            ).fetchone()
            
            if row:
                return self._row_to_user(row)
            return None
    
    def delete(self, user_id: str) -> bool:
        """Delete a user."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            return True
    
    def _row_to_user(self, row: sqlite3.Row) -> User:
        """Convert database row to User object."""
        return User(
            user_id=row['user_id'],
            email=row['email'],
            password_hash=row['password_hash'],
            created_at=datetime.fromisoformat(row['created_at']),
            is_active=bool(row['is_active'])
        )
    
    # === Exchange Account Operations ===
    
    def save_account(self, account: ExchangeAccount) -> bool:
        """Save or update an exchange account."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO exchange_accounts 
                (exchange_account_id, user_id, exchange_name, environment,
                 api_key_encrypted, api_secret_encrypted, label, created_at,
                 last_validated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                account.exchange_account_id,
                account.user_id,
                account.exchange_name,
                account.environment,
                account.api_key_encrypted,
                account.api_secret_encrypted,
                account.label,
                account.created_at.isoformat(),
                account.last_validated_at.isoformat() if account.last_validated_at else None,
                1 if account.is_active else 0
            ))
            conn.commit()
            logger.info(f"✅ Exchange account saved: {account.label}")
            return True
    
    def find_accounts_by_user(self, user_id: str) -> List[ExchangeAccount]:
        """Find all exchange accounts for a user."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM exchange_accounts WHERE user_id = ? AND is_active = 1",
                (user_id,)
            ).fetchall()
            
            return [self._row_to_account(row) for row in rows]
    
    def find_account_by_id(self, exchange_account_id: str) -> Optional[ExchangeAccount]:
        """Find exchange account by ID."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM exchange_accounts WHERE exchange_account_id = ?",
                (exchange_account_id,)
            ).fetchone()
            
            if row:
                return self._row_to_account(row)
            return None
    
    def _row_to_account(self, row: sqlite3.Row) -> ExchangeAccount:
        """Convert database row to ExchangeAccount object."""
        return ExchangeAccount(
            exchange_account_id=row['exchange_account_id'],
            user_id=row['user_id'],
            exchange_name=row['exchange_name'],
            environment=row['environment'],
            api_key_encrypted=row['api_key_encrypted'],
            api_secret_encrypted=row['api_secret_encrypted'],
            label=row['label'] or "",
            created_at=datetime.fromisoformat(row['created_at']),
            last_validated_at=datetime.fromisoformat(row['last_validated_at']) if row['last_validated_at'] else None,
            is_active=bool(row['is_active'])
        )
    
    # === Trade Operations (Scoped by ownership) ===
    
    def save_trade(self, trade: Dict[str, Any]) -> bool:
        """
        Save a trade.
        
        CRITICAL: Trade must have user_id AND exchange_account_id.
        """
        if not trade.get('user_id') or not trade.get('exchange_account_id'):
            raise ValueError("Trade must have user_id and exchange_account_id")
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades 
                (trade_id, user_id, exchange_account_id, symbol, side, 
                 price, quantity, timestamp, pnl, fees)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get('trade_id', str(datetime.now().timestamp())),
                trade['user_id'],
                trade['exchange_account_id'],
                trade.get('symbol', 'BTC/USDT'),
                trade.get('side', 'buy'),
                trade.get('price', 0),
                trade.get('quantity', 0),
                trade.get('timestamp', datetime.now().isoformat()),
                trade.get('pnl', 0),
                trade.get('fees', 0)
            ))
            conn.commit()
            return True
    
    def get_trades(self, user_id: str, exchange_account_id: str, limit: int = 50) -> List[Dict]:
        """
        Get trades for a specific user AND exchange account.
        
        CRITICAL: ALWAYS filters by BOTH ownership keys.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM trades 
                WHERE user_id = ? AND exchange_account_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, exchange_account_id, limit)).fetchall()
            
            return [dict(row) for row in rows]
    
    def delete_trades_for_account(self, user_id: str, exchange_account_id: str) -> int:
        """
        Delete all trades for a specific account.
        
        CRITICAL: ALWAYS filters by BOTH ownership keys.
        Returns number of deleted trades.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                DELETE FROM trades 
                WHERE user_id = ? AND exchange_account_id = ?
            """, (user_id, exchange_account_id))
            conn.commit()
            deleted = cursor.rowcount
            logger.info(f"🗑️ Deleted {deleted} trades for account {exchange_account_id}")
            return deleted
    
    # === Active Session Management ===
    
    def set_active_account(self, user_id: str, exchange_account_id: str):
        """Set the active exchange account for a user."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO active_sessions 
                (user_id, active_exchange_account_id, last_activity)
                VALUES (?, ?, ?)
            """, (user_id, exchange_account_id, datetime.now().isoformat()))
            conn.commit()
    
    def get_active_account_id(self, user_id: str) -> Optional[str]:
        """Get the active exchange account ID for a user."""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT active_exchange_account_id FROM active_sessions WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            
            if row:
                return row[0]
            return None


# Singleton instance
_repository: Optional[SQLiteUserRepository] = None


def get_repository() -> SQLiteUserRepository:
    """Get the global repository instance."""
    global _repository
    if _repository is None:
        _repository = SQLiteUserRepository()
    return _repository
