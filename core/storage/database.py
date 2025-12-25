"""
Robust SQLite Database Manager for Trading Bot
Replaces fragile JSON file persistence with transactional database storage.
"""
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from core.config import settings

logger = logging.getLogger(__name__)

class SQLiteManager:
    """Manages all database operations using SQLite"""
    
    def __init__(self, db_path: Path = settings.DB_PATH):
        self.db_path = db_path
        self._init_db()
        
    def _get_connection(self):
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
        return conn

    def _init_db(self):
        """Initialize database schema"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Trades Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    pnl REAL,
                    status TEXT,
                    strategy TEXT,
                    confidence REAL,
                    meta_data TEXT
                )
                """)
                
                # Signals Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT,
                    action TEXT,
                    price REAL,
                    confidence REAL,
                    strategy TEXT,
                    meta_data TEXT
                )
                """)
                
                # Bot State Table (Single Row)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """)
                
                conn.commit()
                
                # === MIGRATION: Add meta_data column if missing ===
                try:
                    cursor.execute("SELECT meta_data FROM trades LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info("DB: Migrating - adding meta_data column to trades")
                    cursor.execute("ALTER TABLE trades ADD COLUMN meta_data TEXT")
                    conn.commit()
                
                logger.info(f"✓ Database initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def save_trade(self, trade: Dict):
        """Save a new trade or update existing"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # check if is update (if trade has an ID)
                trade_json = json.dumps(trade) 
                
                cursor.execute("""
                INSERT INTO trades (
                    timestamp, symbol, side, entry_price, exit_price, 
                    quantity, pnl, status, strategy, confidence, meta_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.get('timestamp', datetime.now().isoformat()),
                    trade.get('symbol', 'BTCUSDT'),
                    trade.get('side', 'UNKNOWN'),
                    trade.get('entry_price', 0.0),
                    trade.get('exit_price', 0.0),
                    trade.get('quantity', 0.0),
                    trade.get('pnl', 0.0),
                    trade.get('status', 'OPEN'),
                    trade.get('strategy', 'manual'),
                    trade.get('confidence', 0.0),
                    json.dumps(trade)  # Store full object as meta_data
                ))
                conn.commit()
                # Update ID in trade dict
                trade['db_id'] = cursor.lastrowid
                
        except Exception as e:
            logger.error(f"DB: Failed to save trade: {e}")

    def load_trades(self, limit: int = 100) -> List[Dict]:
        """Load recent trades"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT meta_data FROM trades 
                ORDER BY id DESC LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                return [json.loads(row['meta_data']) for row in rows][::-1] # Return oldest to newest
                
        except Exception as e:
            logger.error(f"DB: Failed to load trades: {e}")
            return []

    def save_state(self, state: Dict):
        """Save bot state (equity, streaks, etc)"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Store as single JSON blob for key 'main_state'
                state_json = json.dumps(state)
                cursor.execute("""
                INSERT OR REPLACE INTO bot_state (key, value) 
                VALUES (?, ?)
                """, ('main_state', state_json))
                conn.commit()
        except Exception as e:
            logger.error(f"DB: Failed to save state: {e}")

    def load_state(self) -> Dict:
        """Load bot state"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM bot_state WHERE key = 'main_state'")
                row = cursor.fetchone()
                if row:
                    return json.loads(row['value'])
                return {}
        except Exception as e:
            logger.error(f"DB: Failed to load state: {e}")
            return {}

    def clear_all_data(self):
        """Clear all data (Reset Dashboard)"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM trades")
                cursor.execute("DELETE FROM signals")
                cursor.execute("DELETE FROM bot_state")
                conn.commit()
                logger.info("DB: All data cleared")
        except Exception as e:
            logger.error(f"DB: Failed to clear data: {e}")
