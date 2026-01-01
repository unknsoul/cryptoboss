"""
Database Module
Provides SQLite database management for the trading system.
"""

import sqlite3
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging
import threading

logger = logging.getLogger(__name__)


class SQLiteManager:
    """
    SQLite database manager for trading data persistence.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, db_path: str = "trading_data.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """Initialize database tables if they don't exist."""
        cursor = self._conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL DEFAULT 0,
                pnl_percent REAL DEFAULT 0,
                status TEXT DEFAULT 'open',
                strategy TEXT,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                stop_loss REAL,
                take_profit REAL,
                fees REAL DEFAULT 0,
                metadata TEXT
            )
        """)
        
        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                strategy TEXT,
                executed INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        
        # Equity curve table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                drawdown REAL DEFAULT 0
            )
        """)
        
        self._conn.commit()
        logger.debug(f"Database initialized at {self.db_path}")
    
    def save_trade(self, trade: Dict) -> int:
        """
        Save a trade to the database.
        
        Args:
            trade: Trade dictionary with keys:
                - symbol, side, entry_price, quantity
                - Optional: exit_price, pnl, status, strategy, etc.
        
        Returns:
            Trade ID
        """
        cursor = self._conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (
                symbol, side, entry_price, exit_price, quantity,
                pnl, pnl_percent, status, strategy, entry_time, exit_time,
                stop_loss, take_profit, fees, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.get('symbol', 'BTCUSDT'),
            trade.get('side', 'BUY'),
            trade.get('entry_price', 0),
            trade.get('exit_price'),
            trade.get('quantity', 0),
            trade.get('pnl', 0),
            trade.get('pnl_percent', 0),
            trade.get('status', 'open'),
            trade.get('strategy'),
            trade.get('entry_time', datetime.now().isoformat()),
            trade.get('exit_time'),
            trade.get('stop_loss'),
            trade.get('take_profit'),
            trade.get('fees', 0),
            json.dumps(trade.get('metadata', {}))
        ))
        
        self._conn.commit()
        return cursor.lastrowid
    
    def update_trade(self, trade_id: int, updates: Dict):
        """
        Update an existing trade.
        
        Args:
            trade_id: ID of the trade to update
            updates: Dictionary of fields to update
        """
        allowed_fields = [
            'exit_price', 'pnl', 'pnl_percent', 'status', 
            'exit_time', 'fees', 'metadata'
        ]
        
        set_clauses = []
        values = []
        
        for field, value in updates.items():
            if field in allowed_fields:
                set_clauses.append(f"{field} = ?")
                if field == 'metadata':
                    values.append(json.dumps(value))
                else:
                    values.append(value)
        
        if set_clauses:
            values.append(trade_id)
            cursor = self._conn.cursor()
            cursor.execute(
                f"UPDATE trades SET {', '.join(set_clauses)} WHERE id = ?",
                values
            )
            self._conn.commit()
    
    def get_trade(self, trade_id: int) -> Optional[Dict]:
        """Get a trade by ID."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """
        Get recent trades.
        
        Args:
            limit: Maximum number of trades to return
        
        Returns:
            List of trade dictionaries
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_trades(self) -> List[Dict]:
        """Get all trades."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY id DESC")
        return [dict(row) for row in cursor.fetchall()]
    
    def get_open_trades(self) -> List[Dict]:
        """Get all open trades."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY id DESC"
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def save_signal(self, signal: Dict) -> int:
        """Save a trading signal."""
        cursor = self._conn.cursor()
        
        cursor.execute("""
            INSERT INTO signals (symbol, action, confidence, timestamp, strategy, executed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.get('symbol', 'BTCUSDT'),
            signal.get('action', 'HOLD'),
            signal.get('confidence', 0),
            signal.get('timestamp', datetime.now().isoformat()),
            signal.get('strategy'),
            signal.get('executed', False),
            json.dumps(signal.get('metadata', {}))
        ))
        
        self._conn.commit()
        return cursor.lastrowid
    
    def get_recent_signals(self, limit: int = 100) -> List[Dict]:
        """Get recent signals."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def save_equity(self, equity: float, drawdown: float = 0):
        """Save equity curve data point."""
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO equity_curve (timestamp, equity, drawdown) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), equity, drawdown)
        )
        self._conn.commit()
    
    def get_equity_curve(self, limit: int = 1000) -> List[Dict]:
        """Get equity curve data."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM equity_curve ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics from trades."""
        trades = self.get_all_trades()
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'profit_factor': 0
            }
        
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        winners = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losers = [t for t in closed_trades if t.get('pnl', 0) < 0]
        
        total_wins = sum(t.get('pnl', 0) for t in winners)
        total_losses = abs(sum(t.get('pnl', 0) for t in losers))
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': (len(winners) / len(closed_trades) * 100) if closed_trades else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(closed_trades) if closed_trades else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf')
        }
    
    def close(self):
        """Close the database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


__all__ = ['SQLiteManager']
