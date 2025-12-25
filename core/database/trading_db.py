"""
Database Integration - Trade & Performance Persistence
SQLite-based storage for trades, signals, and metrics

Benefits:
- Persistent trade history
- Historical analysis
- Fast queries
- No external dependencies (SQLite)
- Easy migration to PostgreSQL later
"""

import sqlite3
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TradingDatabase:
    """
    SQLite database for trading data
    
    Tables:
    - trades: All executed trades
    - signals: Trading signals generated
    - performance: Daily performance snapshots
    - events: System events and alerts
    """
    
    def __init__(self, db_path: str = "trading_data.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Create database and tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return dict-like rows
        
        cursor = self.conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                size REAL NOT NULL,
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                mae REAL DEFAULT 0,
                mfe REAL DEFAULT 0,
                duration_seconds INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0,
                strategy TEXT,
                notes TEXT,
                status TEXT DEFAULT 'OPEN'
            )
        ''')
        
        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                confidence REAL NOT NULL,
                reasons TEXT,
                executed BOOLEAN DEFAULT 0
            )
        ''')
        
        # Performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                equity REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0
            )
        ''')
        
        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT DEFAULT 'INFO',
                message TEXT NOT NULL,
                details TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(date)')
        
        self.conn.commit()
        logger.info(f"✓ Database initialized: {self.db_path}")
    
    def insert_trade(self, trade: Dict) -> int:
        """Insert a new trade"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, side, entry_price, exit_price, size,
                              pnl, pnl_pct, mae, mfe, duration_seconds, confidence, strategy, notes, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.get('timestamp', datetime.now().isoformat()),
            trade.get('symbol', 'BTC'),
            trade.get('side', 'LONG'),
            trade['entry_price'],
            trade.get('exit_price'),
            trade['size'],
            trade.get('pnl', 0),
            trade.get('pnl_pct', 0),
            trade.get('mae', 0),
            trade.get('mfe', 0),
            trade.get('duration_seconds', 0),
            trade.get('confidence', 0),
            trade.get('strategy'),
            trade.get('notes'),
            trade.get('status', 'CLOSED')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_signal(self, signal: Dict) -> int:
        """Insert a trading signal"""
        cursor = self.conn.cursor()
        
        reasons_json = json.dumps(signal.get('reasons', []))
        
        cursor.execute('''
            INSERT INTO signals (timestamp, symbol, action, price, confidence, reasons, executed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.get('timestamp', datetime.now().isoformat()),
            signal.get('symbol', 'BTC'),
            signal['action'],
            signal['price'],
            signal['confidence'],
            reasons_json,
            signal.get('executed', False)
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def update_performance(self, date: str, metrics: Dict):
        """Update or insert daily performance"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO performance (date, equity, daily_pnl, total_trades, 
                                               win_rate, sharpe_ratio, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            date,
            metrics['equity'],
            metrics.get('daily_pnl', 0),
            metrics.get('total_trades', 0),
            metrics.get('win_rate', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('max_drawdown', 0)
        ))
        
        self.conn.commit()
    
    def log_event(self, event_type: str, message: str, severity: str = 'INFO', details: Dict = None):
        """Log a system event"""
        cursor = self.conn.cursor()
        
        details_json = json.dumps(details) if details else None
        
        cursor.execute('''
            INSERT INTO events (timestamp, event_type, severity, message, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            event_type,
            severity,
            message,
            details_json
        ))
        
        self.conn.commit()
    
    def get_trades(self, limit: int = 100, symbol: Optional[str] = None) -> List[Dict]:
        """Get recent trades"""
        cursor = self.conn.cursor()
        
        if symbol:
            cursor.execute('''
                SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?
            ''', (symbol, limit))
        else:
            cursor.execute('''
                SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_history(self, days: int = 30) -> List[Dict]:
        """Get performance history"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT * FROM performance ORDER BY date DESC LIMIT ?
        ''', (days,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_trade_statistics(self) -> Dict:
        """Get aggregate trade statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losers,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM trades WHERE status = 'CLOSED'
        ''')
        
        row = cursor.fetchone()
        if row:
            stats = dict(row)
            stats['win_rate'] = (stats['winners'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
            return stats
        
        return {}
    
    def backup(self, backup_path: str):
        """Create database backup"""
        import shutil
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"✓ Database backed up to: {backup_path}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Singleton
_trading_db: Optional[TradingDatabase] = None


def get_trading_db(db_path: str = "trading_data.db") -> TradingDatabase:
    """Get singleton database instance"""
    global _trading_db
    if _trading_db is None:
        _trading_db = TradingDatabase(db_path)
    return _trading_db


if __name__ == '__main__':
    print("=" * 70)
    print("DATABASE INTEGRATION - TEST")
    print("=" * 70)
    
    # Initialize database
    db = TradingDatabase("test_trading.db")
    
    # Insert sample trades
    print("\n📊 Inserting sample trades...")
    for i in range(10):
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTC',
            'side': 'LONG' if i % 2 == 0 else 'SHORT',
            'entry_price': 50000 + i * 100,
            'exit_price': 50000 + i * 100 + (100 if i % 2 == 0 else -50),
            'size': 0.1,
            'pnl': 10 if i % 2 == 0 else -5,
            'pnl_pct': 0.02 if i % 2 == 0 else -0.01,
            'confidence': 0.75,
            'strategy': 'momentum',
            'status': 'CLOSED'
        }
        trade_id = db.insert_trade(trade)
        print(f"  Trade {trade_id}: {trade['side']} @ ${trade['entry_price']}")
    
    # Insert sample signals
    print("\n📡 Inserting sample signals...")
    for i in range(5):
        signal = {
            'symbol': 'BTC',
            'action': 'LONG',
            'price': 50000,
            'confidence': 0.8,
            'reasons': ['Strong momentum', 'HTF trend UP'],
            'executed': i < 3
        }
        db.insert_signal(signal)
    
    # Update performance
    print("\n📈 Updating performance...")
    metrics = {
        'equity': 10150.0,
        'daily_pnl': 150.0,
        'total_trades': 10,
        'win_rate': 0.60,
        'sharpe_ratio': 1.85,
        'max_drawdown': 0.05
    }
    db.update_performance(datetime.now().strftime('%Y-%m-%d'), metrics)
    
    # Log events
    print("\n📝 Logging events...")
    db.log_event('SYSTEM_START', 'Trading bot started', 'INFO')
    db.log_event('TRADE_EXECUTED', 'LONG trade executed', 'INFO', {'symbol': 'BTC', 'price': 50000})
    
    # Query data
    print("\n📊 Querying trades...")
    trades = db.get_trades(limit=5)
    print(f"Retrieved {len(trades)} trades")
    
    stats = db.get_trade_statistics()
    print(f"\n📈 Trade Statistics:")
    print(f"  Total trades: {stats.get('total_trades', 0)}")
    print(f"  Winners: {stats.get('winners', 0)}")
    print(f"  Total P&L: ${stats.get('total_pnl', 0):.2f}")
    print(f"  Win rate: {stats.get('win_rate', 0):.1f}%")
    
    # Backup
    print("\n💾 Creating backup...")
    db.backup("test_trading_backup.db")
    
    db.close()
    
    print("\n" + "=" * 70)
    print("✅ Database integration working!")
    print("=" * 70)
