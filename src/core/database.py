"""
Database Layer - v11.0 PostgreSQL Integration

Production-grade database layer with:
- SQLAlchemy ORM models
- Decision and event persistence
- Connection pooling
- SQLite fallback for development

v11.0 - Production-Grade Platform Upgrade
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database configuration  
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///data/cryptoboss.db')
IS_POSTGRES = DATABASE_URL.startswith('postgresql')

# SQLAlchemy (optional for production)
try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON
    from sqlalchemy.orm import sessionmaker, Session, declarative_base
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_AVAILABLE = True
    
    # SQLAlchemy base
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = None


# ============================================================================
# ORM Models
# ============================================================================

if SQLALCHEMY_AVAILABLE and Base is not None:
    
    class DecisionRecord(Base):
        """Trade decision record for audit trail."""
        __tablename__ = 'decisions'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        decision_id = Column(String(36), unique=True, nullable=False, index=True)
        intent_id = Column(String(36), nullable=True, index=True)
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        
        # Trade details
        symbol = Column(String(20), nullable=False, index=True)
        direction = Column(String(10), nullable=False)
        strategy_id = Column(String(50), nullable=True, index=True)
        
        # Status
        status = Column(String(20), nullable=False, index=True)  # approved/rejected
        rejection_stage = Column(String(50), nullable=True)
        rejection_reason = Column(Text, nullable=True)
        
        # Confidence and context
        confidence_score = Column(Float, default=0.0)
        market_regime = Column(String(30), nullable=True)
        directional_bias = Column(String(30), nullable=True)
        
        # Execution (if approved)
        position_size = Column(Float, nullable=True)
        entry_price = Column(Float, nullable=True)
        stop_loss = Column(Float, nullable=True)
        take_profit = Column(Float, nullable=True)
        
        # Execution result
        executed = Column(Boolean, default=False)
        fill_price = Column(Float, nullable=True)
        fill_size = Column(Float, nullable=True)
        slippage_bps = Column(Float, nullable=True)
        order_id = Column(String(100), nullable=True)
        
        # Pipeline timing
        total_pipeline_ms = Column(Float, default=0.0)
        
        # Full data (JSON)
        stage_results = Column(JSON, nullable=True)
        decision_metadata = Column(JSON, nullable=True)
        
        def __repr__(self):
            return f"<DecisionRecord {self.decision_id[:8]}... {self.status}>"
    
    
    class EventRecord(Base):
        """System event for replayability."""
        __tablename__ = 'events'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        event_id = Column(String(36), unique=True, nullable=False, index=True)
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        
        # Event type and source
        event_type = Column(String(50), nullable=False, index=True)
        source = Column(String(50), nullable=False)
        
        # Event data
        data = Column(JSON, nullable=True)
        
        # Related entities
        decision_id = Column(String(36), nullable=True, index=True)
        symbol = Column(String(20), nullable=True, index=True)
        
        def __repr__(self):
            return f"<EventRecord {self.event_type} @ {self.timestamp}>"
    
    
    class SystemState(Base):
        """System state for crash recovery."""
        __tablename__ = 'system_state'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        key = Column(String(100), unique=True, nullable=False, index=True)
        value = Column(JSON, nullable=True)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        def __repr__(self):
            return f"<SystemState {self.key}>"

else:
    # Placeholders when SQLAlchemy not available
    DecisionRecord = None
    EventRecord = None
    SystemState = None


# ============================================================================
# Database Manager
# ============================================================================

class DatabaseManager:
    """
    Manages database connections and provides ORM session access.
    
    Features:
    - Automatic table creation
    - Connection pooling (PostgreSQL)
    - SQLite fallback
    - Session context manager
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        if not SQLALCHEMY_AVAILABLE:
            logger.warning("SQLAlchemy not available - database disabled")
            self.engine = None
            self.SessionLocal = None
            self._initialized = True
            return
        
        self.db_url = DATABASE_URL
        self.is_postgres = IS_POSTGRES
        
        # Create engine
        try:
            if self.is_postgres:
                # PostgreSQL with connection pooling
                self.engine = create_engine(
                    self.db_url,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,
                    echo=False
                )
                logger.info(f"PostgreSQL database connected: {self.db_url[:30]}...")
            else:
                # SQLite for development
                # Ensure directory exists
                db_path = self.db_url.replace('sqlite:///', '')
                if db_path and '/' in db_path:
                    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
                
                self.engine = create_engine(
                    self.db_url,
                    connect_args={'check_same_thread': False},
                    poolclass=StaticPool,
                    echo=False
                )
                logger.info(f"SQLite database: {db_path}")
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create tables
            self._create_tables()
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.engine = None
            self.SessionLocal = None
        
        self._initialized = True
    
    def _create_tables(self):
        """Create all tables if they don't exist."""
        if Base is not None and self.engine is not None:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified")
    
    @contextmanager
    def session(self):
        """Context manager for database sessions."""
        if self.SessionLocal is None:
            yield None
            return
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def is_available(self) -> bool:
        """Check if database is available."""
        return self.engine is not None


# ============================================================================
# Repository Classes
# ============================================================================

class DecisionRepository:
    """Repository for TradeDecision persistence."""
    
    def __init__(self, db: DatabaseManager = None):
        self.db = db or get_database()
    
    def save(self, decision: "TradeDecision") -> bool:
        """Save a TradeDecision to the database."""
        if not self.db.is_available() or DecisionRecord is None:
            return False
        
        try:
            with self.db.session() as session:
                if session is None:
                    return False
                
                record = DecisionRecord(
                    decision_id=decision.decision_id,
                    intent_id=decision.intent_id,
                    timestamp=decision.timestamp,
                    symbol=decision.symbol,
                    direction=decision.direction,
                    strategy_id=decision.strategy_contributors[0] if decision.strategy_contributors else None,
                    status=decision.status.value if hasattr(decision.status, 'value') else str(decision.status),
                    rejection_stage=decision.rejection_stage.value if decision.rejection_stage else None,
                    rejection_reason=decision.rejection_reason,
                    confidence_score=decision.confidence_score,
                    market_regime=decision.market_regime,
                    directional_bias=decision.directional_bias,
                    position_size=decision.execution_params.position_size if decision.execution_params else None,
                    entry_price=decision.execution_params.entry_price if decision.execution_params else None,
                    stop_loss=decision.execution_params.stop_loss if decision.execution_params else None,
                    take_profit=decision.execution_params.take_profit if decision.execution_params else None,
                    executed=decision.executed,
                    fill_price=decision.fill_price,
                    fill_size=decision.fill_size,
                    slippage_bps=decision.slippage_bps,
                    order_id=decision.order_id,
                    total_pipeline_ms=decision.total_pipeline_ms,
                    stage_results=[sr.to_dict() for sr in decision.stage_results],
                )
                
                session.add(record)
                
            logger.debug(f"Decision {decision.decision_id[:8]} saved to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save decision: {e}")
            return False
    
    def get(self, decision_id: str) -> Optional[Dict]:
        """Get a decision by ID."""
        if not self.db.is_available() or DecisionRecord is None:
            return None
        
        try:
            with self.db.session() as session:
                if session is None:
                    return None
                
                record = session.query(DecisionRecord).filter_by(
                    decision_id=decision_id
                ).first()
                
                if record:
                    return self._record_to_dict(record)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get decision: {e}")
            return None
    
    def get_recent(
        self,
        limit: int = 100,
        symbol: str = None,
        status: str = None
    ) -> List[Dict]:
        """Get recent decisions with optional filters."""
        if not self.db.is_available() or DecisionRecord is None:
            return []
        
        try:
            with self.db.session() as session:
                if session is None:
                    return []
                
                query = session.query(DecisionRecord)
                
                if symbol:
                    query = query.filter_by(symbol=symbol)
                if status:
                    query = query.filter_by(status=status)
                
                records = query.order_by(
                    DecisionRecord.timestamp.desc()
                ).limit(limit).all()
                
                return [self._record_to_dict(r) for r in records]
                
        except Exception as e:
            logger.error(f"Failed to get recent decisions: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get decision statistics."""
        if not self.db.is_available() or DecisionRecord is None:
            return {'available': False}
        
        try:
            with self.db.session() as session:
                if session is None:
                    return {'available': False}
                
                from sqlalchemy import func
                
                total = session.query(func.count(DecisionRecord.id)).scalar() or 0
                approved = session.query(func.count(DecisionRecord.id)).filter_by(
                    status='approved'
                ).scalar() or 0
                executed = session.query(func.count(DecisionRecord.id)).filter(
                    DecisionRecord.executed == True
                ).scalar() or 0
                
                avg_pipeline = session.query(
                    func.avg(DecisionRecord.total_pipeline_ms)
                ).scalar() or 0
                
                return {
                    'available': True,
                    'total_decisions': total,
                    'total_approved': approved,
                    'total_executed': executed,
                    'approval_rate': approved / total if total > 0 else 0,
                    'execution_rate': executed / approved if approved > 0 else 0,
                    'avg_pipeline_ms': float(avg_pipeline),
                }
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'available': False, 'error': str(e)}
    
    def _record_to_dict(self, record: "DecisionRecord") -> Dict:
        """Convert record to dictionary."""
        return {
            'decision_id': record.decision_id,
            'intent_id': record.intent_id,
            'timestamp': record.timestamp.isoformat() if record.timestamp else None,
            'symbol': record.symbol,
            'direction': record.direction,
            'strategy_id': record.strategy_id,
            'status': record.status,
            'rejection_stage': record.rejection_stage,
            'rejection_reason': record.rejection_reason,
            'confidence_score': record.confidence_score,
            'executed': record.executed,
            'fill_price': record.fill_price,
            'slippage_bps': record.slippage_bps,
        }


class EventRepository:
    """Repository for system events."""
    
    def __init__(self, db: DatabaseManager = None):
        self.db = db or get_database()
    
    def log_event(
        self,
        event_type: str,
        source: str,
        data: Dict = None,
        decision_id: str = None,
        symbol: str = None
    ) -> bool:
        """Log a system event."""
        if not self.db.is_available() or EventRecord is None:
            return False
        
        try:
            import uuid
            
            with self.db.session() as session:
                if session is None:
                    return False
                
                record = EventRecord(
                    event_id=str(uuid.uuid4()),
                    event_type=event_type,
                    source=source,
                    data=data,
                    decision_id=decision_id,
                    symbol=symbol,
                )
                
                session.add(record)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
            return False
    
    def get_events(
        self,
        event_type: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get recent events."""
        if not self.db.is_available() or EventRecord is None:
            return []
        
        try:
            with self.db.session() as session:
                if session is None:
                    return []
                
                query = session.query(EventRecord)
                
                if event_type:
                    query = query.filter_by(event_type=event_type)
                
                records = query.order_by(
                    EventRecord.timestamp.desc()
                ).limit(limit).all()
                
                return [
                    {
                        'event_id': r.event_id,
                        'timestamp': r.timestamp.isoformat() if r.timestamp else None,
                        'event_type': r.event_type,
                        'source': r.source,
                        'data': r.data,
                    }
                    for r in records
                ]
                
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []


class StateRepository:
    """Repository for crash-safe state recovery."""
    
    def __init__(self, db: DatabaseManager = None):
        self.db = db or get_database()
    
    def save_state(self, key: str, value: Any) -> bool:
        """Save state value."""
        if not self.db.is_available() or SystemState is None:
            return False
        
        try:
            with self.db.session() as session:
                if session is None:
                    return False
                
                # Upsert
                existing = session.query(SystemState).filter_by(key=key).first()
                
                if existing:
                    existing.value = value
                    existing.updated_at = datetime.utcnow()
                else:
                    record = SystemState(key=key, value=value)
                    session.add(record)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        if not self.db.is_available() or SystemState is None:
            return default
        
        try:
            with self.db.session() as session:
                if session is None:
                    return default
                
                record = session.query(SystemState).filter_by(key=key).first()
                
                return record.value if record else default
                
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return default
    
    def delete_state(self, key: str) -> bool:
        """Delete state value."""
        if not self.db.is_available() or SystemState is None:
            return False
        
        try:
            with self.db.session() as session:
                if session is None:
                    return False
                
                session.query(SystemState).filter_by(key=key).delete()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete state: {e}")
            return False


# ============================================================================
# Singleton Access
# ============================================================================

_database: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get the global DatabaseManager instance."""
    global _database
    if _database is None:
        _database = DatabaseManager()
    return _database


def get_decision_repository() -> DecisionRepository:
    """Get DecisionRepository instance."""
    return DecisionRepository(get_database())


def get_event_repository() -> EventRepository:
    """Get EventRepository instance."""
    return EventRepository(get_database())


def get_state_repository() -> StateRepository:
    """Get StateRepository instance."""
    return StateRepository(get_database())
