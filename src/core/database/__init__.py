"""
CryptoBoss Database Package

CRYPTOBOSS 2.0: Combined database layer with:
- SQLite user/account/trade repository (NEW)
- Original SQLAlchemy ORM for decisions/events

Re-exports from original database.py for backwards compatibility.
"""

from .repository import SQLiteUserRepository, get_repository

# Import from parent database.py for backwards compatibility
from pathlib import Path
import sys
import importlib.util

# Load the original database.py module
parent_db_path = Path(__file__).parent.parent / "database.py"
if parent_db_path.exists():
    spec = importlib.util.spec_from_file_location("parent_database", parent_db_path)
    parent_db = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(parent_db)
        
        # Re-export for backwards compatibility
        DatabaseManager = getattr(parent_db, 'DatabaseManager', None)
        DecisionRepository = getattr(parent_db, 'DecisionRepository', None)
        EventRepository = getattr(parent_db, 'EventRepository', None)
        StateRepository = getattr(parent_db, 'StateRepository', None)
        get_database = getattr(parent_db, 'get_database', None)
        get_decision_repository = getattr(parent_db, 'get_decision_repository', None)
        get_event_repository = getattr(parent_db, 'get_event_repository', None)
        get_state_repository = getattr(parent_db, 'get_state_repository', None)
    except Exception as e:
        # Fallback - set to None
        DatabaseManager = None
        DecisionRepository = None
        EventRepository = None
        StateRepository = None
        get_database = None
        get_decision_repository = None
        get_event_repository = None
        get_state_repository = None
else:
    DatabaseManager = None
    DecisionRepository = None
    EventRepository = None
    StateRepository = None
    get_database = None
    get_decision_repository = None
    get_event_repository = None
    get_state_repository = None

__all__ = [
    # New SQLite repository
    'SQLiteUserRepository', 
    'get_repository',
    # Legacy SQLAlchemy (backwards compatibility)
    'DatabaseManager',
    'DecisionRepository',
    'EventRepository',
    'StateRepository',
    'get_database',
    'get_decision_repository',
    'get_event_repository',
    'get_state_repository',
]
