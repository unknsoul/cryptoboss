"""
Deterministic Replay Engine - Live Readiness Component

Ensures live and replay decisions are identical:
- Records all market events and decisions during live trading
- Replays using identical logic
- Compares outputs for determinism verification
- Flags mismatches for investigation

Critical for debugging and confidence in live trading.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import gzip

logger = logging.getLogger(__name__)


@dataclass
class ReplayEvent:
    """Recorded market event for replay."""
    timestamp: str
    event_type: str  # 'price', 'orderbook', 'fill', 'context', 'bias'
    symbol: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ReplayEvent':
        return cls(**data)


@dataclass
class ReplayDecision:
    """Recorded decision for comparison."""
    timestamp: str
    decision_type: str
    symbol: str
    result: str  # The actual outcome
    context: Dict[str, Any]  # Full context for debugging
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ReplayDecision':
        return cls(**data)


@dataclass
class ReplayMismatch:
    """Records a mismatch between live and replay."""
    timestamp: str
    decision_type: str
    live_result: str
    replay_result: str
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReplaySession:
    """Complete replay session data."""
    session_id: str
    start_time: str
    end_time: Optional[str]
    symbol: str
    events: List[ReplayEvent]
    decisions: List[ReplayDecision]
    is_recording: bool
    event_count: int
    decision_count: int


class DeterministicReplayEngine:
    """
    Records and replays trading decisions for determinism verification.
    
    Recording Mode:
        - Captures all market events (price, orderbook, fills)
        - Captures all decisions (context, bias, permission, trades)
        - Stores in compressed JSON format
    
    Replay Mode:
        - Feeds recorded events to decision logic
        - Compares outputs to recorded decisions
        - Reports mismatches for investigation
    
    Usage:
        # Recording
        replay = DeterministicReplayEngine()
        replay.start_recording("BTC/USDT")
        
        # During trading loop
        replay.record_event('price', symbol, {'price': 40000})
        replay.record_decision('context', symbol, 'RANGING', context_data)
        
        # Stop and save
        session = replay.stop_recording()
        
        # Later - verify
        mismatches = replay.verify_session(session.session_id, decision_func)
    """
    
    def __init__(
        self,
        data_dir: str = "data/replay",
        max_events_per_session: int = 100000
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_events = max_events_per_session
        
        # Current recording state
        self._is_recording = False
        self._current_session_id: Optional[str] = None
        self._current_symbol: Optional[str] = None
        self._events: List[ReplayEvent] = []
        self._decisions: List[ReplayDecision] = []
        self._recording_start: Optional[datetime] = None
        
        logger.info(f"DeterministicReplayEngine initialized: {self.data_dir}")
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
    @property
    def current_session(self) -> Optional[str]:
        """Get current session ID."""
        return self._current_session_id
    
    def start_recording(self, symbol: str) -> str:
        """
        Start recording a new session.
        
        Returns: Session ID
        """
        if self._is_recording:
            logger.warning("Already recording, stopping current session")
            self.stop_recording()
        
        self._current_session_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._current_symbol = symbol
        self._events = []
        self._decisions = []
        self._recording_start = datetime.now()
        self._is_recording = True
        
        logger.info(f"Started recording session: {self._current_session_id}")
        return self._current_session_id
    
    def record_event(self, event_type: str, symbol: str, data: Dict[str, Any]):
        """
        Record a market event.
        
        Args:
            event_type: 'price', 'orderbook', 'fill', etc.
            symbol: Trading symbol
            data: Event data
        """
        if not self._is_recording:
            return
        
        if len(self._events) >= self.max_events:
            logger.warning(f"Max events reached ({self.max_events}), stopping recording")
            self.stop_recording()
            return
        
        event = ReplayEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            symbol=symbol,
            data=data
        )
        self._events.append(event)
    
    def record_decision(
        self,
        decision_type: str,
        symbol: str,
        result: str,
        context: Dict[str, Any]
    ):
        """
        Record a decision made by the system.
        
        Args:
            decision_type: 'context', 'bias', 'permission', 'trade'
            symbol: Trading symbol
            result: The decision result (e.g., 'RANGING', 'LONG_ONLY', 'APPROVED')
            context: Full context data for debugging
        """
        if not self._is_recording:
            return
        
        decision = ReplayDecision(
            timestamp=datetime.now().isoformat(),
            decision_type=decision_type,
            symbol=symbol,
            result=result,
            context=context
        )
        self._decisions.append(decision)
    
    def stop_recording(self) -> Optional[ReplaySession]:
        """
        Stop recording and save session.
        
        Returns: Session summary
        """
        if not self._is_recording:
            return None
        
        self._is_recording = False
        
        session = ReplaySession(
            session_id=self._current_session_id,
            start_time=self._recording_start.isoformat(),
            end_time=datetime.now().isoformat(),
            symbol=self._current_symbol,
            events=self._events,
            decisions=self._decisions,
            is_recording=False,
            event_count=len(self._events),
            decision_count=len(self._decisions)
        )
        
        # Save to disk
        self._save_session(session)
        
        logger.info(
            f"Stopped recording session: {self._current_session_id}, "
            f"events={len(self._events)}, decisions={len(self._decisions)}"
        )
        
        # Clear state
        self._events = []
        self._decisions = []
        self._current_session_id = None
        
        return session
    
    def load_session(self, session_id: str) -> Optional[ReplaySession]:
        """Load a recorded session from disk."""
        session_file = self.data_dir / f"{session_id}.json.gz"
        
        if not session_file.exists():
            logger.error(f"Session not found: {session_id}")
            return None
        
        try:
            with gzip.open(session_file, 'rt') as f:
                data = json.load(f)
            
            return ReplaySession(
                session_id=data['session_id'],
                start_time=data['start_time'],
                end_time=data['end_time'],
                symbol=data['symbol'],
                events=[ReplayEvent.from_dict(e) for e in data['events']],
                decisions=[ReplayDecision.from_dict(d) for d in data['decisions']],
                is_recording=False,
                event_count=len(data['events']),
                decision_count=len(data['decisions'])
            )
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def list_sessions(self) -> List[Dict]:
        """List all recorded sessions."""
        sessions = []
        
        for file in self.data_dir.glob("*.json.gz"):
            try:
                with gzip.open(file, 'rt') as f:
                    data = json.load(f)
                
                sessions.append({
                    'session_id': data['session_id'],
                    'symbol': data['symbol'],
                    'start_time': data['start_time'],
                    'end_time': data['end_time'],
                    'event_count': len(data['events']),
                    'decision_count': len(data['decisions'])
                })
            except Exception as e:
                logger.error(f"Failed to read {file}: {e}")
        
        return sorted(sessions, key=lambda s: s['start_time'], reverse=True)
    
    def verify_session(
        self,
        session_id: str,
        decision_func: callable
    ) -> List[ReplayMismatch]:
        """
        Verify a recorded session against decision function.
        
        Args:
            session_id: Session to verify
            decision_func: Function that takes event and returns decision result
            
        Returns:
            List of mismatches found
        """
        session = self.load_session(session_id)
        if not session:
            return []
        
        mismatches = []
        decision_index = 0
        
        logger.info(f"Verifying session {session_id}...")
        
        for event in session.events:
            # Call decision function with event
            try:
                replay_result = decision_func(event)
            except Exception as e:
                replay_result = f"ERROR: {e}"
            
            # Find matching recorded decision
            if decision_index < len(session.decisions):
                recorded = session.decisions[decision_index]
                
                # Check if this event should produce a decision
                if event.event_type in ['price', 'context', 'bias']:
                    if replay_result != recorded.result:
                        mismatch = ReplayMismatch(
                            timestamp=event.timestamp,
                            decision_type=recorded.decision_type,
                            live_result=recorded.result,
                            replay_result=str(replay_result),
                            context=recorded.context
                        )
                        mismatches.append(mismatch)
                        
                        logger.warning(
                            f"MISMATCH at {event.timestamp}: "
                            f"live={recorded.result}, replay={replay_result}"
                        )
                    
                    decision_index += 1
        
        if mismatches:
            logger.error(f"Verification FAILED: {len(mismatches)} mismatches")
        else:
            logger.info(f"Verification PASSED: {len(session.decisions)} decisions matched")
        
        return mismatches
    
    def _save_session(self, session: ReplaySession):
        """Save session to compressed JSON file."""
        session_file = self.data_dir / f"{session.session_id}.json.gz"
        
        data = {
            'session_id': session.session_id,
            'start_time': session.start_time,
            'end_time': session.end_time,
            'symbol': session.symbol,
            'events': [e.to_dict() for e in session.events],
            'decisions': [d.to_dict() for d in session.decisions]
        }
        
        try:
            with gzip.open(session_file, 'wt') as f:
                json.dump(data, f)
            
            logger.info(f"Session saved: {session_file}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get statistics for a session."""
        session = self.load_session(session_id)
        if not session:
            return None
        
        # Count decision types
        decision_counts = {}
        for d in session.decisions:
            decision_counts[d.decision_type] = decision_counts.get(d.decision_type, 0) + 1
        
        # Count event types
        event_counts = {}
        for e in session.events:
            event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1
        
        return {
            'session_id': session_id,
            'symbol': session.symbol,
            'duration_seconds': (
                datetime.fromisoformat(session.end_time) -
                datetime.fromisoformat(session.start_time)
            ).total_seconds() if session.end_time else 0,
            'total_events': len(session.events),
            'total_decisions': len(session.decisions),
            'event_types': event_counts,
            'decision_types': decision_counts
        }


# Singleton instance
_replay_engine: Optional[DeterministicReplayEngine] = None


def get_replay_engine() -> DeterministicReplayEngine:
    """Get global DeterministicReplayEngine instance."""
    global _replay_engine
    if _replay_engine is None:
        _replay_engine = DeterministicReplayEngine()
    return _replay_engine
