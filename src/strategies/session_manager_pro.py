"""Professional session and kill-zone manager for intraday trading."""

from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start_utc: time
    end_utc: time
    weight: float = 1.0


class SessionManagerPro:
    """Tracks active market sessions and kill-zones in UTC."""

    ASIA = SessionWindow("Asia", time(0, 0), time(8, 30), weight=0.6)
    LONDON = SessionWindow("London", time(7, 0), time(16, 0), weight=1.0)
    NEW_YORK = SessionWindow("NY", time(13, 0), time(22, 0), weight=1.0)
    OVERLAP = SessionWindow("Overlap", time(13, 0), time(16, 0), weight=1.2)

    KILL_ZONES: List[Tuple[time, time, str]] = [
        (time(0, 0), time(3, 0), "Asia Open"),
        (time(7, 0), time(9, 0), "London Open"),
        (time(12, 0), time(14, 0), "NY Open"),
        (time(13, 0), time(16, 0), "London/NY Overlap"),
    ]

    def current_session(self, now_utc: Optional[time] = None) -> Optional[SessionWindow]:
        now = now_utc or datetime.utcnow().time()

        if self.OVERLAP.start_utc <= now <= self.OVERLAP.end_utc:
            return self.OVERLAP
        if self.LONDON.start_utc <= now <= self.LONDON.end_utc:
            return self.LONDON
        if self.NEW_YORK.start_utc <= now <= self.NEW_YORK.end_utc:
            return self.NEW_YORK
        if self.ASIA.start_utc <= now <= self.ASIA.end_utc:
            return self.ASIA
        return None

    def in_kill_zone(self, now_utc: Optional[time] = None) -> Tuple[bool, str]:
        now = now_utc or datetime.utcnow().time()
        for start, end, name in self.KILL_ZONES:
            if start <= now <= end:
                return True, name
        return False, ""

    def session_weight(self, now_utc: Optional[time] = None) -> float:
        session = self.current_session(now_utc=now_utc)
        return session.weight if session else 0.0
