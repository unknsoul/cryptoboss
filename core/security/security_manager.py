"""
Security & Compliance - Enterprise Features #201, #205, #210, #215
API Key Management, Audit Trail, IP Whitelist, Rate Limiting.
"""

import logging
import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict
import base64
import secrets
import threading

logger = logging.getLogger(__name__)


class APIKeyManager:
    """
    Feature #201: API Key Encryption
    
    Securely stores and manages API keys with encryption.
    """
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize API key manager.
        
        Args:
            encryption_key: Key for encryption (generated if not provided)
        """
        self.encryption_key = encryption_key or os.getenv('ENCRYPTION_KEY') or self._generate_key()
        self.stored_keys: Dict[str, Dict] = {}
        
        logger.info("API Key Manager initialized with encryption")
    
    def _generate_key(self) -> str:
        """Generate a secure encryption key."""
        return secrets.token_hex(32)
    
    def _encrypt(self, plaintext: str) -> str:
        """Encrypt a string using XOR with key (simplified for demo)."""
        key_bytes = self.encryption_key.encode()
        plain_bytes = plaintext.encode()
        
        encrypted = bytes([
            p ^ key_bytes[i % len(key_bytes)]
            for i, p in enumerate(plain_bytes)
        ])
        
        return base64.b64encode(encrypted).decode()
    
    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt a string."""
        key_bytes = self.encryption_key.encode()
        cipher_bytes = base64.b64decode(ciphertext.encode())
        
        decrypted = bytes([
            c ^ key_bytes[i % len(key_bytes)]
            for i, c in enumerate(cipher_bytes)
        ])
        
        return decrypted.decode()
    
    def store_key(self, name: str, api_key: str, api_secret: str) -> Dict:
        """Store an encrypted API key pair."""
        key_id = secrets.token_hex(8)
        
        self.stored_keys[key_id] = {
            'name': name,
            'api_key': self._encrypt(api_key),
            'api_secret': self._encrypt(api_secret),
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'is_active': True
        }
        
        logger.info(f"API Key '{name}' stored with ID: {key_id[:8]}...")
        
        return {'key_id': key_id, 'name': name}
    
    def get_key(self, key_id: str) -> Optional[Dict]:
        """Retrieve and decrypt an API key pair."""
        if key_id not in self.stored_keys:
            return None
        
        stored = self.stored_keys[key_id]
        if not stored['is_active']:
            return None
        
        stored['last_used'] = datetime.now().isoformat()
        
        return {
            'name': stored['name'],
            'api_key': self._decrypt(stored['api_key']),
            'api_secret': self._decrypt(stored['api_secret'])
        }
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self.stored_keys:
            self.stored_keys[key_id]['is_active'] = False
            logger.warning(f"API Key {key_id[:8]}... revoked")
            return True
        return False
    
    def list_keys(self) -> List[Dict]:
        """List all stored keys (metadata only)."""
        return [
            {
                'key_id': kid,
                'name': v['name'],
                'created_at': v['created_at'],
                'is_active': v['is_active'],
                'last_used': v['last_used']
            }
            for kid, v in self.stored_keys.items()
        ]


class AuditTrail:
    """
    Feature #205: Audit Trail Logger
    
    Logs all security-relevant actions for compliance.
    """
    
    def __init__(self, log_file: Optional[str] = None, max_entries: int = 10000):
        """
        Initialize audit trail.
        
        Args:
            log_file: Optional file path for persistence
            max_entries: Maximum entries to keep in memory
        """
        self.log_file = log_file
        self.max_entries = max_entries
        self.entries: List[Dict] = []
        self._lock = threading.Lock()
        
        logger.info("Audit Trail initialized")
    
    def log(
        self,
        action: str,
        user: str = 'system',
        resource: str = '',
        details: Optional[Dict] = None,
        severity: str = 'INFO'
    ):
        """
        Log an audit event.
        
        Args:
            action: Action performed
            user: User/system performing action
            resource: Resource affected
            details: Additional details
            severity: INFO, WARNING, ERROR, CRITICAL
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'user': user,
            'resource': resource,
            'details': details or {},
            'severity': severity,
            'id': secrets.token_hex(8)
        }
        
        with self._lock:
            self.entries.append(entry)
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]
        
        # Write to file if configured
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")
        
        if severity in ['ERROR', 'CRITICAL']:
            logger.warning(f"AUDIT [{severity}]: {action} by {user}")
    
    def get_entries(
        self,
        action: Optional[str] = None,
        user: Optional[str] = None,
        severity: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Query audit entries with filters."""
        filtered = self.entries
        
        if action:
            filtered = [e for e in filtered if action in e['action']]
        if user:
            filtered = [e for e in filtered if e['user'] == user]
        if severity:
            filtered = [e for e in filtered if e['severity'] == severity]
        if since:
            since_str = since.isoformat()
            filtered = [e for e in filtered if e['timestamp'] >= since_str]
        
        return filtered[-limit:]
    
    def get_summary(self, hours: int = 24) -> Dict:
        """Get summary of recent audit activity."""
        since = datetime.now() - timedelta(hours=hours)
        recent = self.get_entries(since=since, limit=10000)
        
        by_action = defaultdict(int)
        by_severity = defaultdict(int)
        by_user = defaultdict(int)
        
        for e in recent:
            by_action[e['action']] += 1
            by_severity[e['severity']] += 1
            by_user[e['user']] += 1
        
        return {
            'total_events': len(recent),
            'period_hours': hours,
            'by_action': dict(by_action),
            'by_severity': dict(by_severity),
            'by_user': dict(by_user)
        }


class IPWhitelist:
    """
    Feature #210: IP Whitelist Manager
    
    Restricts API access to whitelisted IPs only.
    """
    
    def __init__(self):
        """Initialize IP whitelist."""
        self.whitelist: Set[str] = set()
        self.blocked_attempts: List[Dict] = []
        self.is_enabled = False
        
        logger.info("IP Whitelist Manager initialized")
    
    def add_ip(self, ip: str, description: str = ''):
        """Add an IP to whitelist."""
        self.whitelist.add(ip)
        logger.info(f"IP whitelisted: {ip} ({description})")
    
    def remove_ip(self, ip: str):
        """Remove an IP from whitelist."""
        self.whitelist.discard(ip)
        logger.info(f"IP removed from whitelist: {ip}")
    
    def enable(self):
        """Enable IP whitelist checking."""
        self.is_enabled = True
        logger.warning("IP Whitelist ENABLED - Only whitelisted IPs can access")
    
    def disable(self):
        """Disable IP whitelist checking."""
        self.is_enabled = False
        logger.info("IP Whitelist disabled")
    
    def check(self, ip: str) -> bool:
        """
        Check if IP is allowed.
        
        Args:
            ip: IP address to check
            
        Returns:
            True if allowed, False if blocked
        """
        if not self.is_enabled:
            return True
        
        if ip in self.whitelist:
            return True
        
        # Log blocked attempt
        self.blocked_attempts.append({
            'timestamp': datetime.now().isoformat(),
            'ip': ip
        })
        self.blocked_attempts = self.blocked_attempts[-1000:]
        
        logger.warning(f"IP blocked: {ip}")
        return False
    
    def get_blocked_attempts(self, since_hours: int = 24) -> List[Dict]:
        """Get recent blocked IP attempts."""
        since = (datetime.now() - timedelta(hours=since_hours)).isoformat()
        return [a for a in self.blocked_attempts if a['timestamp'] >= since]
    
    def list_whitelist(self) -> List[str]:
        """List all whitelisted IPs."""
        return list(self.whitelist)


class PerKeyRateLimiter:
    """
    Feature #215: Rate Limit per API Key
    
    Enforces rate limits on a per-key basis.
    """
    
    def __init__(
        self,
        default_limit: int = 100,       # Requests per minute
        default_burst: int = 20          # Burst allowance
    ):
        """
        Initialize per-key rate limiter.
        
        Args:
            default_limit: Default requests per minute
            default_burst: Default burst capacity
        """
        self.default_limit = default_limit
        self.default_burst = default_burst
        
        self.key_limits: Dict[str, Dict] = {}  # key_id -> limit config
        self.key_usage: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = threading.Lock()
        
        logger.info(f"Per-Key Rate Limiter initialized - Default: {default_limit}/min")
    
    def set_key_limit(self, key_id: str, requests_per_minute: int, burst: int = 20):
        """Set custom limit for a specific key."""
        self.key_limits[key_id] = {
            'rpm': requests_per_minute,
            'burst': burst
        }
    
    def check(self, key_id: str) -> Tuple[bool, Dict]:
        """
        Check if request is allowed for key.
        
        Returns:
            Tuple of (is_allowed, rate_info)
        """
        with self._lock:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            
            # Get limits for this key
            limits = self.key_limits.get(key_id, {
                'rpm': self.default_limit,
                'burst': self.default_burst
            })
            
            # Clean old entries
            self.key_usage[key_id] = [
                t for t in self.key_usage[key_id]
                if t > minute_ago
            ]
            
            current_usage = len(self.key_usage[key_id])
            
            # Check limit
            if current_usage >= limits['rpm']:
                return False, {
                    'allowed': False,
                    'current_usage': current_usage,
                    'limit': limits['rpm'],
                    'reset_in_seconds': 60 - (now - self.key_usage[key_id][0]).seconds if self.key_usage[key_id] else 0
                }
            
            # Record usage
            self.key_usage[key_id].append(now)
            
            return True, {
                'allowed': True,
                'current_usage': current_usage + 1,
                'limit': limits['rpm'],
                'remaining': limits['rpm'] - current_usage - 1
            }
    
    def get_usage_stats(self, key_id: str) -> Dict:
        """Get usage statistics for a key."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        recent = [t for t in self.key_usage.get(key_id, []) if t > minute_ago]
        hourly = [t for t in self.key_usage.get(key_id, []) if t > hour_ago]
        
        limits = self.key_limits.get(key_id, {'rpm': self.default_limit})
        
        return {
            'key_id': key_id,
            'requests_last_minute': len(recent),
            'requests_last_hour': len(hourly),
            'limit_per_minute': limits['rpm'],
            'utilization_pct': round((len(recent) / limits['rpm']) * 100, 1)
        }


# Singleton instances
_key_manager: Optional[APIKeyManager] = None
_audit_trail: Optional[AuditTrail] = None
_ip_whitelist: Optional[IPWhitelist] = None
_key_rate_limiter: Optional[PerKeyRateLimiter] = None


def get_key_manager() -> APIKeyManager:
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager


def get_audit_trail() -> AuditTrail:
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = AuditTrail()
    return _audit_trail


def get_ip_whitelist() -> IPWhitelist:
    global _ip_whitelist
    if _ip_whitelist is None:
        _ip_whitelist = IPWhitelist()
    return _ip_whitelist


def get_key_rate_limiter() -> PerKeyRateLimiter:
    global _key_rate_limiter
    if _key_rate_limiter is None:
        _key_rate_limiter = PerKeyRateLimiter()
    return _key_rate_limiter


if __name__ == '__main__':
    # Test API Key Manager
    km = APIKeyManager()
    result = km.store_key('binance', 'my_api_key', 'my_secret')
    print(f"Stored key: {result}")
    
    retrieved = km.get_key(result['key_id'])
    print(f"Retrieved: {retrieved}")
    
    # Test Audit Trail
    audit = AuditTrail()
    audit.log('LOGIN', 'user123', 'system', {'ip': '192.168.1.1'})
    audit.log('TRADE_EXECUTED', 'bot', 'BTCUSDT', {'side': 'BUY', 'size': 0.1})
    print(f"Audit summary: {audit.get_summary()}")
    
    # Test IP Whitelist
    wl = IPWhitelist()
    wl.add_ip('192.168.1.100', 'Office')
    wl.enable()
    print(f"Check 192.168.1.100: {wl.check('192.168.1.100')}")
    print(f"Check 10.0.0.1: {wl.check('10.0.0.1')}")
