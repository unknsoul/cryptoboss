"""
CryptoBoss 1.0.1 - Authentication Service

Handles user signup, login, logout, and JWT token management.
"""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from ..models.user import User, ExchangeAccount, UserSession

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET = "cryptoboss_jwt_secret_change_in_production"  # Override via env
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24


@dataclass
class AuthResult:
    """Result of authentication operation."""
    success: bool
    user: Optional[User] = None
    token: Optional[str] = None
    error: Optional[str] = None


class AuthService:
    """
    Authentication service for user management.
    
    Handles:
    - User signup with email/password
    - User login with JWT token generation
    - Token verification
    - Session management
    """
    
    def __init__(self, user_repository=None, jwt_secret: str = None):
        """
        Initialize auth service.
        
        Args:
            user_repository: Database repository for users (injected)
            jwt_secret: Secret for JWT signing (uses default if not provided)
        """
        self.user_repo = user_repository or InMemoryUserRepository()
        self.jwt_secret = jwt_secret or JWT_SECRET
        self.sessions: Dict[str, UserSession] = {}
        logger.info("🔐 AuthService initialized")
    
    def signup(self, email: str, password: str) -> AuthResult:
        """
        Create a new user account.
        
        Returns AuthResult with user and token on success.
        """
        # Validate input
        if not email or "@" not in email:
            return AuthResult(success=False, error="Invalid email address")
        
        if not password or len(password) < 8:
            return AuthResult(success=False, error="Password must be at least 8 characters")
        
        # Check if user exists
        existing = self.user_repo.find_by_email(email)
        if existing:
            return AuthResult(success=False, error="Email already registered")
        
        # Create user
        user = User.create(email=email, password=password)
        self.user_repo.save(user)
        
        # Generate token
        token = self._generate_token(user)
        
        logger.info(f"✅ New user created: {email}")
        return AuthResult(success=True, user=user, token=token)
    
    def login(self, email: str, password: str) -> AuthResult:
        """
        Authenticate user and return JWT token.
        """
        # Find user
        user = self.user_repo.find_by_email(email.lower().strip())
        if not user:
            return AuthResult(success=False, error="Invalid email or password")
        
        # Verify password
        if not user.verify_password(password):
            return AuthResult(success=False, error="Invalid email or password")
        
        # Check if active
        if not user.is_active:
            return AuthResult(success=False, error="Account is disabled")
        
        # Generate token
        token = self._generate_token(user)
        
        # Create session
        session = UserSession.create(user.user_id)
        self.sessions[session.session_id] = session
        
        logger.info(f"✅ User logged in: {email}")
        return AuthResult(success=True, user=user, token=token)
    
    def verify_token(self, token: str) -> Optional[User]:
        """
        Verify JWT token and return user if valid.
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("user_id")
            if not user_id:
                return None
            
            user = self.user_repo.find_by_id(user_id)
            if not user or not user.is_active:
                return None
            
            return user
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def logout(self, token: str) -> bool:
        """
        Invalidate a session/token.
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[JWT_ALGORITHM])
            session_id = payload.get("session_id")
            if session_id and session_id in self.sessions:
                del self.sessions[session_id]
                logger.info("User logged out")
            return True
        except:
            return False
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.user_repo.find_by_id(user_id)
    
    def _generate_token(self, user: User) -> str:
        """Generate JWT token for user."""
        session = UserSession.create(user.user_id)
        
        payload = {
            "user_id": user.user_id,
            "email": user.email,
            "session_id": session.session_id,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=JWT_ALGORITHM)


class InMemoryUserRepository:
    """
    In-memory user repository for development/testing.
    
    Replace with database repository in production.
    """
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.emails: Dict[str, str] = {}  # email -> user_id
    
    def save(self, user: User):
        """Save or update a user."""
        self.users[user.user_id] = user
        self.emails[user.email.lower()] = user.user_id
    
    def find_by_id(self, user_id: str) -> Optional[User]:
        """Find user by ID."""
        return self.users.get(user_id)
    
    def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email."""
        user_id = self.emails.get(email.lower())
        if user_id:
            return self.users.get(user_id)
        return None
    
    def delete(self, user_id: str) -> bool:
        """Delete a user."""
        user = self.users.get(user_id)
        if user:
            del self.users[user_id]
            if user.email.lower() in self.emails:
                del self.emails[user.email.lower()]
            return True
        return False


# Singleton
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """
    Get the global AuthService instance.
    
    CRYPTOBOSS 2.0: Uses SQLite repository for persistent storage.
    Users now survive server restarts.
    """
    global _auth_service
    if _auth_service is None:
        # Use SQLite for persistence
        try:
            from ..database.repository import get_repository
            repository = get_repository()
            _auth_service = AuthService(user_repository=repository)
            logger.info("🔐 AuthService using SQLite repository (persistent)")
        except ImportError:
            # Fallback to in-memory for testing
            _auth_service = AuthService()
            logger.warning("⚠️ AuthService using in-memory repository (NOT persistent)")
    return _auth_service
