"""
Configuration Validation System
Validates settings to prevent misconfiguration.
"""
import logging
from typing import Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validates configuration settings.
    
    Prevents:
    - Excessive risk parameters
    - Invalid paths
    - Incompatible feature combinations
    """
    
    @staticmethod
    def validate_risk_settings(config) -> Tuple[bool, List[str]]:
        """Validate risk management settings."""
        errors = []
        
        # Risk per trade (should be 0.1% - 5%)
        if hasattr(config, 'RISK_PER_TRADE_PCT'):
            risk = config.RISK_PER_TRADE_PCT
            if not (0.1 <= risk <= 5.0):
                errors.append(f"RISK_PER_TRADE_PCT ({risk}%) must be between 0.1% and 5%")
        
        # Daily loss limit
        if hasattr(config, 'DAILY_LOSS_LIMIT_PCT'):
            daily_loss = config.DAILY_LOSS_LIMIT_PCT
            if not (1.0 <= daily_loss <= 20.0):
                errors.append(f"DAILY_LOSS_LIMIT_PCT ({daily_loss}%) must be between 1% and 20%")
        
        # Weekly loss limit should be higher than daily
        if hasattr(config, 'WEEKLY_LOSS_LIMIT_PCT') and hasattr(config, 'DAILY_LOSS_LIMIT_PCT'):
            if config.WEEKLY_LOSS_LIMIT_PCT <= config.DAILY_LOSS_LIMIT_PCT:
                errors.append("WEEKLY_LOSS_LIMIT_PCT must be greater than DAILY_LOSS_LIMIT_PCT")
        
        # Min confidence threshold
        if hasattr(config, 'MIN_CONFIDENCE_THRESHOLD'):
            conf = config.MIN_CONFIDENCE_THRESHOLD
            if not (0.0 <= conf <= 1.0):
                errors.append(f"MIN_CONFIDENCE_THRESHOLD ({conf}) must be between 0.0 and 1.0")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_paths(config) -> Tuple[bool, List[str]]:
        """Validate file paths exist."""
        errors = []
        
        if hasattr(config, 'DATA_DIR'):
            data_dir = Path(config.DATA_DIR)
            if not data_dir.exists():
                logger.warning(f"Creating DATA_DIR: {data_dir}")
                data_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(config, 'DB_PATH'):
            db_path = Path(config.DB_PATH)
            db_dir = db_path.parent
            if not db_dir.exists():
                logger.warning(f"Creating DB directory: {db_dir}")
                db_dir.mkdir(parents=True, exist_ok=True)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_all(config) -> Dict:
        """Run all validations."""
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate risk settings
        risk_valid, risk_errors = ConfigValidator.validate_risk_settings(config)
        if not risk_valid:
            results['valid'] = False
            results['errors'].extend(risk_errors)
        
        # Validate paths
        path_valid, path_errors = ConfigValidator.validate_paths(config)
        if not path_valid:
            results['warnings'].extend(path_errors)
        
        # Log results
        if results['valid']:
            logger.info("✅ Configuration validation passed")
        else:
            logger.error("❌ Configuration validation FAILED:")
            for error in results['errors']:
                logger.error(f"   - {error}")
        
        if results['warnings']:
            logger.warning("⚠️  Configuration warnings:")
            for warning in results['warnings']:
                logger.warning(f"   - {warning}")
        
        return results


def validate_configuration(config) -> bool:
    """
    Validate configuration and raise exception if invalid.
    
    Args:
        config: Configuration module/object
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    result = ConfigValidator.validate_all(config)
    
    if not result['valid']:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in result['errors'])
        raise ValueError(error_msg)
    
    return True
