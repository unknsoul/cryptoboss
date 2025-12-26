"""
Model Registry
Version control and metadata management for ML models.

Stores:
- Trained model (.pkl)
- Feature list (.json)
- Scaler (.pkl)
- Training metadata (.json)
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Model version control and storage.
    
    Prevents train/serve skew by storing:
    - Model itself
    - Exact feature list used
    - Scaler/preprocessor
    - Training metadata (date, metrics, config)
    """
    
    def __init__(self, registry_dir: str = "models/registry"):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store models
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model Registry initialized at {self.registry_dir}")
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        features: List[str],
        scaler: Optional[Any] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save model with all artifacts.
        
        Args:
            model: Trained model object
            model_name: Name/version identifier
            features: List of feature names used
            scaler: Fitted scaler (StandardScaler, etc.)
            metadata: Training metadata (metrics, config, etc.)
            
        Returns:
            Version ID (timestamp-based)
        """
        # Create version ID
        version_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = self.registry_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {model_path}")
        
        # Save features list
        features_path = version_dir / "features.json"
        with open(features_path, 'w') as f:
            json.dump({'features': features}, f, indent=2)
        logger.info(f"Saved {len(features)} features to {features_path}")
        
        # Save scaler if provided
        if scaler is not None:
            scaler_path = version_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved scaler to {scaler_path}")
        
        # Save metadata
        full_metadata = {
            'version_id': version_id,
            'model_name': model_name,
            'created_at': datetime.now().isoformat(),
            'num_features': len(features),
            'has_scaler': scaler is not None
        }
        if metadata:
            full_metadata.update(metadata)
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")
        
        logger.info(f"✅ Model registered: {version_id}")
        return version_id
    
    def load_model(self, version_id: str) -> Dict[str, Any]:
        """
        Load model with all artifacts.
        
        Args:
            version_id: Version identifier
            
        Returns:
            Dict with model, features, scaler, metadata
        """
        version_dir = self.registry_dir / version_id
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version_id} not found")
        
        # Load model
        model_path = version_dir / "model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load features
        features_path = version_dir / "features.json"
        with open(features_path, 'r') as f:
            features = json.load(f)['features']
        
        # Load scaler if exists
        scaler_path = version_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = None
        
        # Load metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded model version: {version_id}")
        
        return {
            'model': model,
            'features': features,
            'scaler': scaler,
            'metadata': metadata
        }
    
    def list_models(self) -> List[Dict]:
        """List all registered models."""
        models = []
        
        for version_dir in self.registry_dir.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models.append(metadata)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return models
    
    def get_latest_model(self, model_name: str) -> Optional[str]:
        """Get latest version ID for a model name."""
        models = self.list_models()
        for model in models:
            if model.get('model_name') == model_name:
                return model.get('version_id')
        return None
