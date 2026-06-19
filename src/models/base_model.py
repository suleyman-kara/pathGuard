from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Optional, Dict

class BaseVariantModel(ABC):
    """
    Abstract Base Class standardizing the estimator interface for PathGuard framework.
    Ensures seamless compatibility with cross-validation loops, calibration, and stacking.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        self.model_params = model_params or {}
        self.model: Optional[Any] = None
        self.is_fitted = False
        
    @abstractmethod
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "BaseVariantModel":
        """
        Trains the underlying model, leveraging validation sets for early stopping if available.
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns continuous probabilities for class 1 (Pathogenic).
        Shape: (n_samples,)
        """
        pass
        
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Returns binary predictions based on custom thresholding.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
        
    @abstractmethod
    def save(self, file_path: str) -> None:
        """Saves model weights to local path."""
        pass
        
    @abstractmethod
    def load(self, file_path: str) -> "BaseVariantModel":
        """Loads model weights from local path."""
        pass
