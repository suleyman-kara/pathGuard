import pandas as pd
import numpy as np
from typing import Any, Optional, Dict
from src.models.base_model import BaseVariantModel
from src.models.lgbm_model import LightGBMVariantModel

class PanelMetaLearner:
    """
    Implements the Panel-Aware Meta-Learning pipeline:
    Fuses soft predictions from a parent Master model as a highly informative prior
    feature alongside localized panel-specific variant profiles. Prevents overfitting
    on extremely small sample sets (e.g. CFTR panel N=111).
    """
    def __init__(self, master_model: Any, panel_params: Optional[Dict[str, Any]] = None):
        self.master_model = master_model
        # Configure localized panel model with tighter constraints to prevent small sample overfitting
        default_panel_params = {
            "num_leaves": 15,  # Lower capacity for small sample sizes
            "min_child_samples": 5, # Allow smaller leaf splits for tiny panels
            "reg_alpha": 1.0,  # Strict L1 constraint
            "reg_lambda": 1.0  # Strict L2 constraint
        }
        final_params = {**default_panel_params, **(panel_params or {})}
        self.panel_model = LightGBMVariantModel(model_params=final_params)
        self.is_fitted = False
        
    def _augment_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Appends the master model's global prediction probabilities as a meta-feature."""
        X_aug = X.copy()
        # Ensure we predict probabilities using the trained master model wrapper or soft voting ensemble
        master_probs = self.master_model.predict_proba(X)
        X_aug["master_soft_prediction"] = master_probs
        return X_aug
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "PanelMetaLearner":
        
        # Augment feature matrices
        X_train_aug = self._augment_features(X_train)
        X_val_aug = self._augment_features(X_val) if X_val is not None else None
        
        # Train localized panel sub-model
        self.panel_model.train(X_train_aug, y_train, X_val=X_val_aug, y_val=y_val)
        self.is_fitted = True
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("PanelMetaLearner must be trained before inference.")
        X_aug = self._augment_features(X)
        return self.panel_model.predict_proba(X_aug)
        
    def save(self, file_path: str) -> None:
        self.panel_model.save(file_path)
        
    def load(self, file_path: str) -> "PanelMetaLearner":
        self.panel_model.load(file_path)
        self.is_fitted = True
        return self
