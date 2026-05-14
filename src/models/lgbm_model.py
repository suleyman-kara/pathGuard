import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
from typing import Any, Optional, Dict, Tuple
from src.models.base_model import BaseVariantModel
from src.config import EARLY_STOPPING_ROUNDS, RANDOM_SEED

class LightGBMVariantModel(BaseVariantModel):
    """
    LightGBM Estimator implementation for PathGuard variant classification.
    Optimized for high-dimensional, sparse variant profiles with embedded missingness.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__(model_params)
        # Default base configuration
        self.default_params = {
            "objective": "binary",
            "metric": "binary_logloss", # Using standard logloss for smooth gradient steps
            "boosting_type": "gbdt",
            "n_estimators": 1000,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1
        }
        # Override defaults with provided params
        self.final_params = {**self.default_params, **self.model_params}
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "LightGBMVariantModel":
        
        # Calculate optimal class weight scaling if not explicitly overridden
        if "scale_pos_weight" not in self.final_params:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            # Ensure safe division
            self.final_params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            
        # Extract native categorical features explicitly if present as pandas categories
        cat_features = [col for col in X_train.columns if X_train[col].dtype.name == "category"]
        
        # Prepare evaluation sets
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        callbacks = []
        if eval_set:
            callbacks.append(lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False))
            
        self.model = lgb.LGBMClassifier(**self.final_params)
        
        # Train estimator
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set if eval_set else None,
            categorical_feature=cat_features if cat_features else "auto",
            callbacks=callbacks if callbacks else None
        )
        
        self.is_fitted = True
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before inference.")
        # Return probability corresponding to class 1
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, file_path: str) -> None:
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
        joblib.dump({"model": self.model, "params": self.final_params}, file_path)
        
    def load(self, file_path: str) -> "LightGBMVariantModel":
        data = joblib.load(file_path)
        self.model = data["model"]
        self.final_params = data["params"]
        self.is_fitted = True
        return self
