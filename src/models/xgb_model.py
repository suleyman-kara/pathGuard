import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from typing import Any, Optional, Dict
from src.models.base_model import BaseVariantModel
from src.config import EARLY_STOPPING_ROUNDS, RANDOM_SEED

class XGBoostVariantModel(BaseVariantModel):
    """
    XGBoost Estimator implementation for PathGuard variant classification.
    Leverages depth-wise tree growth to capture highly specific regional non-linearities.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__(model_params)
        self.default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 1000,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "enable_categorical": True  # Enable native pandas category splitting support
        }
        self.final_params = {**self.default_params, **self.model_params}
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "XGBoostVariantModel":
        
        # Calculate optimal class weight scale if omitted
        if "scale_pos_weight" not in self.final_params:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            self.final_params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model = xgb.XGBClassifier(**self.final_params)
        
        # Fit estimator
        fit_params: Dict[str, Any] = {}
        if eval_set:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False
            # set early stopping round via class constructor or fit depending on xgb version
            # passing early_stopping_rounds to constructor is safest in xgb>=2.0
            self.model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            
        self.model.fit(X_train, y_train, **fit_params)
        
        self.is_fitted = True
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before inference.")
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, file_path: str) -> None:
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
        joblib.dump({"model": self.model, "params": self.final_params}, file_path)
        
    def load(self, file_path: str) -> "XGBoostVariantModel":
        data = joblib.load(file_path)
        self.model = data["model"]
        self.final_params = data["params"]
        self.is_fitted = True
        return self
