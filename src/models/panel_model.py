import pandas as pd
import numpy as np
from typing import Any, Optional, Dict
from src.models.lgbm_model import LightGBMVariantModel

class PanelMetaLearner:
    """
    Panel-Aware Meta-Learning Modeli.
    Genel Master modelinin çıktı tahmin olasılığını bir meta-özellik (prior probability) olarak alıp,
    ilgili alt panelin varyant profiline ekler. Bu sayede CFTR (N=111) gibi çok küçük
    panel veri setlerinde aşırı öğrenmeyi (overfitting) güçlü bir şekilde önler.
    """
    def __init__(self, master_model: Any, panel_params: Optional[Dict[str, Any]] = None):
        self.master_model = master_model
        # Küçük örneklemler için sıkı parametre kısıtları uyguluyoruz
        default_panel_params = {
            "num_leaves": 15,
            "min_child_samples": 5,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0
        }
        final_params = {**default_panel_params, **(panel_params or {})}
        self.panel_model = LightGBMVariantModel(model_params=final_params)
        self.is_fitted = False
        
    def _augment_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Master modelin yumuşak tahminlerini (soft predictions) ek özellik olarak veri matrisine ekle
        X_aug = X.copy()
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
        
        # Özellik matrisini genişlet (Master tahmini kolonunu ekle)
        X_train_aug = self._augment_features(X_train)
        X_val_aug = self._augment_features(X_val) if X_val is not None else None
        
        # Alt panel modelini eğit
        self.panel_model.train(X_train_aug, y_train, X_val=X_val_aug, y_val=y_val)
        self.is_fitted = True
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("PanelMetaLearner tahmin yapılabilmesi için eğitilmelidir.")
        X_aug = self._augment_features(X)
        return self.panel_model.predict_proba(X_aug)
        
    def save(self, file_path: str) -> None:
        self.panel_model.save(file_path)
        
    def load(self, file_path: str) -> "PanelMetaLearner":
        self.panel_model.load(file_path)
        self.is_fitted = True
        return self
