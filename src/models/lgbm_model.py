import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
from typing import Any, Optional, Dict
from src.models.base_model import BaseVariantModel
from src.config import EARLY_STOPPING_ROUNDS, RANDOM_SEED

class LightGBMVariantModel(BaseVariantModel):
    """
    Missense genetik varyantları patojenik/benign olarak sınıflandırmak için LightGBM Modeli.
    Eksik verileri (NaN) yerleşik olarak yönetir ve dengesiz sınıfları scale_pos_weight ile ayarlar.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__(model_params)
        # Varsayılan temel model parametreleri
        self.default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 1000,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1
        }
        self.final_params = {**self.default_params, **self.model_params}
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "LightGBMVariantModel":
        
        # Eğer parametrelerde belirtilmemişse dengesiz veri seti için ağırlık katsayısını hesapla
        if "scale_pos_weight" not in self.final_params:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            self.final_params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            
        # Pandas category tipindeki kolonları otomatik olarak kategorik özellik olarak belirle
        cat_features = [col for col in X_train.columns if X_train[col].dtype.name == "category"]
        
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        callbacks = []
        if eval_set:
            callbacks.append(lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False))
            
        # LightGBM sınıflandırıcı nesnesini oluştur ve eğit
        self.model = lgb.LGBMClassifier(**self.final_params)
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
            raise ValueError("Tahmin yapılabilmesi için önce model eğitilmelidir.")
        # Sadece patojenik sınıfın (sınıf 1) olasılığını döndür
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, file_path: str) -> None:
        if not self.is_fitted:
            raise ValueError("Eğitilmemiş model kaydedilemez.")
        joblib.dump({"model": self.model, "params": self.final_params}, file_path)
        
    def load(self, file_path: str) -> "LightGBMVariantModel":
        data = joblib.load(file_path)
        self.model = data["model"]
        self.final_params = data["params"]
        self.is_fitted = True
        return self
