import pandas as pd
import numpy as np
from typing import Any, Optional, Dict
from src.models.lgbm_model import LightGBMVariantModel

class PanelVariantModel:
    """
    Bağımsız Panel Modeli.
    Her hastalık paneli (KANSER, PAH, CFTR) için master modelinden TAMAMEN bağımsız
    şekilde, yalnızca kendi panel verisi üzerinde eğitilen tek bir LightGBM modeli.

    Master modelinin yumuşak tahminini (soft prediction) meta-özellik olarak EKLEMEZ.
    Bu kasıtlı bir tasarım kararıdır: master, tüm master setiyle eğitildiği ve paneller
    master'ın alt kümesi olduğu için master tahmini panel doğrulama satırlarında ezberlenmiş
    (sızdırılmış) etiket taşıyordu ve yanıltıcı yüksek OOF F1 üretiyordu. Bağımsız eğitim
    hem bu sızıntıyı ortadan kaldırır hem de yarışmanın "her panel için ayrı/bağımsız model"
    şartına (soru-cevap.md) uyar.

    Küçük örneklemli paneller (örn. CFTR, N=111) için sıkı düzenlileştirme parametreleri
    aşırı öğrenmeyi (overfitting) güçlü biçimde sınırlar.
    """
    def __init__(self, panel_params: Optional[Dict[str, Any]] = None):
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

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "PanelVariantModel":

        # Panel modelini doğrudan kendi panel özellik matrisi üzerinde eğit
        self.panel_model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("PanelVariantModel tahmin yapılabilmesi için eğitilmelidir.")
        return self.panel_model.predict_proba(X)

    def save(self, file_path: str) -> None:
        self.panel_model.save(file_path)

    def load(self, file_path: str) -> "PanelVariantModel":
        self.panel_model.load(file_path)
        self.is_fitted = True
        return self
