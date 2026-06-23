import pandas as pd
import numpy as np
import joblib
from typing import Any, Optional, Dict, List
from src.models.lgbm_model import LightGBMVariantModel
from src.models.xgb_model import XGBoostVariantModel

class PanelVariantModel:
    """
    Bağımsız Panel Modeli.
    Her hastalık paneli (KANSER, PAH, CFTR) için master modelinden TAMAMEN bağımsız
    şekilde, yalnızca kendi panel verisi üzerinde eğitilir.

    Master modelinin yumuşak tahminini (soft prediction) meta-özellik olarak EKLEMEZ.
    Bu kasıtlı bir tasarım kararıdır: master, tüm master setiyle eğitildiği ve paneller
    master'ın alt kümesi olduğu için master tahmini panel doğrulama satırlarında ezberlenmiş
    (sızdırılmış) etiket taşıyordu ve yanıltıcı yüksek OOF F1 üretiyordu. Bağımsız eğitim
    hem bu sızıntıyı ortadan kaldırır hem de yarışmanın "her panel için ayrı/bağımsız model"
    şartına (soru-cevap.md) uyar.

    İki mod:
    - use_ensemble=False (varsayılan): tek, sıkı düzenlileştirilmiş LightGBM. Küçük örneklemli
      paneller için aşırı öğrenmeyi güçlü biçimde sınırlar.
    - use_ensemble=True: LightGBM + XGBoost (ikisi de sıkı düzenlileştirilmiş) HAM (kalibrasyonsuz)
      ağırlıklı ortalama (soft-voting). Ensemble yalnızca pipeline'daki panel-başına geçiş (gate)
      OOF test-prior F1'i tek-LGBM'i geçtiğinde kullanılır.

      NOT (kalibrasyon neden yok): OOF üzerinde fit edilen izotonik kalibratörleri full-data
      modelin olasılıklarına uygulamak, OOF'ta seçilen eşiğin inference'a transfer olmamasına yol
      açıyordu (özellikle CFTR: kalibrasyon inference olasılıklarını eşik altına sıkıştırıp tüm
      tahminleri 0 yapıyordu). Ham ortalama tek-model gibi sorunsuz transfer eder ve çeşitlilik
      kazancını (ensemble'ın asıl faydası) korur.
    """
    def __init__(
        self,
        panel_params: Optional[Dict[str, Any]] = None,
        use_ensemble: bool = False,
        weights: Optional[List[float]] = None,
    ):
        # Küçük örneklemler için sıkı parametre kısıtları (LightGBM)
        default_panel_params = {
            "num_leaves": 15,
            "min_child_samples": 5,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
        }
        lgbm_params = {**default_panel_params, **(panel_params or {})}
        self.panel_model = LightGBMVariantModel(model_params=lgbm_params)

        self.use_ensemble = use_ensemble
        self.weights = list(weights) if weights is not None else [0.5, 0.5]

        if use_ensemble:
            # XGBoost üyesi için sıkı düzenlileştirme + sınırlı ağaç sayısı (panel OOF döngüsünde
            # erken durdurma yok; küçük veride aşırı öğrenmeyi ve süreyi sınırlamak için).
            default_xgb_params = {
                "max_depth": 3,
                "min_child_weight": 5,
                "reg_alpha": 1.0,
                "reg_lambda": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_estimators": 300,
                "learning_rate": 0.05,
            }
            self.xgb_model: Optional[XGBoostVariantModel] = XGBoostVariantModel(model_params=default_xgb_params)
        else:
            self.xgb_model = None

        self.is_fitted = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "PanelVariantModel":

        # Panel LightGBM'ini doğrudan kendi panel özellik matrisi üzerinde eğit
        self.panel_model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        if self.use_ensemble and self.xgb_model is not None:
            self.xgb_model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("PanelVariantModel tahmin yapılabilmesi için eğitilmelidir.")
        p_lgb = self.panel_model.predict_proba(X)
        if not self.use_ensemble or self.xgb_model is None:
            return p_lgb
        p_xgb = self.xgb_model.predict_proba(X)
        return self.weights[0] * p_lgb + self.weights[1] * p_xgb

    def save(self, file_path: str) -> None:
        if not self.is_fitted:
            raise ValueError("Eğitilmemiş model kaydedilemez.")
        payload = {
            "format": "panel_v2",
            "use_ensemble": self.use_ensemble,
            "weights": self.weights,
            "lgbm": {"model": self.panel_model.model, "params": self.panel_model.final_params},
            "xgb": (
                {"model": self.xgb_model.model, "params": self.xgb_model.final_params}
                if (self.use_ensemble and self.xgb_model is not None) else None
            ),
        }
        joblib.dump(payload, file_path)

    def load(self, file_path: str) -> "PanelVariantModel":
        data = joblib.load(file_path)
        # Geriye dönük uyum: eski format tek LightGBM'i {"model":..., "params":...} olarak saklıyordu.
        if not isinstance(data, dict) or data.get("format") != "panel_v2":
            self.panel_model.model = data["model"]
            self.panel_model.final_params = data["params"]
            self.panel_model.is_fitted = True
            self.use_ensemble = False
            self.xgb_model = None
            self.is_fitted = True
            return self

        self.use_ensemble = data["use_ensemble"]
        self.weights = list(data["weights"])
        self.panel_model.model = data["lgbm"]["model"]
        self.panel_model.final_params = data["lgbm"]["params"]
        self.panel_model.is_fitted = True
        if self.use_ensemble and data["xgb"] is not None:
            self.xgb_model = XGBoostVariantModel()
            self.xgb_model.model = data["xgb"]["model"]
            self.xgb_model.final_params = data["xgb"]["params"]
            self.xgb_model.is_fitted = True
        else:
            self.xgb_model = None
        self.is_fitted = True
        return self
