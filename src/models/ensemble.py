import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from src.models.base_model import BaseVariantModel
from src.config import CLINICAL_RECALL_TARGET

class CalibratedVariantModel:
    """
    Bir temel model (LightGBM veya XGBoost) etrafında olasılık kalibrasyon katmanı.
    İzotonik Regresyon (Isotonic Regression) kullanarak ham model olasılıklarını
    gerçek popülasyon olasılıklarına dönüştürür.
    """
    def __init__(self, base_model: BaseVariantModel):
        self.base_model = base_model
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted = False
        
    def fit_calibration(self, oof_probs: np.ndarray, y_true: np.ndarray) -> "CalibratedVariantModel":
        # Out-of-fold (OOF) doğrulaması olasılıkları üzerinde kalibratörü eğitir
        self.calibrator.fit(oof_probs, y_true)
        self.is_fitted = True
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_probs = self.base_model.predict_proba(X)
        if self.is_fitted:
            return self.calibrator.transform(raw_probs)
        return raw_probs

class SoftVotingEnsemble:
    """
    Birden fazla kalibre edilmiş modelin tahminlerini ağırlıklı ortalama ile birleştiren sınıf.
    """
    def __init__(self, models: List[CalibratedVariantModel], weights: Optional[List[float]] = None):
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            s = sum(weights)
            self.weights = [w / s for w in weights] # Ağırlıkları normalize et
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        final_probs = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            final_probs += weight * model.predict_proba(X)
        return final_probs

class LogisticStackingEnsemble:
    """
    Kalibre edilmiş modellerin çıktı olasılıkları üzerinde eğitilen
    Lojistik Regresyon tabanlı Stacking (meta-öğrenme) katmanı.
    """
    def __init__(
        self,
        models: List[CalibratedVariantModel],
        stacker: Optional[LogisticRegression] = None
    ):
        self.models = models
        self.stacker = stacker or LogisticRegression(max_iter=1000, class_weight="balanced")
        self.is_fitted = stacker is not None

    def _meta_features(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([model.predict_proba(X) for model in self.models])

    def fit(self, base_oof_probs: np.ndarray, y_true: np.ndarray) -> "LogisticStackingEnsemble":
        self.stacker.fit(base_oof_probs, y_true)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Stacking Ensemble eğitilmeden tahmin yapılamaz.")
        return self.stacker.predict_proba(self._meta_features(X))[:, 1]

def optimize_decision_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    recall_target: float = CLINICAL_RECALL_TARGET
) -> Tuple[float, float, float]:
    """
    Karar Eşiği Optimizasyonu:
    - Olasılık çıktıları (0.01 - 0.99) arasında tarama yapar.
    - recall >= recall_target kısıtını sağlayan eşikler arasından en yüksek Patojenik
      Sınıf F1 Skorunu (Class 1 F1) veren eşiği seçer.

    Not: recall_target varsayılan olarak 0.0'dır (config.CLINICAL_RECALL_TARGET). Bu durumda
    her eşik kısıtı sağlar; dolayısıyla fonksiyon doğrudan yarışma metriği olan Class 1 F1'i
    maksimize eden eşiği döndürür. Önceki 0.90'lık klinik recall kısıtı kaldırılmıştır
    (bkz. docs/rapor_guncellemeleri.md). recall_target > 0 verilirse klinik recall tabanı
    yine uygulanabilir.

    Girdi:
        y_true: Gerçek etiketler (0/1)
        probs: Tahmin olasılıkları
        recall_target: Sağlanması gereken minimum duyarlılık tabanı (varsayılan: 0.0 = kısıtsız)
    Çıktı:
        (optimal_esik, en_iyi_f1, saglanan_duyarlilik)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_f1 = -1.0
    best_recall = -1.0
    
    viable_candidates = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        # Macro F1 yerine doğrudan pathogenic sınıfı (class 1) F1 skorunu alıyoruz
        c1_f1 = f1_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        
        # Klinik duyarlılık kısıtını kontrol et
        if recall >= recall_target:
            viable_candidates.append((c1_f1, recall, t))
            
    if viable_candidates:
        # Patojenik F1 skoruna göre azalan sırada sırala ve en yüksek olanı seç
        viable_candidates.sort(key=lambda x: x[0], reverse=True)
        best_f1, best_recall, best_thresh = viable_candidates[0]
    else:
        # Eğer hiçbir eşik klinik recall kısıtını sağlamıyorsa, doğrudan F1-pathogenic skorunu en üst düzeye çıkaranı seç
        for t in thresholds:
            preds = (probs >= t).astype(int)
            c1_f1 = f1_score(y_true, preds, zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
            if c1_f1 > best_f1:
                best_f1 = c1_f1
                best_recall = recall
                best_thresh = t
                
    return float(best_thresh), float(best_f1), float(best_recall)
