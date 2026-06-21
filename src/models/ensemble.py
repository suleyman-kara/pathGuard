import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from src.models.base_model import BaseVariantModel
from src.config import CLINICAL_RECALL_TARGET, TEST_PATHOGENIC_PRIOR

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

def prior_adjusted_class1_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_prior: Optional[float]
) -> float:
    """
    Patojenik (class 1) F1'ini, verilen test prior'u (azınlık sınıf oranı) altında hesaplar.

    Recall (TPR) yalnızca gerçek pozitifler, FPR yalnızca gerçek negatifler üzerinden
    hesaplandığı için ikisi de sınıf oranından (prior) BAĞIMSIZDIR; eğitim dağılımında
    ölçülüp test dağılımına taşınabilir. Hedef prior π altında precision yeniden kurulur:
        precision_π = π·TPR / (π·TPR + (1−π)·FPR)
        F1_π        = 2·precision_π·TPR / (precision_π + TPR)

    test_prior None ise standart (gözlemlenen dağılımdaki) Class 1 F1 döner.
    """
    if test_prior is None:
        return float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    denom = test_prior * tpr + (1.0 - test_prior) * fpr
    if denom <= 0 or tpr <= 0:
        return 0.0
    precision = test_prior * tpr / denom
    return float(2 * precision * tpr / (precision + tpr))


def optimize_decision_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    test_prior: Optional[float] = TEST_PATHOGENIC_PRIOR,
    recall_target: float = CLINICAL_RECALL_TARGET
) -> Tuple[float, float, float]:
    """
    Karar Eşiği Optimizasyonu (test dağılımına göre):
    - Olasılık çıktıları (0.01 - 0.99) arasında tarama yapar.
    - recall >= recall_target kısıtını sağlayan eşikler arasından, **test prior'u altında**
      (varsayılan ~%20 patojenik) hesaplanan Patojenik Sınıf F1'ini (F1_π) maksimize eden
      eşiği seçer.

    Eğitim/OOF dağılımı (~%80 patojenik) test dağılımının tersi olduğundan, eşiği gözlemlenen
    F1'e göre seçmek test setinde precision'ı çökertir (bkz. docs/rapor_guncellemeleri.md).
    Bu fonksiyon eşiği gerçek değerlendirme dağılımına hizalar. test_prior=None verilirse eski
    davranışa (gözlemlenen dağılımda F1) döner.

    Girdi:
        y_true: Gerçek etiketler (0/1)
        probs: Tahmin olasılıkları
        test_prior: Beklenen test patojenik oranı (None ise gözlemlenen dağılım kullanılır)
        recall_target: Sağlanması gereken minimum duyarlılık tabanı (varsayılan: 0.0 = kısıtsız)
    Çıktı:
        (optimal_esik, en_iyi_f1_test_prior, saglanan_recall)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_f1 = -1.0
    best_recall = -1.0

    viable_candidates = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        # Eşik skoru: test prior'u altındaki patojenik (class 1) F1
        f1_obj = prior_adjusted_class1_f1(y_true, preds, test_prior)
        recall = recall_score(y_true, preds, zero_division=0)

        # Recall tabanı kısıtını kontrol et (varsayılan 0 → tüm eşikler uygun)
        if recall >= recall_target:
            viable_candidates.append((f1_obj, recall, t))

    if viable_candidates:
        # Test-prior F1'ine göre azalan sırada sırala ve en yüksek olanı seç
        viable_candidates.sort(key=lambda x: x[0], reverse=True)
        best_f1, best_recall, best_thresh = viable_candidates[0]
    else:
        # Hiçbir eşik recall tabanını sağlamıyorsa doğrudan test-prior F1'ini maksimize et
        for t in thresholds:
            preds = (probs >= t).astype(int)
            f1_obj = prior_adjusted_class1_f1(y_true, preds, test_prior)
            recall = recall_score(y_true, preds, zero_division=0)
            if f1_obj > best_f1:
                best_f1 = f1_obj
                best_recall = recall
                best_thresh = t

    return float(best_thresh), float(best_f1), float(best_recall)
