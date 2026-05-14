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
    Wraps a base model with an Isotonic Regression calibrator.
    Maps uncalibrated tree outputs to true empirical target posteriors.
    """
    def __init__(self, base_model: BaseVariantModel):
        self.base_model = base_model
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted = False
        
    def fit_calibration(self, oof_probs: np.ndarray, y_true: np.ndarray) -> "CalibratedVariantModel":
        """Fits isotonic regression post-hoc on Out-of-Fold validation probabilities."""
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
    Combines multiple calibrated estimators via optimized soft-voting probability combination.
    """
    def __init__(self, models: List[CalibratedVariantModel], weights: Optional[List[float]] = None):
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            s = sum(weights)
            self.weights = [w / s for w in weights]
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        final_probs = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            final_probs += weight * model.predict_proba(X)
        return final_probs

class LogisticStackingEnsemble:
    """
    Logistic Regression meta-model over calibrated base estimator probabilities.
    This keeps the report's stacking layer explicit while preserving the same
    calibrated base-model interface used by soft voting.
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
            raise ValueError("LogisticStackingEnsemble must be fitted before inference.")
        return self.stacker.predict_proba(self._meta_features(X))[:, 1]

def optimize_decision_threshold(
    y_true: np.ndarray, 
    probs: np.ndarray, 
    recall_target: float = CLINICAL_RECALL_TARGET
) -> Tuple[float, float, float]:
    """
    Pareto-optimal Decision Threshold Search under Asymmetric Shift:
    Evaluates thresholds between 0.01 and 0.99.
    Selects the threshold maximizing Macro F1 while satisfying Clinical Sensitivity (Recall) >= target.
    
    Returns:
        best_threshold: Optimized decision boundary
        best_f1: Macro F1 achieved at best threshold
        achieved_recall: Recall achieved at best threshold
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_f1 = -1.0
    best_recall = -1.0
    
    # Store viable candidates satisfying clinical recall target
    viable_candidates = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        # Using zero_division=0 to prevent verbose warnings during boundary edge scanning
        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        
        if recall >= recall_target:
            viable_candidates.append((macro_f1, recall, t))
            
    if viable_candidates:
        # Sort by Macro F1 descending
        viable_candidates.sort(key=lambda x: x[0], reverse=True)
        best_f1, best_recall, best_thresh = viable_candidates[0]
    else:
        # Fallback: if no candidate hits the recall target safely, select threshold maximizing Macro F1 directly
        for t in thresholds:
            preds = (probs >= t).astype(int)
            macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_recall = recall
                best_thresh = t
                
    return float(best_thresh), float(best_f1), float(best_recall)
