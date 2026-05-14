import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="pathguard_matplotlib_"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix
)
from src.config import OUTPUT_DIR

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """Calculates the complete competition evaluation metric suite."""
    # Ensure safe zero divisions
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    class1_f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate PR-AUC safely
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    
    # Calculate ROC-AUC safely
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        roc_auc = 0.5  # Only one class present in small sample sets
        
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    
    # Specificity calculation from confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / max(tn + fp, 1)
    else:
        specificity = 0.0
        
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        brier = brier_score_loss(y_true, y_probs)
    except ValueError:
        brier = 0.0
    
    return {
        "Macro_F1": float(macro_f1),
        "Class1_F1": float(class1_f1),
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc_auc),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Balanced_Accuracy": float(bal_acc),
        "MCC": float(mcc),
        "Brier_Score": float(brier)
    }

def save_metrics_report(metrics: Dict[str, float], file_name: str = "evaluation_metrics.json") -> None:
    out_path = OUTPUT_DIR / file_name
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

def save_json_report(report: Dict[str, Any], file_name: str) -> None:
    out_path = OUTPUT_DIR / file_name
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)

def plot_precision_recall_curve(y_true: np.ndarray, y_probs: np.ndarray, file_name: str = "pr_curve.png") -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / file_name, dpi=300, bbox_inches="tight")
    plt.close()

def plot_reliability_diagram(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10, file_name: str = "calibration_curve.png") -> None:
    """Plots custom empirical calibration curve to verify post-calibration behavior."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.clip(np.digitize(y_probs, bins) - 1, 0, n_bins - 1)
    
    bin_sums = np.bincount(binids, weights=y_probs, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)
    
    nonzero = bin_total > 0
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label="Model empirical outputs")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / file_name, dpi=300, bbox_inches="tight")
    plt.close()

def save_error_analysis(
    variant_ids: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    file_name: str,
    max_rows: Optional[int] = None
) -> None:
    """Writes false-positive/false-negative rows for clinical review."""
    out_df = pd.DataFrame({
        "Variant_ID": variant_ids.reset_index(drop=True),
        "True_Label": y_true,
        "Prediction": y_pred,
        "Probability": y_probs
    })
    out_df["Error_Type"] = np.select(
        [
            (out_df["True_Label"] == 1) & (out_df["Prediction"] == 0),
            (out_df["True_Label"] == 0) & (out_df["Prediction"] == 1)
        ],
        ["FN", "FP"],
        default="Correct"
    )
    out_df = out_df[out_df["Error_Type"] != "Correct"].sort_values(
        by=["Error_Type", "Probability"],
        ascending=[True, False]
    )
    if max_rows is not None:
        out_df = out_df.head(max_rows)
    out_df.to_csv(OUTPUT_DIR / file_name, index=False)

def save_feature_importance(
    model: Any,
    feature_names: list[str],
    file_name: str,
    importance_type: str = "gain"
) -> None:
    """Saves LightGBM/XGBoost compatible feature importances when available."""
    raw_model = getattr(model, "model", model)
    if hasattr(raw_model, "booster_"):
        importances = raw_model.booster_.feature_importance(importance_type=importance_type)
    elif hasattr(raw_model, "feature_importances_"):
        importances = raw_model.feature_importances_
    else:
        return

    out_df = pd.DataFrame({
        "feature": feature_names,
        f"importance_{importance_type}": importances
    }).sort_values(f"importance_{importance_type}", ascending=False)
    out_df.to_csv(OUTPUT_DIR / file_name, index=False)

def save_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    file_name: str,
    n_repeats: int = 5
) -> None:
    """Computes a compact permutation importance report for feature-selection review."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model.model if hasattr(model, "model") else model,
        X,
        y,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=1
    )
    out_df = pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)
    out_df.to_csv(OUTPUT_DIR / file_name, index=False)
