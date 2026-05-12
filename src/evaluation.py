import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from sklearn.metrics import (
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
    
    return {
        "Macro_F1": float(macro_f1),
        "Class1_F1": float(class1_f1),
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc_auc),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Balanced_Accuracy": float(bal_acc),
        "MCC": float(mcc)
    }

def save_metrics_report(metrics: Dict[str, float], file_name: str = "evaluation_metrics.json") -> None:
    out_path = OUTPUT_DIR / file_name
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

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
    binids = np.digitize(y_probs, bins) - 1
    
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
