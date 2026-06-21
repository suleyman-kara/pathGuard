import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="pathguard_matplotlib_"))

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXPERIMENT_LOG = OUTPUT_DIR / "experiment_log.jsonl"

# Ensure output directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datasets
MASTER_CSV = RAW_DATA_DIR / "YARISMA_TRAIN_MASTER.csv"
KANSER_CSV = RAW_DATA_DIR / "YARISMA_TRAIN_KANSER.csv"
PAH_CSV = RAW_DATA_DIR / "YARISMA_TRAIN_PAH.csv"
CFTR_CSV = RAW_DATA_DIR / "YARISMA_TRAIN_CFTR.csv"

# Global Constants
RANDOM_SEED = 42
TARGET_COL = "Label"
ID_COL = "Variant_ID"

# Categorical and string columns mapped from raw CSV headers
CAT_COLS = ["CAT_1", "CAT_2", "CAT_3", "CAT_4", "CAT_5", "CAT_6"]
AA_COLS = ["AA_1", "AA_2"]

# Cross Validation Parameters
CV_FOLDS = 5
PANEL_CV_REPEATS = 10
EARLY_STOPPING_ROUNDS = 50
OPTUNA_TRIALS = 30  # Optimized for 10-minute compile/run budget on CPU

# LightGBM Search Space Config
LGBM_PARAM_SPACE = {
    "num_leaves": (15, 255),
    "learning_rate": (0.01, 0.3),
    "min_child_samples": (10, 100),
    "reg_lambda": (0.0, 5.0),
    "reg_alpha": (0.0, 5.0),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0)
}

# XGBoost Search Space Config
XGB_PARAM_SPACE = {
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "min_child_weight": (1, 10),
    "reg_lambda": (0.0, 5.0),
    "reg_alpha": (0.0, 5.0),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0)
}

# Decision Threshold Parameters
# Recall (sensitivity) constraint for threshold selection. Set to 0.0 so that
# optimize_decision_threshold directly maximizes the pathogenic-class (class 1) F1
# score — the sole competition ranking metric — without any clinical recall floor.
# The previous 0.90 clinical-recall constraint (reported as an "originality") was
# dropped because it capped class 1 F1; see docs/rapor_guncellemeleri.md.
CLINICAL_RECALL_TARGET = 0.0

# Expected pathogenic (class 1) prior of the HIDDEN test set. Per soru-cevap.md the
# training set is ~80% pathogenic while the test set is reversed to ~20% pathogenic
# / ~80% benign. Recall (TPR) and FPR estimated on the training-distribution OOF are
# prior-invariant, so we re-derive precision/F1 at this test prior to (a) select the
# decision threshold against the real evaluation distribution and (b) report an
# honest "expected test" class 1 F1. See docs/rapor_guncellemeleri.md.
TEST_PATHOGENIC_PRIOR = 0.20
