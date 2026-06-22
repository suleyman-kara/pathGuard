#!/usr/bin/env python
import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_DIR, OUTPUT_DIR
from src.data_loader import load_data
from src.models.lgbm_model import LightGBMVariantModel
from src.models.xgb_model import XGBoostVariantModel
from src.models.ensemble import CalibratedVariantModel, LogisticStackingEnsemble, SoftVotingEnsemble
from src.models.panel_model import PanelVariantModel

def load_master_ensemble() -> Tuple[object, dict]:
    """Loads the trained calibrated master ensemble and optimal threshold metadata."""
    required = [
        MODEL_DIR / "master_lgbm.joblib",
        MODEL_DIR / "master_xgb.joblib",
        MODEL_DIR / "calibrator_lgbm.joblib",
        MODEL_DIR / "calibrator_xgb.joblib",
        MODEL_DIR / "ensemble_meta.joblib",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trained artifacts. Run training first. Missing: {missing}")

    lgbm = LightGBMVariantModel().load(str(MODEL_DIR / "master_lgbm.joblib"))
    xgb = XGBoostVariantModel().load(str(MODEL_DIR / "master_xgb.joblib"))
    
    cal_lgbm = CalibratedVariantModel(lgbm)
    cal_lgbm.calibrator = joblib.load(MODEL_DIR / "calibrator_lgbm.joblib")
    cal_lgbm.is_fitted = True
    
    cal_xgb = CalibratedVariantModel(xgb)
    cal_xgb.calibrator = joblib.load(MODEL_DIR / "calibrator_xgb.joblib")
    cal_xgb.is_fitted = True
    
    meta = joblib.load(MODEL_DIR / "ensemble_meta.joblib")
    ensemble_type = meta.get("ensemble_type", "soft_voting")
    if ensemble_type == "logistic_stacking":
        stacker_path = MODEL_DIR / "stacker_logistic.joblib"
        if not stacker_path.exists():
            raise FileNotFoundError(f"Missing stacking artifact: {stacker_path}")
        ensemble = LogisticStackingEnsemble([cal_lgbm, cal_xgb], stacker=joblib.load(stacker_path))
    else:
        # Eğitimde OOF üzerinde optimize edilen ağırlıkları kullan (yoksa eski 0.6/0.4'e düş)
        weights = meta.get("soft_voting_weights", [0.6, 0.4])
        ensemble = SoftVotingEnsemble([cal_lgbm, cal_xgb], weights=list(weights))
    return ensemble, meta

def main() -> None:
    parser = argparse.ArgumentParser(description="PathGuard Fast Inference Interface")
    parser.add_argument("input_csv", type=str, help="Path to input unseen test CSV file.")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output predictions filename.")
    parser.add_argument(
        "--panel", 
        type=str, 
        default="MASTER", 
        choices=["MASTER", "KANSER", "PAH", "CFTR"],
        help="Specify context to load specialized Transfer Learning models."
    )
    parser.add_argument(
        "--submission-only",
        action="store_true",
        help="Output only Variant_ID and Prediction columns for Teknofest submission formatting."
    )
    args = parser.parse_args()
    
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)
        
    print(f"Loading test data from {input_path}...")
    start_load = time.time()
    X_raw, _, variant_ids = load_data(str(input_path), is_test=True)
    
    print("Loading global Feature Encoder...")
    encoder_path = MODEL_DIR / "feature_encoder.joblib"
    if not encoder_path.exists():
        print(f"Error: Missing encoder artifact {encoder_path}. Run training first.")
        sys.exit(1)
    encoder = joblib.load(encoder_path)
    try:
        X_prep = encoder.transform(X_raw, for_tree=True)
    except ValueError as exc:
        print(f"Error: Input schema validation failed. {exc}")
        sys.exit(1)
    
    print(f"Loading inference weights for context: {args.panel}...")
    if args.panel == "MASTER":
        try:
            master_ensemble, meta = load_master_ensemble()
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        best_threshold = meta.get("threshold", 0.5)
        start_inf = time.time()
        probs = master_ensemble.predict_proba(X_prep)
    else:
        # Bağımsız panel modeli: master ağırlıklarına ihtiyaç yoktur.
        # Yalnızca panel modeli + ensemble_meta.joblib içindeki panel eşiği gerekir.
        meta_path = MODEL_DIR / "ensemble_meta.joblib"
        if not meta_path.exists():
            print(f"Error: Missing metadata artifact {meta_path}. Run training first.")
            sys.exit(1)
        meta = joblib.load(meta_path)

        # Alt panele ait karar eşiğini yükle, yoksa genel eşiği kullan
        panel_thresholds = meta.get("panel_thresholds", {})
        best_threshold = panel_thresholds.get(args.panel, meta.get("threshold", 0.5))

        panel_file = MODEL_DIR / f"panel_{args.panel}.joblib"
        if not panel_file.exists():
            print(f"Error: Specialized weights {panel_file} missing. Run training first.")
            sys.exit(1)

        learner = PanelVariantModel().load(str(panel_file))
        start_inf = time.time()
        probs = learner.predict_proba(X_prep)
        
    preds = (probs >= best_threshold).astype(int)
    inf_time = time.time() - start_inf
    total_time = time.time() - start_load
    
    # Calculate per variant latency
    latency_ms = (inf_time / max(len(variant_ids), 1)) * 1000.0
    
    # Generate outputs
    if args.submission_only:
        out_df = pd.DataFrame({
            "Variant_ID": variant_ids,
            "Prediction": preds
        })
    else:
        out_df = pd.DataFrame({
            "Variant_ID": variant_ids,
            "Probability": probs,
            "Prediction": preds
        })
    
    out_file = OUTPUT_DIR / args.output
    out_df.to_csv(out_file, index=False)
    
    print(f"\n=== Inference Finished Successfully ===")
    print(f"Processed {len(variant_ids)} variants in {inf_time:.4f}s pure inference time.")
    print(f"Per-Variant Latency: {latency_ms:.3f} ms (Target: < 5 ms)")
    print(f"Total end-to-end execution time: {total_time:.4f}s")
    print(f"Predictions saved to {out_file}")

if __name__ == "__main__":
    main()
