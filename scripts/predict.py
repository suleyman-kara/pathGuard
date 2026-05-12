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
from src.models.ensemble import CalibratedVariantModel, SoftVotingEnsemble
from src.models.panel_model import PanelMetaLearner

def load_master_ensemble() -> Tuple[SoftVotingEnsemble, float]:
    """Loads the fully trained calibrated Soft-Voting Master Ensemble and optimal threshold."""
    lgbm = LightGBMVariantModel().load(str(MODEL_DIR / "master_lgbm.joblib"))
    xgb = XGBoostVariantModel().load(str(MODEL_DIR / "master_xgb.joblib"))
    
    cal_lgbm = CalibratedVariantModel(lgbm)
    cal_lgbm.calibrator = joblib.load(MODEL_DIR / "calibrator_lgbm.joblib")
    cal_lgbm.is_fitted = True
    
    cal_xgb = CalibratedVariantModel(xgb)
    cal_xgb.calibrator = joblib.load(MODEL_DIR / "calibrator_xgb.joblib")
    cal_xgb.is_fitted = True
    
    ensemble = SoftVotingEnsemble([cal_lgbm, cal_xgb], weights=[0.6, 0.4])
    meta = joblib.load(MODEL_DIR / "ensemble_meta.joblib")
    return ensemble, meta["threshold"]

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
    args = parser.parse_args()
    
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)
        
    print(f"Loading test data from {input_path}...")
    start_load = time.time()
    X_raw, _, variant_ids = load_data(str(input_path), is_test=True)
    
    print("Loading global Feature Encoder...")
    encoder = joblib.load(MODEL_DIR / "feature_encoder.joblib")
    X_prep = encoder.transform(X_raw, for_tree=True)
    
    print(f"Loading inference weights for context: {args.panel}...")
    master_ensemble, best_threshold = load_master_ensemble()
    
    start_inf = time.time()
    if args.panel == "MASTER":
        probs = master_ensemble.predict_proba(X_prep)
    else:
        panel_file = MODEL_DIR / f"panel_{args.panel}.joblib"
        if not panel_file.exists():
            print(f"Error: Specialized weights {panel_file} missing. Run training first.")
            sys.exit(1)
            
        learner = PanelMetaLearner(master_ensemble).load(str(panel_file))
        probs = learner.predict_proba(X_prep)
        
    preds = (probs >= best_threshold).astype(int)
    inf_time = time.time() - start_inf
    total_time = time.time() - start_load
    
    # Calculate per variant latency
    latency_ms = (inf_time / max(len(variant_ids), 1)) * 1000.0
    
    # Generate outputs
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
