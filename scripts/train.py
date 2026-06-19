#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

# Add project root to path to ensure absolute module resolution works perfectly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline import PathGuardTrainingPipeline
from src.config import OPTUNA_TRIALS

def main() -> None:
    parser = argparse.ArgumentParser(description="PathGuard Model Training Execution Interface")
    parser.add_argument(
        "--trials", 
        type=int, 
        default=OPTUNA_TRIALS, 
        help="Number of Optuna Bayesian optimization trials to run on Master set."
    )
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=10,
        help="Number of repeated stratified CV passes for small panel OOF validation."
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["oof", "holdout"],
        default="oof",
        help="Calibration strategy: OOF is default; holdout uses a 20%% calibration split."
    )
    parser.add_argument(
        "--enable-stacking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Logistic Regression stacking over calibrated base model probabilities."
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP plots for faster smoke-test training runs."
    )
    args = parser.parse_args()
    
    print(f"=== Initializing PathGuard Training Suite with {args.trials} Optuna Trials ===")
    pipeline = PathGuardTrainingPipeline(
        n_trials=args.trials,
        cv_repeats=args.cv_repeats,
        calibration_mode=args.calibration_mode,
        enable_stacking=args.enable_stacking,
        skip_shap=args.skip_shap,
    )
    pipeline.execute_all()
    print("=== Training Complete ===")

if __name__ == "__main__":
    main()
