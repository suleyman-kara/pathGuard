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
    args = parser.parse_args()
    
    print(f"=== Initializing PathGuard Training Suite with {args.trials} Optuna Trials ===")
    pipeline = PathGuardTrainingPipeline(n_trials=args.trials)
    pipeline.execute_all()
    print("=== Training Complete ===")

if __name__ == "__main__":
    main()
