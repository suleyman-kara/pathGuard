import os
import joblib
import pandas as pd
import numpy as np
import optuna
from typing import Dict, Any, Tuple
from sklearn.metrics import precision_recall_curve, auc

from src.config import (
    MASTER_CSV,
    KANSER_CSV,
    PAH_CSV,
    CFTR_CSV,
    MODEL_DIR,
    OPTUNA_TRIALS,
    LGBM_PARAM_SPACE,
    OUTPUT_DIR
)
from src.data_loader import load_data, deduplicate_dataset, get_cv_splits
from src.preprocessing import VariantFeatureEncoder
from src.models.lgbm_model import LightGBMVariantModel
from src.models.xgb_model import XGBoostVariantModel
from src.models.ensemble import CalibratedVariantModel, SoftVotingEnsemble, optimize_decision_threshold
from src.models.panel_model import PanelMetaLearner
from src.evaluation import calculate_metrics, save_metrics_report, plot_precision_recall_curve, plot_reliability_diagram
from src.explainability import VariantExplainabilityEngine

optuna.logging.set_verbosity(optuna.logging.WARNING)

class PathGuardTrainingPipeline:
    """
    End-to-end orchestration pipeline for PathGuard model compilation:
    1. Loads, cleans, and deduplicates high-dimensional variant tables.
    2. Runs Bayesian hyperparameter optimization via Optuna.
    3. Fits primary LightGBM and secondary XGBoost models.
    4. Applies Isotonic Regression probability calibration and threshold tuning.
    5. Executes Panel-Aware Meta-Learning for hereditary cancer, PAH, and CFTR sets.
    6. Serializes all artifacts and generates full explainability graphics.
    """
    def __init__(self, n_trials: int = OPTUNA_TRIALS):
        self.n_trials = n_trials
        self.encoder = VariantFeatureEncoder()
        self.master_ensemble: Optional[SoftVotingEnsemble] = None
        self.best_threshold = 0.5
        self.panel_learners: Dict[str, PanelMetaLearner] = {}
        
    def _optimize_lgbm_hyperparams(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Runs fast Bayesian Optimization to find optimal LightGBM configurations."""
        print("Starting Optuna Hyperparameter Optimization on Master set...")
        cv_splits = get_cv_splits(y, is_small_panel=False)
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", *LGBM_PARAM_SPACE["num_leaves"]),
                "learning_rate": trial.suggest_float("learning_rate", *LGBM_PARAM_SPACE["learning_rate"], log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", *LGBM_PARAM_SPACE["min_child_samples"]),
                "reg_lambda": trial.suggest_float("reg_lambda", *LGBM_PARAM_SPACE["reg_lambda"]),
                "reg_alpha": trial.suggest_float("reg_alpha", *LGBM_PARAM_SPACE["reg_alpha"]),
                "subsample": trial.suggest_float("subsample", *LGBM_PARAM_SPACE["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *LGBM_PARAM_SPACE["colsample_bytree"]),
                "verbose": -1,
                "n_jobs": -1
            }
            
            oof_probs = np.zeros(len(y))
            for train_idx, val_idx in cv_splits:
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_v, y_v = X.iloc[val_idx], y.iloc[val_idx]
                
                model = LightGBMVariantModel(model_params=params)
                model.train(X_tr, y_tr, X_val=X_v, y_val=y_v)
                oof_probs[val_idx] = model.predict_proba(X_v)
                
            precision, recall, _ = precision_recall_curve(y, oof_probs)
            return float(auc(recall, precision))
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        print(f"Optimization finished. Best PR-AUC: {study.best_value:.4f}")
        return study.best_params

    def run_master_pipeline(self) -> Tuple[np.ndarray, np.ndarray]:
        """Trains Master models and builds Calibrated Soft-Voting Ensemble."""
        print(f"Loading Master dataset from {MASTER_CSV}...")
        X_raw, y_raw, ids_raw = load_data(str(MASTER_CSV), is_test=False)
        assert y_raw is not None
        
        print("Deduplicating Master dataset...")
        X_clean, y_clean, ids_clean = deduplicate_dataset(X_raw, y_raw, ids_raw)
        
        print("Fitting Feature Encoder...")
        X_prep = self.encoder.fit_transform(X_clean, for_tree=True)
        
        # 1. Hyperparameter tuning if n_trials > 0
        best_lgbm_params = {}
        if self.n_trials > 0:
            best_lgbm_params = self._optimize_lgbm_hyperparams(X_prep, y_clean)
            
        print("Training Final Master LightGBM & XGBoost Estimators via Stratified CV...")
        cv_splits = get_cv_splits(y_clean, is_small_panel=False)
        
        lgbm_oof = np.zeros(len(y_clean))
        xgb_oof = np.zeros(len(y_clean))
        
        # Fit final global instances on entire clean dataset for ultimate inference power
        final_lgbm = LightGBMVariantModel(model_params=best_lgbm_params)
        final_lgbm.train(X_prep, y_clean)
        
        final_xgb = XGBoostVariantModel()
        final_xgb.train(X_prep, y_clean)
        
        # Get Out-of-fold predictions to train honest Calibrators
        for train_idx, val_idx in cv_splits:
            X_tr, y_tr = X_prep.iloc[train_idx], y_clean.iloc[train_idx]
            X_v, y_v = X_prep.iloc[val_idx], y_clean.iloc[val_idx]
            
            m_lgb = LightGBMVariantModel(model_params=best_lgbm_params).train(X_tr, y_tr, X_val=X_v, y_val=y_v)
            lgbm_oof[val_idx] = m_lgb.predict_proba(X_v)
            
            m_xgb = XGBoostVariantModel().train(X_tr, y_tr, X_val=X_v, y_val=y_v)
            xgb_oof[val_idx] = m_xgb.predict_proba(X_v)
            
        print("Fitting Isotonic Regression Calibrators...")
        cal_lgbm = CalibratedVariantModel(final_lgbm).fit_calibration(lgbm_oof, y_clean.values)
        cal_xgb = CalibratedVariantModel(final_xgb).fit_calibration(xgb_oof, y_clean.values)
        
        self.master_ensemble = SoftVotingEnsemble([cal_lgbm, cal_xgb], weights=[0.6, 0.4])
        
        # Calculate ensemble out-of-fold calibrated probabilities to tune threshold safely
        # To avoid nested CV leakage, map base oof via fitted calibrators
        lgbm_oof_cal = cal_lgbm.calibrator.transform(lgbm_oof)
        xgb_oof_cal = cal_xgb.calibrator.transform(xgb_oof)
        ensemble_oof_cal = 0.6 * lgbm_oof_cal + 0.4 * xgb_oof_cal
        
        print("Optimizing Decision Threshold under Asymmetric Shift constraints...")
        best_t, best_f1, best_rec = optimize_decision_threshold(y_clean.values, ensemble_oof_cal)
        self.best_threshold = best_t
        print(f"Optimal Master Threshold: {best_t:.3f} (OOF Macro F1: {best_f1:.4f}, Recall: {best_rec:.4f})")
        
        # Evaluate & Save Master metrics
        metrics = calculate_metrics(y_clean.values, (ensemble_oof_cal >= best_t).astype(int), ensemble_oof_cal)
        save_metrics_report(metrics, file_name="master_metrics.json")
        plot_precision_recall_curve(y_clean.values, ensemble_oof_cal, file_name="master_pr_curve.png")
        plot_reliability_diagram(y_clean.values, ensemble_oof_cal, file_name="master_reliability.png")
        
        # Explainability reporting
        print("Running SHAP Explainability Engine...")
        engine = VariantExplainabilityEngine(final_lgbm).fit(X_prep)
        groups = engine.perform_shap_clustering(X_prep)
        joblib.dump(groups, OUTPUT_DIR / "feature_biological_groups.joblib")
        engine.generate_summary_plot(file_name="master_shap_summary.png")
        engine.generate_waterfall_plot(sample_idx=0, file_name="master_shap_waterfall.png")
        
        # Save master objects
        joblib.dump(self.encoder, MODEL_DIR / "feature_encoder.joblib")
        joblib.dump({"threshold": self.best_threshold}, MODEL_DIR / "ensemble_meta.joblib")
        final_lgbm.save(str(MODEL_DIR / "master_lgbm.joblib"))
        final_xgb.save(str(MODEL_DIR / "master_xgb.joblib"))
        joblib.dump(cal_lgbm.calibrator, MODEL_DIR / "calibrator_lgbm.joblib")
        joblib.dump(cal_xgb.calibrator, MODEL_DIR / "calibrator_xgb.joblib")
        
        return X_prep.values, y_clean.values

    def run_panel_pipelines(self) -> None:
        """Executes Panel-Aware Meta-Learning loops for small panel subsets."""
        panels = {
            "KANSER": KANSER_CSV,
            "PAH": PAH_CSV,
            "CFTR": CFTR_CSV
        }
        
        assert self.master_ensemble is not None, "Master pipeline must run before panels."
        
        for p_name, p_path in panels.items():
            print(f"\n--- Running Transfer Learning Pipeline for Panel: {p_name} ---")
            if not os.path.exists(p_path):
                print(f"Warning: Panel file {p_path} missing. Skipping.")
                continue
                
            X_raw, y_raw, ids_raw = load_data(str(p_path), is_test=False)
            assert y_raw is not None
            X_clean, y_clean, ids_clean = deduplicate_dataset(X_raw, y_raw, ids_raw)
            
            # Transform using fitted master encoder
            X_prep = self.encoder.transform(X_clean, for_tree=True)
            
            learner = PanelMetaLearner(self.master_ensemble)
            learner.train(X_prep, y_clean)
            self.panel_learners[p_name] = learner
            
            # Save panel model weights
            learner.save(str(MODEL_DIR / f"panel_{p_name}.joblib"))
            
            # Predict & Evaluate panel specific performance
            probs = learner.predict_proba(X_prep)
            preds = (probs >= self.best_threshold).astype(int)
            metrics = calculate_metrics(y_clean.values, preds, probs)
            
            save_metrics_report(metrics, file_name=f"panel_{p_name}_metrics.json")
            plot_precision_recall_curve(y_clean.values, probs, file_name=f"panel_{p_name}_pr_curve.png")
            print(f"Panel {p_name} final Macro F1: {metrics['Macro_F1']:.4f}, Recall: {metrics['Sensitivity']:.4f}")

    def execute_all(self) -> None:
        self.run_master_pipeline()
        self.run_panel_pipelines()
        print("\nAll pipeline executions and serialization completed successfully.")
