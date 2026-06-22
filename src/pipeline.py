import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import (
    CFTR_CSV,
    EXPERIMENT_LOG,
    KANSER_CSV,
    LGBM_PARAM_SPACE,
    MASTER_CSV,
    MODEL_DIR,
    OPTUNA_TRIALS,
    OUTPUT_DIR,
    PAH_CSV,
    PANEL_CV_REPEATS,
    RANDOM_SEED,
    XGB_PARAM_SPACE,
)
from src.data_loader import (
    build_data_quality_report,
    deduplicate_dataset,
    get_cv_splits,
    load_data,
)
from src.evaluation import (
    calculate_metrics,
    plot_precision_recall_curve,
    plot_reliability_diagram,
    save_error_analysis,
    save_feature_importance,
    save_json_report,
    save_metrics_report,
    save_permutation_importance,
)
from src.explainability import VariantExplainabilityEngine
from src.models.ensemble import (
    CalibratedVariantModel,
    LogisticStackingEnsemble,
    SoftVotingEnsemble,
    optimize_decision_threshold,
)
from src.models.lgbm_model import LightGBMVariantModel
from src.models.panel_model import PanelVariantModel
from src.models.xgb_model import XGBoostVariantModel
from src.preprocessing import VariantFeatureEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)


class PathGuardTrainingPipeline:
    """
    End-to-end PathGuard training pipeline aligned with docs/yarisma-raporu.md:
    data quality reporting, leakage-aware panel checks, calibrated GBDT ensemble,
    optional Logistic Regression stacking, OOF panel validation, feature importance,
    error analysis, and SHAP outputs.
    """

    def __init__(
        self,
        n_trials: int = OPTUNA_TRIALS,
        cv_repeats: int = PANEL_CV_REPEATS,
        calibration_mode: str = "oof",
        enable_stacking: bool = True,
        skip_shap: bool = False,
    ):
        if calibration_mode not in {"oof", "holdout"}:
            raise ValueError("calibration_mode must be either 'oof' or 'holdout'.")

        self.n_trials = n_trials
        self.cv_repeats = cv_repeats
        self.calibration_mode = calibration_mode
        self.enable_stacking = enable_stacking
        self.skip_shap = skip_shap
        self.encoder = VariantFeatureEncoder()
        self.master_ensemble: Optional[Any] = None
        self.best_threshold = 0.5
        self.ensemble_type = "soft_voting"
        self.soft_voting_weights = [0.6, 0.4]
        self.panel_learners: Dict[str, PanelVariantModel] = {}
        self.panel_thresholds: Dict[str, float] = {}
        self.data_quality_report: Dict[str, Any] = {}

    def _log_experiment(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": event,
            **payload,
        }
        with open(EXPERIMENT_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _panel_paths(self) -> Dict[str, str]:
        return {
            "KANSER": str(KANSER_CSV),
            "PAH": str(PAH_CSV),
            "CFTR": str(CFTR_CSV),
        }

    def _write_data_quality_report(self) -> None:
        print("Building data quality and leakage report...")
        self.data_quality_report = build_data_quality_report(str(MASTER_CSV), self._panel_paths())
        save_json_report(self.data_quality_report, "data_quality_report.json")
        self._log_experiment("data_quality_report", self.data_quality_report)

    def _optimize_lgbm_hyperparams(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
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
                "n_jobs": -1,
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
        with tqdm(total=self.n_trials, desc="Optuna Optimization", unit="trial") as pbar:
            def callback(study: optuna.Study, trial: optuna.Trial) -> None:
                pbar.update(1)
                pbar.set_postfix({"Best PR-AUC": f"{study.best_value:.4f}"})

            study.optimize(objective, n_trials=self.n_trials, callbacks=[callback])

        print(f"Optimization finished. Best PR-AUC: {study.best_value:.4f}")
        self._log_experiment("optuna_lgbm", {"best_value": float(study.best_value), "best_params": study.best_params})
        return study.best_params

    def _optimize_xgb_hyperparams(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """XGBoost için Optuna araması (LGBM ile aynı PR-AUC hedefi). Daha önce XGB default
        parametrelerle eğitiliyordu; ayarlanmış XGB master ensemble'ın çeşitliliğini ve
        sıralama (ranking) gücünü artırır."""
        print("Starting Optuna Hyperparameter Optimization for XGBoost on Master set...")
        cv_splits = get_cv_splits(y, is_small_panel=False)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", *XGB_PARAM_SPACE["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *XGB_PARAM_SPACE["learning_rate"], log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", *XGB_PARAM_SPACE["min_child_weight"]),
                "reg_lambda": trial.suggest_float("reg_lambda", *XGB_PARAM_SPACE["reg_lambda"]),
                "reg_alpha": trial.suggest_float("reg_alpha", *XGB_PARAM_SPACE["reg_alpha"]),
                "subsample": trial.suggest_float("subsample", *XGB_PARAM_SPACE["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *XGB_PARAM_SPACE["colsample_bytree"]),
            }

            oof_probs = np.zeros(len(y))
            for train_idx, val_idx in cv_splits:
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_v, y_v = X.iloc[val_idx], y.iloc[val_idx]

                model = XGBoostVariantModel(model_params=params)
                model.train(X_tr, y_tr, X_val=X_v, y_val=y_v)
                oof_probs[val_idx] = model.predict_proba(X_v)

            precision, recall, _ = precision_recall_curve(y, oof_probs)
            return float(auc(recall, precision))

        # XGB Optuna trial'ı LGBM'in yarısı (en az 1) ile sınırlanır: tam trial bütçesi master'a
        # yalnızca ~0.1pp katarken çalışma süresini ~3 dk artırıyordu (10 dk bütçesi). Kazancın
        # çoğu zaten eksiklik bayrakları + LGBM ayarından geliyor; XGB çeşitlilik için yeterli.
        xgb_trials = max(1, self.n_trials // 2)
        study = optuna.create_study(direction="maximize")
        with tqdm(total=xgb_trials, desc="Optuna XGB Optimization", unit="trial") as pbar:
            def callback(study: optuna.Study, trial: optuna.Trial) -> None:
                pbar.update(1)
                pbar.set_postfix({"Best PR-AUC": f"{study.best_value:.4f}"})

            study.optimize(objective, n_trials=xgb_trials, callbacks=[callback])

        print(f"XGB Optimization finished. Best PR-AUC: {study.best_value:.4f}")
        self._log_experiment("optuna_xgb", {"best_value": float(study.best_value), "best_params": study.best_params})
        return study.best_params

    def _fit_calibrators(
        self,
        X_prep: pd.DataFrame,
        y_clean: pd.Series,
        lgbm_oof: np.ndarray,
        xgb_oof: np.ndarray,
        final_lgbm: LightGBMVariantModel,
        final_xgb: XGBoostVariantModel,
        best_lgbm_params: Dict[str, Any],
        best_xgb_params: Dict[str, Any],
    ) -> Tuple[CalibratedVariantModel, CalibratedVariantModel]:
        cal_lgbm = CalibratedVariantModel(final_lgbm)
        cal_xgb = CalibratedVariantModel(final_xgb)

        if self.calibration_mode == "holdout":
            train_idx, cal_idx = train_test_split(
                np.arange(len(y_clean)),
                test_size=0.20,
                stratify=y_clean,
                random_state=RANDOM_SEED,
            )
            hold_lgbm = LightGBMVariantModel(model_params=best_lgbm_params).train(
                X_prep.iloc[train_idx],
                y_clean.iloc[train_idx],
                X_val=X_prep.iloc[cal_idx],
                y_val=y_clean.iloc[cal_idx],
            )
            hold_xgb = XGBoostVariantModel(model_params=best_xgb_params).train(
                X_prep.iloc[train_idx],
                y_clean.iloc[train_idx],
                X_val=X_prep.iloc[cal_idx],
                y_val=y_clean.iloc[cal_idx],
            )
            cal_lgbm.fit_calibration(hold_lgbm.predict_proba(X_prep.iloc[cal_idx]), y_clean.iloc[cal_idx].values)
            cal_xgb.fit_calibration(hold_xgb.predict_proba(X_prep.iloc[cal_idx]), y_clean.iloc[cal_idx].values)
        else:
            cal_lgbm.fit_calibration(lgbm_oof, y_clean.values)
            cal_xgb.fit_calibration(xgb_oof, y_clean.values)

        return cal_lgbm, cal_xgb

    def _select_master_ensemble(
        self,
        y_true: np.ndarray,
        lgbm_oof_cal: np.ndarray,
        xgb_oof_cal: np.ndarray,
        cal_lgbm: CalibratedVariantModel,
        cal_xgb: CalibratedVariantModel,
    ) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        # Soft-voting ağırlığını OOF üzerinde optimize et (sabit 0.6/0.4 yerine). Her aday
        # ağırlık için eşik yeniden optimize edilir; test-prior class 1 F1'ini maksimize eden
        # ağırlık seçilir. Kalibre olasılıkların ortalanması ranking'i değiştirdiği için bu
        # adım F1'e doğrudan etki edebilir (kalibrasyonun aksine — o monotoniktir, F1'i değiştirmez).
        w_lgbm = 0.6
        best_soft_score = -1.0
        for w in np.linspace(0.0, 1.0, 11):
            probs_w = w * lgbm_oof_cal + (1.0 - w) * xgb_oof_cal
            t_w, _, _ = optimize_decision_threshold(y_true, probs_w)
            score_w = calculate_metrics(y_true, (probs_w >= t_w).astype(int), probs_w)["Class1_F1_TestPrior"]
            if score_w > best_soft_score:
                best_soft_score = score_w
                w_lgbm = float(w)
        self.soft_voting_weights = [w_lgbm, 1.0 - w_lgbm]
        print(f"Optimized soft-voting weights (LGBM/XGB): {w_lgbm:.2f}/{1.0 - w_lgbm:.2f}")

        soft_oof = w_lgbm * lgbm_oof_cal + (1.0 - w_lgbm) * xgb_oof_cal
        candidates: Dict[str, Dict[str, Any]] = {}

        soft_t, _, _ = optimize_decision_threshold(y_true, soft_oof)
        candidates["soft_voting"] = {
            "ensemble": SoftVotingEnsemble([cal_lgbm, cal_xgb], weights=list(self.soft_voting_weights)),
            "probs": soft_oof,
            "threshold": soft_t,
            "metrics": calculate_metrics(y_true, (soft_oof >= soft_t).astype(int), soft_oof),
        }

        if self.enable_stacking:
            base_oof = np.column_stack([lgbm_oof_cal, xgb_oof_cal])
            stacker = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
            stacker.fit(base_oof, y_true)
            stack_oof = stacker.predict_proba(base_oof)[:, 1]
            stack_t, _, _ = optimize_decision_threshold(y_true, stack_oof)
            candidates["logistic_stacking"] = {
                "ensemble": LogisticStackingEnsemble([cal_lgbm, cal_xgb], stacker=stacker),
                "probs": stack_oof,
                "threshold": stack_t,
                "metrics": calculate_metrics(y_true, (stack_oof >= stack_t).astype(int), stack_oof),
                "stacker": stacker,
            }

        # Ensemble seçimi de test dağılımına göre yapılır (eğitim dağılımındaki F1 yanıltıcı)
        selected_name = max(candidates, key=lambda name: candidates[name]["metrics"]["Class1_F1_TestPrior"])
        selected = candidates[selected_name]
        self.ensemble_type = selected_name
        self.best_threshold = float(selected["threshold"])

        comparison = {
            name: {
                "threshold": float(candidate["threshold"]),
                "metrics": candidate["metrics"],
            }
            for name, candidate in candidates.items()
        }
        save_json_report(comparison, "master_ensemble_comparison.json")
        return selected["ensemble"], selected["probs"], selected

    def run_master_pipeline(self) -> Tuple[np.ndarray, np.ndarray]:
        print(f"Loading Master dataset from {MASTER_CSV}...")
        X_raw, y_raw, ids_raw = load_data(str(MASTER_CSV), is_test=False)
        assert y_raw is not None

        print("Deduplicating Master dataset...")
        X_clean, y_clean, ids_clean = deduplicate_dataset(X_raw, y_raw, ids_raw)

        print("Fitting Feature Encoder...")
        X_prep = self.encoder.fit_transform(X_clean, for_tree=True)

        best_lgbm_params: Dict[str, Any] = {}
        best_xgb_params: Dict[str, Any] = {}
        if self.n_trials > 0:
            best_lgbm_params = self._optimize_lgbm_hyperparams(X_prep, y_clean)
            best_xgb_params = self._optimize_xgb_hyperparams(X_prep, y_clean)

        print("Training final Master LightGBM & XGBoost estimators...")
        final_lgbm = LightGBMVariantModel(model_params=best_lgbm_params).train(X_prep, y_clean)
        final_xgb = XGBoostVariantModel(model_params=best_xgb_params).train(X_prep, y_clean)

        cv_splits = get_cv_splits(y_clean, is_small_panel=False)
        lgbm_oof = np.zeros(len(y_clean))
        xgb_oof = np.zeros(len(y_clean))

        print("Calculating OOF predictions for calibration and model selection...")
        for train_idx, val_idx in tqdm(cv_splits, desc="Master OOF CV", unit="fold"):
            X_tr, y_tr = X_prep.iloc[train_idx], y_clean.iloc[train_idx]
            X_v, y_v = X_prep.iloc[val_idx], y_clean.iloc[val_idx]

            m_lgb = LightGBMVariantModel(model_params=best_lgbm_params).train(X_tr, y_tr, X_val=X_v, y_val=y_v)
            lgbm_oof[val_idx] = m_lgb.predict_proba(X_v)

            m_xgb = XGBoostVariantModel(model_params=best_xgb_params).train(X_tr, y_tr, X_val=X_v, y_val=y_v)
            xgb_oof[val_idx] = m_xgb.predict_proba(X_v)

        print(f"Fitting calibrators using {self.calibration_mode} mode...")
        cal_lgbm, cal_xgb = self._fit_calibrators(
            X_prep,
            y_clean,
            lgbm_oof,
            xgb_oof,
            final_lgbm,
            final_xgb,
            best_lgbm_params,
            best_xgb_params,
        )
        lgbm_oof_cal = cal_lgbm.calibrator.transform(lgbm_oof)
        xgb_oof_cal = cal_xgb.calibrator.transform(xgb_oof)

        self.master_ensemble, master_oof_probs, selected = self._select_master_ensemble(
            y_clean.values,
            lgbm_oof_cal,
            xgb_oof_cal,
            cal_lgbm,
            cal_xgb,
        )
        master_preds = (master_oof_probs >= self.best_threshold).astype(int)
        master_metrics = calculate_metrics(y_clean.values, master_preds, master_oof_probs)
        master_metrics.update({
            "Validation_Mode": "OOF",
            "Selected_Ensemble": self.ensemble_type,
            "Calibration_Mode": self.calibration_mode,
        })

        print(
            f"Selected Master Ensemble: {self.ensemble_type} "
            f"(threshold={self.best_threshold:.3f}, OOF Class 1 F1={master_metrics['Class1_F1']:.4f}, "
            f"Expected Test Class 1 F1={master_metrics['Class1_F1_TestPrior']:.4f})"
        )
        save_metrics_report(master_metrics, file_name="master_metrics.json")
        plot_precision_recall_curve(y_clean.values, master_oof_probs, file_name="master_pr_curve.png")
        plot_reliability_diagram(y_clean.values, master_oof_probs, file_name="master_reliability.png")
        save_error_analysis(ids_clean, y_clean.values, master_preds, master_oof_probs, "error_analysis_master.csv")
        save_feature_importance(final_lgbm, list(X_prep.columns), "feature_importance_master_gain.csv", importance_type="gain")
        permutation_sample = X_prep.sample(n=min(len(X_prep), 400), random_state=RANDOM_SEED)
        save_permutation_importance(
            final_lgbm,
            permutation_sample,
            y_clean.loc[permutation_sample.index],
            "feature_importance_master_permutation.csv",
            n_repeats=3,
        )

        if not self.skip_shap:
            print("Running SHAP Explainability Engine for Master...")
            engine = VariantExplainabilityEngine(final_lgbm).fit(X_prep)
            groups = engine.perform_shap_clustering(X_prep)
            joblib.dump(groups, OUTPUT_DIR / "feature_biological_groups.joblib")
            engine.generate_summary_plot(file_name="master_shap_summary.png")
            engine.generate_waterfall_plot(sample_idx=0, file_name="master_shap_waterfall.png")

        joblib.dump(self.encoder, MODEL_DIR / "feature_encoder.joblib")
        joblib.dump(
            {
                "threshold": self.best_threshold,
                "ensemble_type": self.ensemble_type,
                "calibration_mode": self.calibration_mode,
                "enable_stacking": self.enable_stacking,
                "soft_voting_weights": list(self.soft_voting_weights),
            },
            MODEL_DIR / "ensemble_meta.joblib",
        )
        final_lgbm.save(str(MODEL_DIR / "master_lgbm.joblib"))
        final_xgb.save(str(MODEL_DIR / "master_xgb.joblib"))
        joblib.dump(cal_lgbm.calibrator, MODEL_DIR / "calibrator_lgbm.joblib")
        joblib.dump(cal_xgb.calibrator, MODEL_DIR / "calibrator_xgb.joblib")
        if self.enable_stacking and self.ensemble_type == "logistic_stacking":
            joblib.dump(selected["stacker"], MODEL_DIR / "stacker_logistic.joblib")

        self._log_experiment("master_training", master_metrics)
        return X_prep.values, y_clean.values

    def run_panel_pipelines(self) -> None:
        # Paneller artık master MODELİNE bağımlı değildir; yalnızca master setinde fit edilmiş
        # paylaşılan ön-işleme encoder'ına ihtiyaç duyarlar (etiket sızıntısı yok).
        assert self.encoder.is_fitted, "Feature encoder must be fitted (run master pipeline) before panels."

        overlap_report = self.data_quality_report.get("master_panel_overlap", {})
        for p_name, p_path in self._panel_paths().items():
            print(f"\n--- Running Independent Panel Pipeline for Panel: {p_name} ---")
            if not os.path.exists(p_path):
                print(f"Warning: Panel file {p_path} missing. Skipping.")
                continue

            X_raw, y_raw, ids_raw = load_data(str(p_path), is_test=False)
            assert y_raw is not None
            X_clean, y_clean, ids_clean = deduplicate_dataset(X_raw, y_raw, ids_raw)
            X_prep = self.encoder.transform(X_clean, for_tree=True)

            cv_splits = get_cv_splits(y_clean, is_small_panel=True, n_repeats=self.cv_repeats)
            lgb_sums = np.zeros(len(y_clean))
            xgb_sums = np.zeros(len(y_clean))
            prob_counts = np.zeros(len(y_clean))

            print(f"Calculating panel OOF predictions across {len(cv_splits)} folds/repeats...")
            for train_idx, val_idx in tqdm(cv_splits, desc=f"{p_name} OOF CV", unit="fold"):
                X_tr, y_tr = X_prep.iloc[train_idx], y_clean.iloc[train_idx]
                X_v = X_prep.iloc[val_idx]
                # Hem LGBM hem XGB üyesini eğit; ham (kalibre edilmemiş) OOF olasılıklarını
                # ayrı ayrı topla (kalibrasyon + ortalama OOF düzeyinde uygulanır).
                learner = PanelVariantModel(use_ensemble=True)
                learner.train(X_tr, y_tr)
                lgb_sums[val_idx] += learner.panel_model.predict_proba(X_v)
                xgb_sums[val_idx] += learner.xgb_model.predict_proba(X_v)
                prob_counts[val_idx] += 1

            if np.any(prob_counts == 0):
                raise ValueError(f"Panel CV failed to produce OOF predictions for every row: {p_name}")

            lgb_oof = lgb_sums / prob_counts
            xgb_oof = xgb_sums / prob_counts

            # Tek-LGBM baseline (mevcut yaklaşım)
            lgb_t, _, _ = optimize_decision_threshold(y_clean.values, lgb_oof)
            lgb_score = calculate_metrics(
                y_clean.values, (lgb_oof >= lgb_t).astype(int), lgb_oof
            )["Class1_F1_TestPrior"]

            # Ham (kalibrasyonsuz) 0.5/0.5 soft-voting ensemble adayı. Kalibrasyon kasıtlı
            # KULLANILMAZ: OOF üzerinde fit edilen izotonik kalibratörleri full-data modelin
            # olasılıklarına uygulamak, OOF'ta seçilen eşiğin inference'a transfer olmamasına
            # yol açıyordu (CFTR'de tüm tahminler 0 oluyordu). Ham ortalama tek-model gibi
            # transfer eder ve çeşitlilik kazancını (ensemble'ın asıl faydası) korur.
            ens_oof = 0.5 * lgb_oof + 0.5 * xgb_oof
            ens_t, _, _ = optimize_decision_threshold(y_clean.values, ens_oof)
            ens_score = calculate_metrics(
                y_clean.values, (ens_oof >= ens_t).astype(int), ens_oof
            )["Class1_F1_TestPrior"]

            # PANEL-BAŞINA GEÇİŞ (gate): ensemble yalnızca tek-LGBM'in OOF test-prior F1'ini
            # geçerse kullanılır. Küçük panellerde (örn. CFTR) ensemble fayda sağlamayabilir →
            # orada tek LGBM korunur.
            use_panel_ensemble = ens_score > lgb_score
            if use_panel_ensemble:
                oof_probs, panel_t = ens_oof, ens_t
            else:
                oof_probs, panel_t = lgb_oof, lgb_t
            print(
                f"Panel {p_name} gate: single-LGBM F1={lgb_score:.4f} vs "
                f"ensemble F1={ens_score:.4f} -> "
                f"{'ENSEMBLE' if use_panel_ensemble else 'SINGLE-LGBM'}"
            )

            # Her panel için bağımsız karar eşiği (seçilen adaya göre)
            self.panel_thresholds[p_name] = panel_t

            preds = (oof_probs >= panel_t).astype(int)
            metrics = calculate_metrics(y_clean.values, preds, oof_probs)
            metrics.update({
                "Validation_Mode": "RepeatedStratifiedKFold_OOF",
                "CV_Repeats": int(self.cv_repeats),
                "OOF_Prediction_Count": int(len(oof_probs)),
                "Panel_Model_Type": "lgbm_xgb_ensemble" if use_panel_ensemble else "single_lgbm",
                "Gate_SingleLGBM_F1_TestPrior": float(lgb_score),
                "Gate_Ensemble_F1_TestPrior": float(ens_score),
                "Leakage_Aware": bool(overlap_report.get(p_name, {}).get("leakage_warning", False)),
                "Master_Panel_Overlap_Count": int(overlap_report.get(p_name, {}).get("overlap_count", 0)),
                "Master_Panel_Overlap_Ratio": float(overlap_report.get(p_name, {}).get("overlap_ratio", 0.0)),
            })

            save_metrics_report(metrics, file_name=f"panel_{p_name}_metrics.json")
            plot_precision_recall_curve(y_clean.values, oof_probs, file_name=f"panel_{p_name}_pr_curve.png")
            plot_reliability_diagram(y_clean.values, oof_probs, file_name=f"panel_{p_name}_reliability.png")
            save_error_analysis(ids_clean, y_clean.values, preds, oof_probs, f"error_analysis_panel_{p_name}.csv")

            # Final panel modeli: gate kararına göre ham (kalibrasyonsuz) ensemble veya tek-LGBM.
            if use_panel_ensemble:
                final_learner = PanelVariantModel(use_ensemble=True)
                final_learner.train(X_prep, y_clean)
            else:
                final_learner = PanelVariantModel(use_ensemble=False)
                final_learner.train(X_prep, y_clean)
            self.panel_learners[p_name] = final_learner
            final_learner.save(str(MODEL_DIR / f"panel_{p_name}.joblib"))
            save_feature_importance(
                final_learner.panel_model,
                list(X_prep.columns),
                f"feature_importance_panel_{p_name}_gain.csv",
                importance_type="gain",
            )

            if not self.skip_shap:
                print(f"Running SHAP Explainability Engine for Panel {p_name}...")
                engine = VariantExplainabilityEngine(final_learner.panel_model).fit(X_prep)
                engine.generate_summary_plot(file_name=f"panel_{p_name}_shap_summary.png")
                for idx in np.where(preds != y_clean.values)[0][:5]:
                    engine.generate_waterfall_plot(sample_idx=int(idx), file_name=f"panel_{p_name}_shap_waterfall_{idx}.png")

            print(
                f"Panel {p_name} OOF Class 1 F1: {metrics['Class1_F1']:.4f}, "
                f"Expected Test Class 1 F1: {metrics['Class1_F1_TestPrior']:.4f}, "
                f"Recall: {metrics['Sensitivity']:.4f}, Leakage-aware: {metrics['Leakage_Aware']}"
            )
            self._log_experiment(f"panel_{p_name}_training", metrics)

        # Panel eşik değerlerini ensemble_meta.joblib dosyasına ekle
        meta_path = MODEL_DIR / "ensemble_meta.joblib"
        if meta_path.exists():
            meta = joblib.load(meta_path)
        else:
            meta = {}
        meta["panel_thresholds"] = self.panel_thresholds
        joblib.dump(meta, meta_path)

    def execute_all(self) -> None:
        self._write_data_quality_report()
        self.run_master_pipeline()
        self.run_panel_pipelines()
        print("\nAll pipeline executions and serialization completed successfully.")
