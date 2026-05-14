import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Any, Optional
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer, SimpleImputer
from src.config import CAT_COLS, AA_COLS

class VariantFeatureEncoder:
    """
    Handles preprocessing of variant features:
    - Robust encoding of masked string/categorical variables (CAT_1..CAT_6, AA_1, AA_2).
    - Frequency encoding to map population source strings to informative numeric scores.
    - Handles unseen test-time categories safely.
    - Scales continuous variables and imputes missingness specifically for non-tree algorithms (e.g. Logistic Regression).
    """
    def __init__(
        self,
        rare_threshold: int = 3,
        low_missing_threshold: float = 0.30,
        high_missing_threshold: float = 0.60,
        drop_high_missing: bool = False,
        iterative_imputer_max_features: int = 40
    ):
        self.rare_threshold = rare_threshold
        self.low_missing_threshold = low_missing_threshold
        self.high_missing_threshold = high_missing_threshold
        self.drop_high_missing = drop_high_missing
        self.iterative_imputer_max_features = iterative_imputer_max_features
        self.cat_cols = CAT_COLS + AA_COLS
        self.freq_maps: Dict[str, Dict[str, float]] = {}
        self.label_maps: Dict[str, Dict[str, int]] = {}
        self.continuous_cols: List[str] = []
        self.feature_columns_: List[str] = []
        self.output_columns_tree_: List[str] = []
        self.output_columns_linear_: List[str] = []
        self.dropped_cols_: List[str] = []
        self.missing_indicator_cols_: List[str] = []
        self.missingness_report_: Dict[str, Dict[str, Any]] = {}
        self.scaler = StandardScaler()
        self.low_imputer = SimpleImputer(strategy="median")
        self.mid_imputer: Optional[IterativeImputer] = None
        self.mid_simple_imputer = SimpleImputer(strategy="median")
        self.use_iterative_mid_imputer_ = False
        self.high_imputer = SimpleImputer(strategy="median")
        self.low_continuous_cols_: List[str] = []
        self.mid_continuous_cols_: List[str] = []
        self.high_continuous_cols_: List[str] = []
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> "VariantFeatureEncoder":
        if X.empty:
            raise ValueError("Cannot fit encoder on an empty feature matrix.")

        self.feature_columns_ = list(X.columns)
        missing_rates = X.isna().mean()
        self.dropped_cols_ = [
            col for col, rate in missing_rates.items()
            if self.drop_high_missing and rate > self.high_missing_threshold
        ]
        self.missing_indicator_cols_ = [
            col for col, rate in missing_rates.items()
            if rate > 0 and col not in self.dropped_cols_
        ]
        self.missingness_report_ = {
            col: {
                "missing_rate": float(rate),
                "strategy": self._missing_strategy(col, float(rate))
            }
            for col, rate in missing_rates.items()
        }

        fit_df = self._replace_inf(X.drop(columns=self.dropped_cols_, errors="ignore"))

        # Identify continuous columns
        self.continuous_cols = [col for col in fit_df.columns if col not in self.cat_cols]
        
        # Fit Categorical encodings
        for col in self.cat_cols:
            if col in fit_df.columns:
                # Fill missing categories
                filled = fit_df[col].fillna("Missing").astype(str)
                
                # Compute frequencies
                counts = filled.value_counts()
                freqs = filled.value_counts(normalize=True).to_dict()
                
                # Identify rare categories
                rare_cats = set(counts[counts < self.rare_threshold].index)
                
                # Build frequency map
                self.freq_maps[col] = freqs
                
                # Build label integer map reserving 0 for Missing/Unknown
                unique_valid = [c for c in counts.index if c not in rare_cats and c != "Missing"]
                l_map = {"Missing": 0, "Rare/Unknown": 1}
                for idx, val in enumerate(unique_valid, start=2):
                    l_map[val] = idx
                self.label_maps[col] = l_map
                
        # Fit continuous imputers and scaler for linear meta-models.
        if self.continuous_cols:
            cont_missing = fit_df[self.continuous_cols].isna().mean()
            self.low_continuous_cols_ = [
                col for col, rate in cont_missing.items()
                if rate <= self.low_missing_threshold
            ]
            self.mid_continuous_cols_ = [
                col for col, rate in cont_missing.items()
                if self.low_missing_threshold < rate <= self.high_missing_threshold
            ]
            self.high_continuous_cols_ = [
                col for col, rate in cont_missing.items()
                if rate > self.high_missing_threshold
            ]

            if self.low_continuous_cols_:
                self.low_imputer.fit(fit_df[self.low_continuous_cols_])
            if self.mid_continuous_cols_:
                self.use_iterative_mid_imputer_ = len(self.mid_continuous_cols_) <= self.iterative_imputer_max_features
                if self.use_iterative_mid_imputer_:
                    self.mid_imputer = IterativeImputer(random_state=42, max_iter=5, sample_posterior=False)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        self.mid_imputer.fit(fit_df[self.mid_continuous_cols_])
                else:
                    self.mid_simple_imputer.fit(fit_df[self.mid_continuous_cols_])
            if self.high_continuous_cols_:
                self.high_imputer.fit(fit_df[self.high_continuous_cols_])

            if self.mid_continuous_cols_ and not self.use_iterative_mid_imputer_:
                for col in self.mid_continuous_cols_:
                    self.missingness_report_[col]["strategy"] = (
                        "median_impute_for_linear_due_high_dimensional_mid_missing__native_nan_for_tree"
                    )

            imputed_cont = self._impute_continuous_for_linear(fit_df.copy())[self.continuous_cols]
            self.scaler.fit(imputed_cont)
            
        self.is_fitted = True
        self.output_columns_tree_ = list(self._transform_internal(X, for_tree=True).columns)
        self.output_columns_linear_ = list(self._transform_internal(X, for_tree=False).columns)
        return self

    def _missing_strategy(self, col: str, missing_rate: float) -> str:
        if self.drop_high_missing and missing_rate > self.high_missing_threshold:
            return "drop_high_missing"
        if col in self.cat_cols:
            return "categorical_missing_token"
        if missing_rate == 0:
            return "none"
        if missing_rate <= self.low_missing_threshold:
            return "median_impute_for_linear__native_nan_for_tree"
        if missing_rate <= self.high_missing_threshold:
            return "iterative_impute_for_linear__native_nan_for_tree"
        return "median_impute_with_missing_indicator_for_linear__native_nan_for_tree"

    def _validate_schema(self, X: pd.DataFrame) -> None:
        incoming = set(X.columns)
        expected = set(self.feature_columns_)
        missing = sorted(expected - incoming)
        extra = sorted(incoming - expected)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing columns: {missing[:20]}")
            if extra:
                parts.append(f"extra columns: {extra[:20]}")
            raise ValueError("Input feature schema does not match the fitted encoder (" + "; ".join(parts) + ").")

    def _ensure_backward_compatibility(self, X: pd.DataFrame) -> None:
        """Migrates encoders serialized before schema-locking fields existed."""
        if not hasattr(self, "feature_columns_"):
            self.feature_columns_ = list(X.columns)
        if not hasattr(self, "output_columns_tree_"):
            self.output_columns_tree_ = []
        if not hasattr(self, "output_columns_linear_"):
            self.output_columns_linear_ = []
        if not hasattr(self, "dropped_cols_"):
            self.dropped_cols_ = []
        if not hasattr(self, "missing_indicator_cols_"):
            self.missing_indicator_cols_ = []
        if not hasattr(self, "missingness_report_"):
            self.missingness_report_ = {}
        if not hasattr(self, "low_continuous_cols_"):
            self.low_continuous_cols_ = list(getattr(self, "continuous_cols", []))
        if not hasattr(self, "mid_continuous_cols_"):
            self.mid_continuous_cols_ = []
        if not hasattr(self, "high_continuous_cols_"):
            self.high_continuous_cols_ = []
        if not hasattr(self, "low_imputer") and hasattr(self, "imputer"):
            self.low_imputer = self.imputer
        if not hasattr(self, "mid_simple_imputer"):
            self.mid_simple_imputer = SimpleImputer(strategy="median")
        if not hasattr(self, "high_imputer"):
            self.high_imputer = SimpleImputer(strategy="median")
        if not hasattr(self, "use_iterative_mid_imputer_"):
            self.use_iterative_mid_imputer_ = False

    def _replace_inf(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return out

    def _impute_continuous_for_linear(self, out_df: pd.DataFrame) -> pd.DataFrame:
        if self.low_continuous_cols_:
            out_df[self.low_continuous_cols_] = self.low_imputer.transform(out_df[self.low_continuous_cols_])
        if self.mid_continuous_cols_:
            if self.use_iterative_mid_imputer_:
                assert self.mid_imputer is not None
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out_df[self.mid_continuous_cols_] = self.mid_imputer.transform(out_df[self.mid_continuous_cols_])
            else:
                out_df[self.mid_continuous_cols_] = self.mid_simple_imputer.transform(out_df[self.mid_continuous_cols_])
        if self.high_continuous_cols_:
            out_df[self.high_continuous_cols_] = self.high_imputer.transform(out_df[self.high_continuous_cols_])
        return out_df
        
    def transform(
        self, 
        X: pd.DataFrame, 
        for_tree: bool = True
    ) -> pd.DataFrame:
        """
        Transforms features:
        - for_tree=True: Keeps raw continuous NaNs intact (LightGBM/XGBoost handle natively),
          maps strings to pandas category dtype or integer encoding. Appends frequency encodings.
        - for_tree=False: Completely imputes NaNs and standard scales continuous features.
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before calling transform.")
        self._ensure_backward_compatibility(X)
        self._validate_schema(X)
        return self._transform_internal(X, for_tree=for_tree)

    def _transform_internal(
        self,
        X: pd.DataFrame,
        for_tree: bool = True
    ) -> pd.DataFrame:
        out_df = self._replace_inf(X)
        out_df = out_df[self.feature_columns_]
        out_df = out_df.drop(columns=self.dropped_cols_, errors="ignore")

        indicator_data = {
            f"{col}_missing": out_df[col].isna().astype(int)
            for col in self.missing_indicator_cols_
            if col in out_df.columns
        }
        if indicator_data:
            out_df = pd.concat([out_df, pd.DataFrame(indicator_data, index=out_df.index)], axis=1)
        
        # 1. Process Categoricals
        freq_features: Dict[str, pd.Series] = {}
        for col in self.cat_cols:
            if col in out_df.columns:
                filled = out_df[col].fillna("Missing").astype(str)
                
                # Create frequency encoding features
                freq_map = self.freq_maps.get(col, {})
                # Unseen categories default to 0.0 frequency
                freq_features[f"{col}_freq"] = filled.map(freq_map).fillna(0.0).astype(float)
                
                # Map to target labels safely
                l_map = self.label_maps.get(col, {})
                
                def map_label(val: str) -> int:
                    if val == "Missing":
                        return 0
                    return l_map.get(val, 1)  # 1 is Rare/Unknown
                    
                mapped_series = filled.map(map_label).astype(int)
                
                if for_tree:
                    # Keep as integer or convert to pandas category for native splits
                    out_df[col] = mapped_series.astype("category")
                else:
                    out_df[col] = mapped_series.astype(float)

        if freq_features:
            out_df = pd.concat([out_df, pd.DataFrame(freq_features, index=out_df.index)], axis=1)
                    
        # 2. Process Continuous Features
        if not for_tree and self.continuous_cols:
            # Full imputation and standardization needed for linear models
            out_df = self._impute_continuous_for_linear(out_df)
            scaled = self.scaler.transform(out_df[self.continuous_cols])
            out_df[self.continuous_cols] = scaled

        expected_cols = self.output_columns_tree_ if for_tree else self.output_columns_linear_
        if expected_cols:
            missing_outputs = [col for col in expected_cols if col not in out_df.columns]
            if missing_outputs:
                raise ValueError(f"Encoder transform failed to create expected output columns: {missing_outputs[:20]}")
            out_df = out_df[expected_cols]

        return out_df

    def fit_transform(self, X: pd.DataFrame, for_tree: bool = True) -> pd.DataFrame:
        return self.fit(X).transform(X, for_tree=for_tree)
