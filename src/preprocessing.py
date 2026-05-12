import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.config import CAT_COLS, AA_COLS

class VariantFeatureEncoder:
    """
    Handles preprocessing of variant features:
    - Robust encoding of masked string/categorical variables (CAT_1..CAT_6, AA_1, AA_2).
    - Frequency encoding to map population source strings to informative numeric scores.
    - Handles unseen test-time categories safely.
    - Scales continuous variables and imputes missingness specifically for non-tree algorithms (e.g. Logistic Regression).
    """
    def __init__(self, rare_threshold: int = 3):
        self.rare_threshold = rare_threshold
        self.cat_cols = CAT_COLS + AA_COLS
        self.freq_maps: Dict[str, Dict[str, float]] = {}
        self.label_maps: Dict[str, Dict[str, int]] = {}
        self.continuous_cols: List[str] = []
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> "VariantFeatureEncoder":
        # Identify continuous columns
        self.continuous_cols = [col for col in X.columns if col not in self.cat_cols]
        
        # Fit Categorical encodings
        for col in self.cat_cols:
            if col in X.columns:
                # Fill missing categories
                filled = X[col].fillna("Missing").astype(str)
                
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
                
        # Fit continuous imputer and scaler for linear meta-models
        if self.continuous_cols:
            # Fit imputer using continuous columns
            imputed_cont = self.imputer.fit_transform(X[self.continuous_cols])
            self.scaler.fit(imputed_cont)
            
        self.is_fitted = True
        return self
        
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
            
        out_df = X.copy()
        
        # 1. Process Categoricals
        for col in self.cat_cols:
            if col in out_df.columns:
                filled = out_df[col].fillna("Missing").astype(str)
                
                # Create frequency encoding features
                freq_map = self.freq_maps.get(col, {})
                # Unseen categories default to 0.0 frequency
                out_df[f"{col}_freq"] = filled.map(freq_map).fillna(0.0).astype(float)
                
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
                    
        # 2. Process Continuous Features
        if not for_tree and self.continuous_cols:
            # Full imputation and standardization needed for linear models
            cont_data = out_df[self.continuous_cols]
            imputed = self.imputer.transform(cont_data)
            scaled = self.scaler.transform(imputed)
            out_df[self.continuous_cols] = scaled
            
        return out_df

    def fit_transform(self, X: pd.DataFrame, for_tree: bool = True) -> pd.DataFrame:
        return self.fit(X).transform(X, for_tree=for_tree)
