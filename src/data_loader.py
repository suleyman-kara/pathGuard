import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from src.config import ID_COL, TARGET_COL, RANDOM_SEED, CV_FOLDS, PANEL_CV_REPEATS

def load_data(file_path: str, is_test: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
    """
    Loads dataset from CSV file.
    
    Returns:
        X: Feature DataFrame
        y: Target Series (None if is_test=True)
        variant_ids: Variant identifier Series
    """
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {file_path}")
    
    # Extract variant IDs
    if ID_COL in df.columns:
        variant_ids = df[ID_COL]
        df = df.drop(columns=[ID_COL])
    else:
        if is_test:
            variant_ids = pd.Series([f"VAR_{i}" for i in range(len(df))], name=ID_COL)
        else:
            raise ValueError(f"Training dataset must include '{ID_COL}': {file_path}")
        
    # Extract labels
    if not is_test and TARGET_COL in df.columns:
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
        if y.isna().any():
            raise ValueError(f"Target column '{TARGET_COL}' contains missing values: {file_path}")
        if y.nunique(dropna=False) < 2:
            raise ValueError(f"Training dataset must contain at least two target classes: {file_path}")
        unexpected_labels = sorted(set(y.dropna().unique()) - {0, 1})
        if unexpected_labels:
            raise ValueError(f"Target column '{TARGET_COL}' must be binary 0/1. Unexpected labels: {unexpected_labels}")
    else:
        if not is_test:
            raise ValueError(f"Training dataset must include target column '{TARGET_COL}': {file_path}")
        y = None
        X = df
        if TARGET_COL in X.columns:
            X = X.drop(columns=[TARGET_COL])

    if X.empty or X.shape[1] == 0:
        raise ValueError(f"Dataset contains no feature columns after removing id/target: {file_path}")

    return X, y, variant_ids

def get_duplicate_profile(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> Dict[str, int]:
    """Summarizes exact feature-vector duplication and contradictory labels."""
    if X.empty:
        return {
            "duplicate_rows": 0,
            "duplicate_groups": 0,
            "contradictory_duplicate_groups": 0,
            "contradictory_duplicate_rows": 0
        }

    feature_cols = list(X.columns)
    df_filled = X[feature_cols].fillna("MISSING_DUMMY")
    duplicated_mask = df_filled.duplicated(keep=False)
    duplicate_rows = int(duplicated_mask.sum())
    if duplicate_rows == 0:
        return {
            "duplicate_rows": 0,
            "duplicate_groups": 0,
            "contradictory_duplicate_groups": 0,
            "contradictory_duplicate_rows": 0
        }

    duplicates = df_filled[duplicated_mask].copy()
    duplicate_groups = 0
    contradictory_groups = 0
    contradictory_rows = 0
    grouped = duplicates.groupby(feature_cols, dropna=False, sort=False)
    for idx, group in grouped:
        duplicate_groups += 1
        if y is not None:
            labels = y.loc[group.index].unique()
            if len(labels) > 1:
                contradictory_groups += 1
                contradictory_rows += len(group)

    return {
        "duplicate_rows": duplicate_rows,
        "duplicate_groups": int(duplicate_groups),
        "contradictory_duplicate_groups": int(contradictory_groups),
        "contradictory_duplicate_rows": int(contradictory_rows)
    }

def profile_dataset(file_path: str, is_test: bool = False) -> Dict[str, Any]:
    """Creates the competition-facing data quality profile for one CSV."""
    X, y, variant_ids = load_data(file_path, is_test=is_test)
    duplicate_profile = get_duplicate_profile(X, y)
    missing_rates = X.isna().mean().sort_values(ascending=False)
    top_missing = {
        col: float(rate)
        for col, rate in missing_rates.head(25).items()
        if rate > 0
    }

    report: Dict[str, Any] = {
        "file": str(file_path),
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "variant_id_unique": bool(variant_ids.is_unique),
        "missing_value_cells": int(X.isna().sum().sum()),
        "columns_with_missing": int((X.isna().mean() > 0).sum()),
        "top_missing_rates": top_missing,
        **duplicate_profile
    }
    if y is not None:
        report["class_distribution"] = {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        }
    return report

def build_data_quality_report(
    master_path: str,
    panel_paths: Dict[str, str]
) -> Dict[str, Any]:
    """Profiles all datasets and records Variant_ID overlap from panels into master."""
    report: Dict[str, Any] = {
        "master": profile_dataset(master_path, is_test=False),
        "panels": {},
        "master_panel_overlap": {}
    }
    master_ids = set(pd.read_csv(master_path, usecols=[ID_COL])[ID_COL])

    for panel_name, panel_path in panel_paths.items():
        path = Path(panel_path)
        if not path.exists():
            report["panels"][panel_name] = {"file": str(panel_path), "missing": True}
            report["master_panel_overlap"][panel_name] = {
                "overlap_count": 0,
                "panel_rows": 0,
                "overlap_ratio": 0.0,
                "leakage_warning": False
            }
            continue

        panel_report = profile_dataset(str(path), is_test=False)
        panel_ids = set(pd.read_csv(path, usecols=[ID_COL])[ID_COL])
        overlap_count = len(master_ids & panel_ids)
        report["panels"][panel_name] = panel_report
        report["master_panel_overlap"][panel_name] = {
            "overlap_count": int(overlap_count),
            "panel_rows": int(len(panel_ids)),
            "overlap_ratio": float(overlap_count / max(len(panel_ids), 1)),
            "leakage_warning": bool(overlap_count > 0)
        }

    return report

def deduplicate_dataset(
    X: pd.DataFrame, 
    y: pd.Series, 
    variant_ids: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs systematic quality control:
    - Finds rows with identical feature vectors.
    - If labels are consistent, keeps only one unique copy.
    - If labels are contradictory (same features, different labels), removes all copies to prevent noisy label confusion.
    """
    # Combine X and y temporarily to assess duplicates
    df = X.copy()
    df[TARGET_COL] = y
    df[ID_COL] = variant_ids
    
    # Feature columns for matching
    feature_cols = list(X.columns)
    
    # Find all rows that share identical feature profiles
    # Fill NAs temporarily with a dummy value to enable exact duplication checking
    df_filled = df[feature_cols].fillna("MISSING_DUMMY")
    duplicated_mask = df_filled.duplicated(keep=False)
    
    if not duplicated_mask.any():
        return X, y, variant_ids
        
    # Separate non-duplicated and duplicated sets
    non_duplicates = df[~duplicated_mask]
    duplicates = df[duplicated_mask]
    
    # Group duplicated records by features to verify label consistency
    clean_duplicates = []
    
    # Use MD5 hash or string join of row values for grouping high-dimensional features efficiently
    # Using python's hashable representations
    grouped = duplicates.groupby(list(df_filled[duplicated_mask].columns))
    
    for _, group in grouped:
        unique_labels = group[TARGET_COL].unique()
        if len(unique_labels) == 1:
            # Consistent label across all identical profiles -> keep the first record
            clean_duplicates.append(group.iloc[0:1])
        else:
            # Contradictory labels found -> drop entire group (noisy labels)
            pass
            
    if clean_duplicates:
        cleaned_df = pd.concat([non_duplicates] + clean_duplicates, ignore_index=True)
    else:
        cleaned_df = non_duplicates.reset_index(drop=True)
        
    final_X = cleaned_df[feature_cols]
    final_y = cleaned_df[TARGET_COL]
    final_ids = cleaned_df[ID_COL]
    
    return final_X, final_y, final_ids

def get_cv_splits(
    y: pd.Series, 
    is_small_panel: bool = False,
    n_repeats: int = PANEL_CV_REPEATS
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates cross-validation split indices securely.
    - For regular datasets, uses StratifiedKFold.
    - For extremely small panels (e.g. CFTR N=111), uses RepeatedStratifiedKFold (10 repeats of 5-fold)
      or standard StratifiedKFold depending on stability requirements.
    """
    n_samples = len(y)
    
    if is_small_panel and n_samples < 150:
        # Small panel set -> Repeated Stratified CV to assess test variance
        cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=n_repeats, random_state=RANDOM_SEED)
        return list(cv.split(np.zeros(n_samples), y))
    else:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        return list(cv.split(np.zeros(n_samples), y))
