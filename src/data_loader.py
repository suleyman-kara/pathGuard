import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, RepeatedStratifiedKFold
from src.config import ID_COL, TARGET_COL, RANDOM_SEED, CV_FOLDS

def load_data(file_path: str, is_test: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
    """
    Loads dataset from CSV file.
    
    Returns:
        X: Feature DataFrame
        y: Target Series (None if is_test=True)
        variant_ids: Variant identifier Series
    """
    df = pd.read_csv(file_path)
    
    # Extract variant IDs
    if ID_COL in df.columns:
        variant_ids = df[ID_COL]
        df = df.drop(columns=[ID_COL])
    else:
        # Fallback if no explicit column
        variant_ids = pd.Series([f"VAR_{i}" for i in range(len(df))], name=ID_COL)
        
    # Extract labels
    if not is_test and TARGET_COL in df.columns:
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
    else:
        y = None
        X = df
        if TARGET_COL in X.columns:
            X = X.drop(columns=[TARGET_COL])
            
    return X, y, variant_ids

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
    is_small_panel: bool = False
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
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
        return list(cv.split(np.zeros(n_samples), y))
    else:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        return list(cv.split(np.zeros(n_samples), y))
