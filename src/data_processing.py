from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, List

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def standardize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, scaler

def sequ_feature_selection(
    estimator: BaseEstimator,
    n_features: int,
    X: pd.DataFrame,
    y: ArrayLike,
    corr_based: bool = False,
    method: str = "SFS",
    corr_threshold: float = 0.85,
    scoring: str = "accuracy",
    cv: int = 3,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Sequential feature selection with optional correlation-based prefiltering.
    
    corr_based : bool, optional
        If True, first remove highly correlated features 
        (|r| > corr_threshold) by keeping the less redundant one.
    method : {"SBS", "SFS", "SBFS", "SFFS"}
        - "SBS" : Sequential Backward Selection
        - "SFS" : Sequential Forward Selection
        - "SBFS": Sequential Backward Floating Selection
        - "SFFS": Sequential Forward Floating Selection
    """
    
    # Optional correlation-based filtering of features
    if corr_based:
        corr_matrix = X.corr().abs()
        mean_corr = corr_matrix.mean(axis=0)
        correlated_to_drop = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > corr_threshold:
                    # Drop the more redundant one (higher mean correlation)
                    if mean_corr[i] > mean_corr[j]:
                        colname = corr_matrix.columns[i]
                    else:
                        colname = corr_matrix.columns[j]
                    correlated_to_drop.add(colname)
        
        X_prefiltered = X.drop(columns=list(correlated_to_drop))
    else:
        X_prefiltered = X.copy()
    
    # Configure SFS 
    forward = method in ("SFS", "SFFS")
    floating = method in ("SBFS", "SFFS")
    
    sfs = SFS(
        estimator,
        k_features=n_features,
        forward=forward,
        floating=floating,
        verbose=2,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )
    
    sfs = sfs.fit(X_prefiltered.values, y)  # mlxtend expects numpy arrays
    feat_idx = list(sfs.k_feature_idx_)
    feat_names = list(X_prefiltered.columns[feat_idx])
    
    return X_prefiltered[feat_names], feat_names
