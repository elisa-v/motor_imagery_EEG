from __future__ import annotations

import collections
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


def augment_eeg_sequences(
    x: np.ndarray,
    y: np.ndarray,
    n_channels: int = 3,
    total_length: int = 750,  # corresponds to 3 s (750 / fs = 3 s)
    factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    
    crop_length = total_length // factor

    # reshape to (n_trials, factor, crop_length, n_channels)
    x_reshaped = x.reshape((x.shape[0], factor, crop_length, n_channels))

    # stack along trial axis -> (n_trials * factor, crop_length, n_channels)
    x_aug = x_reshaped.reshape((-1, crop_length, n_channels))

    # repeat labels for each crop
    y_aug = np.zeros(shape=(x_aug.shape[0],), dtype=y.dtype)
    for i in range(y.shape[0]):
        y_aug[i * factor : i * factor + factor].fill(y[i])

    print("Augmented data:")
    print("  x_aug shape:", x_aug.shape)
    print("  y_aug shape:", y_aug.shape)

    # sanity check: class balance
    print("Label distribution:", collections.Counter(y_aug))

    return x_aug, y_aug


def to_video_windows(
    x_aug: np.ndarray,
    n_frames: int,
    frame_length: int,
    n_channels: int = 3) -> np.ndarray:
    
    x_vid = x_aug.copy()

    # total_length should equal n_frames * frame_length
    total_length = x_vid.shape[1]
    assert (
        total_length == n_frames * frame_length
    ), f"Expected total_length={n_frames*frame_length}, got {total_length}"

    x_vid.resize((x_aug.shape[0], n_frames, frame_length, n_channels))

    print("Video-like representation:")
    print("  x_aug shape:", x_aug.shape)
    print("  x_vid shape:", x_vid.shape)

    return x_vid
