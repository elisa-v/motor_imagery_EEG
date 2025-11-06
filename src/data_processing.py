from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def sequ_feature_selection(
    estimator,
    n_features,
    X,
    y,
    corr_based=False,
    method="SFS",
    corr_threshold=0.85,
    scoring="accuracy",
    cv=3
):
    """
    Sequential feature selection with optional correlation-based prefiltering.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Model used to evaluate subsets (e.g. LDA, LogisticRegression).
    n_features : int
        Target number of features to select.
    X : pd.DataFrame
        Input feature matrix.
    y : array-like
        Target labels.
    corr_based : bool, optional
        If True, first remove highly correlated features 
        (|r| > corr_threshold) by keeping the less redundant one.
    method : {"SBS", "SFS", "SBFS", "SFFS"}
        - "SBS" : Sequential Backward Selection
        - "SFS" : Sequential Forward Selection
        - "SBFS": Sequential Backward Floating Selection
        - "SFFS": Sequential Forward Floating Selection
    corr_threshold : float
        Absolute correlation threshold for pruning.
    scoring : str
        Scikit-learn scoring metric.
    cv : int
        Number of CV folds.
        
    Returns
    -------
    X_selected : pd.DataFrame
        Reduced feature matrix containing only the selected features.
    feat_names : list of str
        Names of selected features.
    """
    
    # ----- Optional correlation-based filtering -----
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
    
    # ----- Configure SFS -----
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



def split_into_subsequences(X, y, factor=3, samples_per_window=250):
    """
    Split each EEG segment [T, C] into 'factor' shorter windows.
    Returns new arrays (X_aug, y_aug) with length N * factor.

    X: [N, T, C]
    """
    N, T, C = X.shape
    assert T >= factor * samples_per_window, (
        f"Sequence length {T} < factor * samples_per_window "
        f"({factor * samples_per_window})"
    )

    # Take only the first factor * samples_per_window samples (e.g. 3 * 250 = 750)
    X_trimmed = X[:, : factor * samples_per_window, :]   # [N, factor*L, C]

    # Reshape to [N, factor, L, C]
    X_split = X_trimmed.reshape(N, factor, samples_per_window, C)

    # Collapse factor dimension
    X_aug = X_split.reshape(-1, samples_per_window, C)

    # Repeat labels accordingly
    y_aug = np.repeat(y, factor)

    return X_aug, y_aug