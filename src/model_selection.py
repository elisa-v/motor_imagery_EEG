from __future__ import annotations

from typing import Mapping, Sequence, Any, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from .data_visualisation import plot_confusion_matrix, plot_roc_curve


def hyperp_search(
    classifier: BaseEstimator,
    param_grid: Mapping[str, Sequence[Any]],
    X_tr: pd.DataFrame | np.ndarray,
    y_tr: ArrayLike,
    X_te: pd.DataFrame | np.ndarray,
    y_te: ArrayLike,
    class_labels: Sequence[str] | None = None,
) -> Tuple[BaseEstimator, float]:
    gs = GridSearchCV(
        classifier,
        param_grid,
        cv=3,
        scoring="accuracy",
        verbose=0,
        n_jobs=-1,
    )
    gs.fit(X_tr, y_tr)

    print(f"Best CV accuracy: {gs.best_score_:.3f} using {gs.best_params_}")

    best_model: BaseEstimator = gs.best_estimator_
    y_pred = best_model.predict(X_te)
    test_accuracy = metrics.accuracy_score(y_te, y_pred)
    print(f"Test accuracy: {test_accuracy:.3f}")

    if hasattr(best_model, "predict_proba"):
        y_pred_proba = best_model.predict_proba(X_te)[:, 1]
        plot_confusion_matrix(y_te, y_pred, class_labels=class_labels)
        plot_roc_curve(y_te, y_pred_proba, label=best_model.__class__.__name__)
    else:
        plot_confusion_matrix(y_te, y_pred, class_labels=class_labels)

    return best_model, test_accuracy