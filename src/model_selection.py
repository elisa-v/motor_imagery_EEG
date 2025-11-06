
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from data_visualisation import plot_confusion_matrix, plot_roc_curve


def hyperp_search(classifier, param_grid, X_tr, y_tr, X_te, y_te):
    gs = GridSearchCV(
        classifier,
        param_grid,
        cv=3,
        scoring="accuracy",
        verbose=0,
        n_jobs=-1
    )
    gs.fit(X_tr, y_tr)
    
    print(f"Best CV accuracy: {gs.best_score_:.3f}  using {gs.best_params_}")
    
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_te)
    acc_test = metrics.accuracy_score(y_te, y_pred)
    print(f"Test accuracy: {acc_test:.3f}")
    
    if hasattr(best_model, "predict_proba"):
        y_pred_proba = best_model.predict_proba(X_te)[:, 1]
        plot_confusion_matrix(y_te, y_pred)
        plot_roc_curve(y_te, y_pred_proba)
    else:
        plot_confusion_matrix(y_te, y_pred)
