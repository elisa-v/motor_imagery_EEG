from pathlib import Path
from typing import Sequence, Tuple
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import classification_report

from src.data_visualisation import plot_confusion_matrix, plot_roc_curve


def load_feature_datasets(
    train_path: str | Path,
    test_path: str | Path,
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
  
    train_path = Path(train_path)
    test_path = Path(test_path)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    y_train = df_train[label_col].copy()
    y_test = df_test[label_col].copy()

    X_train = df_train.drop(columns=[label_col])
    X_test = df_test.drop(columns=[label_col])

    return X_train, X_test, y_train, y_test


def load_eeg_mat_dataset(mat_path: str | Path,
    struct_name: str = "training_set") -> Tuple[np.ndarray, np.ndarray]:
    mat_file = scipy.io.loadmat(mat_path)
    struct_data = mat_file[struct_name] 

    eeg_field = struct_data["eeg_sequences"]   # cell array / struct field
    label_field = struct_data["label"]

    sequences = [eeg_field[0][i] for i in range(len(eeg_field[0]))]
    labels = [label_field[0][0][i][0] for i in range(len(label_field[0][0]))]

    x = np.array(sequences)
    y = np.array(labels, dtype="uint8")

    print(f"Loaded {struct_name} from {mat_path}")
    print("  x shape:", x.shape)
    print("  y shape:", y.shape)

    return x, y


def save_linear_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    results_dir: Path,
    model_name: str,
    class_labels: Sequence[str],
    y_pred_proba: np.ndarray | None = None,
    ) -> None:
 
    results_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix (PNG) 
    cm_path = results_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true,
        y_pred,
        normalize=True,
        class_labels=class_labels,
        save_path=cm_path,
        show=False,
        print_report=False,
    )

    # ROC curve, if probs available 
    if y_pred_proba is not None:
        roc_path = results_dir / f"{model_name}_roc_curve.png"
        plot_roc_curve(
            y_true,
            y_pred_proba,
            label=model_name,
            save_path=roc_path,
            show=False,
        )
    else:
        roc_path = None

    # Classification report (TXT) 
    report = classification_report(y_true, y_pred, target_names=class_labels)
    report_path = results_dir / f"{model_name}_classification_report.txt"
    report_path.write_text(report)

    print(f"Results saved to {results_dir}")
    print(f" - Confusion matrix: {cm_path.name}")
    if roc_path is not None:
        print(f" - ROC curve: {roc_path.name}")
    print(f" - Classification report: {report_path.name}")