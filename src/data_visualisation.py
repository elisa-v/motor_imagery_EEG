
from __future__ import annotations
from pathlib import Path

import keras
from matplotlib import pyplot as plt

from typing import Sequence

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
    

def plot_confusion_matrix(
    true_labels: Sequence[int] | np.ndarray,
    predicted_labels: Sequence[int] | np.ndarray,
    normalize: bool = True,
    class_labels: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    print_report: bool = True,
    ) -> None:

    if normalize:
        cm = confusion_matrix(true_labels, predicted_labels, normalize="true")
    else:
        cm = confusion_matrix(true_labels, predicted_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")

    if class_labels is None:
        class_labels = ["0", "1"]

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    if print_report:
        report = classification_report(true_labels, predicted_labels,
                                       target_names=class_labels)
        print(report)


def plot_roc_curve(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    label: str = "ROC",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(
    y: pd.Series | Sequence[int] | np.ndarray,
    title: str = "Class distribution",
    figsize: tuple[int, int] = (4, 3),
    xlabel: str = "Class",
    ylabel: str = "Count",
    palette: str = "Set2",
) -> None:

    # Convert to DataFrame for seaborn
    y_series = pd.Series(y, name=xlabel)

    class_counts = y_series.value_counts().sort_index()

    print("Class counts:")
    print(class_counts)

    plt.figure(figsize=figsize)
    sns.countplot(x=y_series, palette=palette)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_training_curves(history: keras.callbacks.History) -> None:
    loss = history.history.get("loss", [])
    acc = history.history.get("accuracy", [])

    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, label="Training loss")
    plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, acc, label="Training accuracy")
    plt.title("Training accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

