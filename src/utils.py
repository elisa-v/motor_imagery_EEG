from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import scipy


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