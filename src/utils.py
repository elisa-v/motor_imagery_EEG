from pathlib import Path
from typing import Tuple
import pandas as pd


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