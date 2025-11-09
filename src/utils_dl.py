import os
import datetime as dt
import collections
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    f1_score,
)

import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    ConvLSTM2D,
    MaxPool3D,
    Flatten,
    Dense,
)
from keras.utils import plot_model

PROJECT_ROOT = Path(r"C:\Users\elisa\Documents\elisa_projects\motor_imagery_EEG")
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


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


def build_conv_lstm(
    sequence_size: Tuple[int, int, int]) -> keras.Model:
    n_frames, frame_length, n_channels = sequence_size

    f_out = [16, 32, 64]
    kernel_t = [3, 3, 3]
    kernel_e = [1, 1, 1]

    inputs = keras.Input(shape=sequence_size + (1,))  # add "channels_last" dim

    # 1st ConvLSTM block
    x = ConvLSTM2D(
        filters=f_out[0],
        kernel_size=(kernel_t[0], kernel_e[0]),
        activation="tanh",
        data_format="channels_last",
        recurrent_dropout=0.2,
        return_sequences=True,
    )(inputs)
    x = MaxPool3D(pool_size=(1, 2, 1), data_format="channels_last")(x)

    # 2nd ConvLSTM block
    x = ConvLSTM2D(
        filters=f_out[1],
        kernel_size=(kernel_t[1], kernel_e[1]),
        activation="tanh",
        data_format="channels_last",
        recurrent_dropout=0.2,
        return_sequences=True,
    )(x)
    x = MaxPool3D(pool_size=(1, 2, 1), data_format="channels_last")(x)

    # 3rd ConvLSTM block
    x = ConvLSTM2D(
        filters=f_out[2],
        kernel_size=(kernel_t[2], kernel_e[2]),
        activation="tanh",
        data_format="channels_last",
        recurrent_dropout=0.2,
        return_sequences=True,
    )(x)
    x = MaxPool3D(pool_size=(1, 2, 1), data_format="channels_last")(x)

    x = Flatten()(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="ConvLSTM")

    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    return model


def train_and_save_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 64,
    num_epochs: int = 100,
    model_prefix: str = "ConvLSTM",
    model_path: Path | str = ""
) -> Tuple[keras.callbacks.History, Path]:
    # add last dim (channel) for ConvLSTM input if needed
    if x_train.ndim == 4:
        x_train = x_train[..., np.newaxis]

    print("Using GPU devices:", tf.config.list_physical_devices("GPU"))

    with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
        history = model.fit(
            x_train,
            y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
        )

    # timestamped filename
    date_time_format = "%Y_%m_%d__%H_%M_%S"
    timestamp = dt.datetime.now().strftime(date_time_format)

    model_file = model_path / f"{model_prefix}_Date_Time_{timestamp}.h5"
    history_file = model_path / f"{model_prefix}_history_{timestamp}.npy"

    model.save(model_file)
    np.save(history_file, history.history)

    print(f"Model saved to: {model_file}")
    print(f"History saved to: {history_file}")


    return history, str(model_file)


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


def plot_confusion_matrix(
    true_labels: Sequence[int] | np.ndarray,
    predicted_labels: Sequence[int] | np.ndarray,
    normalize: bool = True,
) -> None:
    if normalize:
        cm = confusion_matrix(true_labels, predicted_labels, normalize="true")
    else:
        cm = confusion_matrix(true_labels, predicted_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    print(classification_report(true_labels, predicted_labels, target_names=["0", "1"]))


def plot_roc_curve(
    y_true: Sequence[int] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def evaluate_classifier(
    model: keras.Model,
    x_test_vid: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    # Add channel dimension if needed
    if x_test_vid.ndim == 4:
        x_test_vid = x_test_vid[..., np.newaxis]

    y_pred_prob = model.predict(x_test_vid)
    y_pred = (y_pred_prob >= threshold).astype(int).ravel()

    print("Crop-level accuracy:", accuracy_score(y_test, y_pred))
    print("Crop-level F1 score:", f1_score(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)

    return y_pred, y_pred_prob

def majority_vote_over_crops(
    y_pred_crops: np.ndarray,
    factor: int = 3,
) -> np.ndarray:
    y_flat = y_pred_crops.flatten()
    assert len(y_flat) % factor == 0, "Number of crops must be a multiple of factor."

    y_grouped = y_flat.reshape((-1, factor))

    def most_common(x):
        counts = np.bincount(x)
        return np.argmax(counts)

    y_trial_pred = np.apply_along_axis(most_common, axis=1, arr=y_grouped)

    print(f"Number of crop labels: {len(y_flat)}")
    print(f"Number of trial-level labels: {len(y_trial_pred)}")

    return y_trial_pred

