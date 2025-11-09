import os
import datetime as dt
import collections
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import scipy.io

from sklearn.metrics import (
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

from src.data_visualisation import plot_confusion_matrix, plot_roc_curve


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

def majority_vote_over_imagery_sequence(
    y_pred_crops: np.ndarray,
    y_pred_prob_crops: np.ndarray,
    factor: int = 3,
    agg: Literal["mean", "max"] = "mean",
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


def aggregate_crops_to_trials_from_probs(
    y_pred_prob_crops: np.ndarray,
    factor: int = 3,
    threshold: float = 0.5,
    agg: str = "mean",  
    ) -> Tuple[np.ndarray, np.ndarray]:

    probs_flat = np.asarray(y_pred_prob_crops).ravel()
    assert len(probs_flat) % factor == 0, "Number of crops must be a multiple of factor."

    n_trials = len(probs_flat) // factor
    probs_grouped = probs_flat.reshape(n_trials, factor)

    if agg == "mean":
        trial_probs = probs_grouped.mean(axis=1)
    elif agg == "max":
        trial_probs = probs_grouped.max(axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {agg!r}")

    y_trial_pred = (trial_probs >= threshold).astype(int)

    print(f"Number of crops: {len(probs_flat)}")
    print(f"Number of trials: {n_trials}")

    return y_trial_pred, trial_probs


