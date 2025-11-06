
import keras


def build_conv_lstm_model(input_shape, num_classes=2):
    """
    Build a ConvLSTM-style model for EEG sequences.

    input_shape: (T, C, 1)  -> (time, electrodes, channels_in)
    """
    T, C, ch_in = input_shape

    inputs = keras.Input(shape=input_shape, name="eeg_input")

    # 1st ConvLSTM block
    x = layers.ConvLSTM2D(
        filters=16,
        kernel_size=(5, 1),         # temporal kernel 5, spatial kernel 1 over electrodes
        activation="tanh",
        data_format="channels_last",
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 1, 1))(tf.expand_dims(x, axis=-1))
    x = tf.squeeze(x, axis=-1)

    # 2nd ConvLSTM block
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(5, 1),
        activation="tanh",
        data_format="channels_last",
        return_sequences=False,
        dropout=0.2,
        recurrent_dropout=0.2
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    # Dense classifier
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="EEG_ConvLSTM")
    return model

input_shape = X_tr_dl.shape[1:]  # (T, C, 1)
model = build_conv_lstm_model(input_shape=input_shape, num_classes=2)

model.summary()
