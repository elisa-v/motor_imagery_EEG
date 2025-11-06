
import scipy.io
import numpy as np


def load_eeg_mat(path, struct_name="training_set"):
    """
    Load a Matlab struct with fields:
      - eeg_sequences: cell array/list of [T x C]
      - label: vector of length N

    Returns
    -------
    X : np.ndarray, shape (N, T, C)
    y : np.ndarray, shape (N,)
    """
    mat = scipy.io.loadmat(path, squeeze_me=True)
    data_struct = mat[struct_name]

    eeg_seq = data_struct["eeg_sequences"]
    labels  = data_struct["label"]

    # Ensure we have a 1D list/array of sequences
    eeg_seq = np.atleast_1d(eeg_seq)

    X_list = []
    for seq in eeg_seq:
        arr = np.array(seq, dtype=np.float32)   # [T x C]
        X_list.append(arr)

    X = np.stack(X_list, axis=0)               # [N x T x C]
    y = np.array(labels).astype(int).ravel()   # [N]

    return X, y