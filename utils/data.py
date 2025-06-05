import pandas as pd
import numpy as np
from collections import OrderedDict

from sklearn.model_selection import train_test_split


def normalize_labels(y):
    unq_labels = np.unique(y)
    label_dict = OrderedDict(zip(unq_labels, np.arange(len(unq_labels))))
    return pd.Series(y).replace(label_dict).values

def data_splitting(X: np.ndarray, y: np.ndarray, seed: int, train_size):
    X_train, X_hold_out, y_train, y_hold_out = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_hold_out,
        y_hold_out,
        train_size=train_size,
        random_state=seed,
        stratify=y_hold_out,
        shuffle=True,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
