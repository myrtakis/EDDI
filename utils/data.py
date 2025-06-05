import pandas as pd
import numpy as np
from collections import OrderedDict

def normalize_labels(y):
    unq_labels = np.unique(y)
    label_dict = OrderedDict(zip(unq_labels, np.arange(len(unq_labels))))
    return pd.Series(y).replace(label_dict).values
