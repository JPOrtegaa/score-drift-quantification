"""The evaluation protocol of the IBDD paper, driven by this repository's IBDD.

Mirrors IBDD() in the authors' detectors.py: a RandomForest trained on the
initial training set predicts every stream example (prequential accuracy), the
detector slides its window over the example, and after each drift the model is
retrained on the examples in the window with their labels.
"""
import os
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from methods.drift_detectors import IBDD  # noqa: E402


def load_split(path):
    """A .data file of the IBDD datasets: CSV without header, label in the last column."""
    data = pd.read_csv(path, header=None, index_col=False, sep=',')
    return data.iloc[:, :-1].to_numpy(dtype=np.float64), data.iloc[:, -1].to_numpy()


def new_model():
    return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)


def run_ours(train_X, train_y, test_X, test_y, window_length, consecutive_values, image_backend="jpeg",
             progress=False, desc="ours"):
    """Returns a dict with drift_points, vet_acc, mean_acc, execution_time and
    msd_trace (calibration MSDs followed by one per stream example). progress
    shows a tqdm bar over the stream examples."""
    model = new_model().fit(train_X, train_y)
    window_length = min(window_length, len(train_y))
    recent_y = list(train_y[-window_length:])

    detector = IBDD(window_length=window_length, consecutive_values=consecutive_values,
                    image_backend=image_backend).start_stream(train_X)

    # The model only changes at a drift, so the examples between two drifts are
    # predicted in one call; same predictions as the original's one-by-one loop.
    vet_acc = np.zeros(len(test_y))
    segment_start = 0
    start = timer()
    for i in tqdm(range(len(test_y)), desc=desc, unit="ex", mininterval=2, disable=not progress):
        recent_y = recent_y[1:] + [test_y[i]]
        if detector.update(test_X[i]):
            vet_acc[segment_start:i + 1] = model.predict(test_X[segment_start:i + 1]) == test_y[segment_start:i + 1]
            model.fit(detector.window, np.asarray(recent_y))
            segment_start = i + 1
    vet_acc[segment_start:] = model.predict(test_X[segment_start:]) == test_y[segment_start:]
    execution_time = timer() - start

    return {
        "drift_points": list(detector.drift_points),
        "vet_acc": vet_acc,
        "mean_acc": np.mean(vet_acc) * 100,
        "execution_time": execution_time,
        "msd_trace": list(detector.msd_history),
    }


def run_baseline(train_X, train_y, test_X, test_y):
    """Static classifier, never updated (the paper's Baseline). Mean accuracy in %."""
    model = new_model().fit(train_X, train_y)
    return np.mean(model.predict(test_X) == test_y) * 100
