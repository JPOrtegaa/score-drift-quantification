import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import pandas as pd


# Everything a detector may look at for one test batch. The experiment builds it
# once per batch and every detector reads only the fields it needs: CDT works on
# classifier scores, IBDD on the raw features.
@dataclass
class BatchContext:
    X: Optional[pd.DataFrame] = None            # test batch features (no 'class' column)
    test_scores: Optional[np.ndarray] = None    # positive-class scores of the test batch
    pos_scores: Optional[np.ndarray] = None     # training scores of the positive class
    neg_scores: Optional[np.ndarray] = None     # training scores of the negative class


# Common interface of the drift detectors. A detector is fitted on training data,
# reduces a batch to one statistic, and flags drift when that statistic falls
# outside [thr_lower, thr_upper].
class DriftDetector(ABC):
    name: ClassVar[str]
    # True: one detector per binary (one-vs-rest) model, since it depends on that
    # model's scores. False: one detector per dataset, shared by every class.
    per_class: ClassVar[bool]

    thr_upper = None
    thr_lower = None
    distances = None

    @abstractmethod
    def fit(self, train):
        ...

    # Reduce a test batch to the scalar compared against the thresholds.
    @abstractmethod
    def statistic(self, ctx):
        ...

    # Return if the statistic indicates drift based on the two-sided threshold.
    def predict(self, distance):
        return bool((distance >= self.thr_upper) or (distance <= self.thr_lower))

    def detect(self, ctx):
        return self.predict(self.statistic(ctx))

    # Rebuild a fitted detector from stored thresholds only (e.g. read back from
    # distances.csv). A missing lower bound becomes -inf so only the upper gates.
    @classmethod
    def from_thresholds(cls, upper, lower=None, **kwargs):
        detector = cls(**kwargs)
        detector.thr_upper = upper
        detector.thr_lower = lower if lower is not None else -np.inf
        return detector

    # Persist the calibration distances collected during fit() to a CSV file.
    # Appends one row per model so every model of a dataset shares a single
    # file. Each row stores the model id, the fitted thresholds, and the
    # serialized distance array. Pass overwrite=True for the first model of a
    # dataset so any file from a previous run is rewritten from scratch.
    def save_distances(self, path, model_id=None, overwrite=False):
        if self.distances is None:
            raise ValueError("No distances to save; call fit() before save_distances().")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        row = {
            "model_id": model_id,
            "thr_lower": self.thr_lower,
            "thr_upper": self.thr_upper,
            "distances": json.dumps(np.asarray(self.distances).tolist()),
        }
        mode = "w" if overwrite else "a"
        header = overwrite or not os.path.exists(path)
        pd.DataFrame([row]).to_csv(path, mode=mode, header=header, index=False)
