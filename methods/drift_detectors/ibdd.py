import random

import numpy as np
import pandas as pd

from ._images import IMAGE_BACKENDS, window_to_image, msd
from .base import BatchContext, DriftDetector


def _features(data):
    if isinstance(data, BatchContext):
        data = data.X
    if isinstance(data, pd.DataFrame):
        data = data.drop(columns=['class'], errors='ignore')
    return np.asarray(data, dtype=np.float64)


# Image-Based Drift Detector (IBDD), Souza et al., "Unsupervised Drift Detection
# on High-speed Data Streams", IEEE BigData 2020.
#
# A window of w examples is drawn as a grayscale image (one row per feature, one
# column per example) and compared with a reference image of the training data
# through the Mean-Squared Deviation (MSD). The initial thresholds are
# mean +/- 2*std of the MSD between the reference image and n_permutations
# random windows of the training data.
#
# Two ways to use it:
#   * batch mode (fit + detect): each test batch of exactly w examples is one
#     window, compared against the training reference with the fixed initial
#     thresholds. Stateless, so batches can be processed in any order or in
#     parallel. This is what gates QuaDapt in the experiments.
#     The image comparison is pixel by pixel, so it depends on the order of the
#     examples. The calibration windows are random permutations of the training
#     data, while a test bag may come in any order (e.g. sorted by class), which
#     alone shifts its MSD away from the calibration values. With shuffle_batch
#     (default) each batch is put in a random order first, a fixed permutation
#     drawn from random_state, so batches are compared on the same footing as
#     the calibration windows and the result stays deterministic.
#   * stream mode (start_stream + update): the original algorithm, a sliding
#     window updated one example at a time, drift after `consecutive_values`
#     MSDs in a row beyond a threshold, and thresholds that adapt in stable
#     periods and after each drift. A line-by-line port of the authors' code,
#     kept to validate this implementation against it.
class IBDD(DriftDetector):
    name = "ibdd"
    per_class = False

    # Constants of the authors' implementation (detectors.py).
    STABLE_PERIOD = 60   # examples without drift before a periodic threshold update
    HISTORY = 50         # MSD values used by the threshold updates

    def __init__(self, window_length=None, consecutive_values=3, n_permutations=20, image_backend="jpeg",
                 shuffle_batch=True, random_state=0):
        if image_backend not in IMAGE_BACKENDS:
            raise ValueError(f"image_backend must be one of {IMAGE_BACKENDS}, got {image_backend!r}")
        self.window_length = window_length
        self.consecutive_values = consecutive_values
        self.n_permutations = n_permutations
        self.image_backend = image_backend
        self.shuffle_batch = shuffle_batch
        self.random_state = random_state
        self.reference_image = None
        self.distances = None
        self.thr_upper, self.thr_lower = None, None

    def _image(self, window):
        return window_to_image(window, self.image_backend)

    # Initial thresholds: compare the reference image with n_permutations random
    # windows of the training data. Port of find_initial_threshold(): the index
    # sequence is re-seeded with the run number and shuffled in place, so each
    # run shuffles the previous permutation again.
    def _initial_thresholds(self, X):
        sequence = [i for i in range(X.shape[0])]
        distances = []
        for run in range(self.n_permutations):
            random.Random(run).shuffle(sequence)
            window = X[sequence[:self.window_length]]
            distances.append(msd(self.reference_image, self._image(window)))

        upper = np.mean(distances) + 2 * np.std(distances)
        lower = np.mean(distances) - 2 * np.std(distances)
        if lower < 0:
            lower = 0
        return upper, lower, distances

    # Fit the reference image (last window_length training examples) and the
    # initial thresholds. train may carry a 'class' column; it is ignored. The
    # training matrix itself is not kept, so the fitted detector stays small.
    def fit(self, train):
        X = _features(train)
        if self.window_length is None or self.window_length > X.shape[0]:
            self.window_length = X.shape[0]

        self.reference_image = self._image(X[-self.window_length:])
        self.thr_upper, self.thr_lower, distances = self._initial_thresholds(X)
        self.distances = np.array(distances)
        return self

    # MSD between the training reference and one test batch (batch mode).
    def statistic(self, ctx):
        if self.reference_image is None:
            raise ValueError("IBDD is not fitted; call fit() first.")
        X = _features(ctx)
        if X.shape[0] != self.window_length:
            raise ValueError(
                f"IBDD compares windows of {self.window_length} examples, got a batch of {X.shape[0]}; "
                "fit it with window_length equal to the batch size."
            )
        if self.shuffle_batch:
            X = X[np.random.default_rng(self.random_state).permutation(X.shape[0])]
        return msd(self.reference_image, self._image(X))

    # ---- Stream mode (the original algorithm) ----

    # Fit on the training data and start a stream whose sliding window holds the
    # last window_length training examples.
    def start_stream(self, train):
        X = _features(train)
        self.fit(X)
        self._window = X[-self.window_length:].copy()
        self.msd_history = list(self.distances)
        self._threshold_diffs = [self.thr_upper - self.thr_lower]
        self._last_update = 0
        self._t = 0
        self.drift_points = []
        return self

    # Slide the window over one new example; return True if a drift is flagged
    # at this example. Mirrors the per-example body of the authors' IBDD loop
    # (classification and retraining are left to the caller).
    def update(self, x):
        i = self._t
        self._t += 1
        history = self.msd_history

        self._window = np.vstack([self._window[1:], _features(x).reshape(1, -1)])
        history.append(msd(self.reference_image, self._image(self._window)))

        # Stable period: re-estimate both thresholds from the recent MSDs.
        if i - self._last_update > self.STABLE_PERIOD:
            recent = history[-self.HISTORY:]
            self.thr_upper = np.mean(recent) + 2 * np.std(recent)
            self.thr_lower = np.mean(recent) - 2 * np.std(recent)
            self._threshold_diffs.append(self.thr_upper - self.thr_lower)
            self._last_update = i

        last_values = history[-self.consecutive_values:]
        drift = False
        # Drift: move the crossed threshold just past the current MSD and keep the
        # other one at the average gap between thresholds seen so far.
        if all(v >= self.thr_upper for v in last_values):
            self.thr_upper = history[-1] + np.std(history[-self.HISTORY:-1])
            self.thr_lower = history[-1] - np.mean(self._threshold_diffs)
            drift = True
        elif all(v <= self.thr_lower for v in last_values):
            self.thr_lower = history[-1] - np.std(history[-self.HISTORY:-1])
            self.thr_upper = history[-1] + np.mean(self._threshold_diffs)
            drift = True

        if drift:
            self._threshold_diffs.append(self.thr_upper - self.thr_lower)
            self.drift_points.append(i)
            self._last_update = i
        return drift

    # Current sliding window (e.g. to retrain a model after a drift).
    @property
    def window(self):
        return self._window
