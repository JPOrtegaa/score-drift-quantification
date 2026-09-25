"""T0: unit tests for IBDD and its image helpers (no external data)."""
import random

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from PIL import Image

from methods.drift_detectors import IBDD, BatchContext
from methods.drift_detectors import ibdd as ibdd_module
from methods.drift_detectors._images import window_to_image, msd

D = 50
SHAPE = np.sin(np.linspace(0, 2 * np.pi, D))


# Time-series-like examples (a scaled sine plus noise): the kind of structured
# rows IBDD is designed for. flip=True reverses the sign, like the paper's
# StarLightCurves-YReversed drift.
def curves(rng, n, flip=False):
    X = rng.uniform(0.5, 1.5, size=(n, 1)) * SHAPE + rng.normal(scale=0.2, size=(n, D))
    return -X if flip else X


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---- images and MSD ----

def test_jpeg_image_matches_the_original_disk_round_trip(rng, tmp_path):
    window = rng.normal(size=(40, 12))
    path = tmp_path / "w1.jpeg"
    plt.imsave(path, pd.DataFrame(window).transpose(), cmap='Greys', dpi=100)
    with Image.open(path) as image:
        from_disk = np.asarray(image)

    np.testing.assert_array_equal(window_to_image(window, "jpeg"), from_disk)


def test_jpeg_image_matches_skimage_imread(rng, tmp_path):
    skimage_io = pytest.importorskip("skimage.io")
    window = rng.normal(size=(40, 12))
    path = tmp_path / "w1.jpeg"
    plt.imsave(path, window.T, cmap='Greys', dpi=100)
    np.testing.assert_array_equal(window_to_image(window, "jpeg"), skimage_io.imread(path))


def test_jpeg_image_is_features_by_examples_rgb_uint8(rng):
    image = window_to_image(rng.normal(size=(40, 12)), "jpeg")
    assert image.shape == (12, 40, 3)
    assert image.dtype == np.uint8


def test_array_image_is_min_max_normalized(rng):
    image = window_to_image(rng.normal(size=(40, 12)) * 7 + 3, "array")
    assert image.shape == (12, 40)
    assert image.min() == 0.0 and image.max() == 1.0
    assert np.all(window_to_image(np.ones((5, 3)), "array") == 0)


def test_unknown_backend_is_rejected(rng):
    with pytest.raises(ValueError):
        window_to_image(rng.normal(size=(4, 3)), "png")
    with pytest.raises(ValueError):
        IBDD(image_backend="png")


def test_msd_properties(rng):
    a = window_to_image(rng.normal(size=(30, 8)))
    b = window_to_image(rng.normal(size=(30, 8)))
    assert msd(a, a) == 0.0
    assert msd(a, b) == msd(b, a) > 0
    assert msd(a, b) == pytest.approx(np.mean((a.astype(float) - b.astype(float)) ** 2), rel=0, abs=0)
    with pytest.raises(ValueError):
        msd(a, a[:, :-1])


def test_msd_matches_skimage(rng):
    metrics = pytest.importorskip("skimage.metrics")
    a = window_to_image(rng.normal(size=(30, 8)))
    b = window_to_image(rng.normal(size=(30, 8)))
    assert msd(a, b) == metrics.mean_squared_error(a, b)


# ---- calibration ----

def test_initial_thresholds_follow_the_original_permutations(rng):
    X = rng.normal(size=(300, 10))
    w = 60
    detector = IBDD(window_length=w).fit(X)

    # find_initial_threshold() of the authors: global random, re-seeded per run,
    # shuffling the same index list in place.
    reference = window_to_image(X[-w:])
    sequence = [i for i in range(X.shape[0])]
    expected = []
    for i in range(20):
        random.seed(i)
        random.shuffle(sequence)
        expected.append(msd(reference, window_to_image(X[sequence[:w]])))

    assert detector.distances.tolist() == expected
    assert detector.thr_upper == np.mean(expected) + 2 * np.std(expected)
    assert detector.thr_lower == max(np.mean(expected) - 2 * np.std(expected), 0)


def test_lower_threshold_is_clipped_at_zero(rng, monkeypatch):
    values = iter([0.0] * 19 + [100.0])
    monkeypatch.setattr(ibdd_module, "msd", lambda a, b: next(values))
    detector = IBDD(window_length=10).fit(rng.normal(size=(50, 4)))
    assert detector.thr_lower == 0
    assert detector.thr_upper > 0


def test_window_is_capped_at_training_size(rng):
    detector = IBDD(window_length=500).fit(rng.normal(size=(120, 4)))
    assert detector.window_length == 120
    assert IBDD().fit(rng.normal(size=(80, 4))).window_length == 80


def test_fit_ignores_the_class_column(rng):
    X = rng.normal(size=(200, 5))
    df = pd.DataFrame(X)
    df['class'] = rng.integers(0, 2, size=200)
    a = IBDD(window_length=50).fit(df)
    b = IBDD(window_length=50).fit(X)
    assert a.distances.tolist() == b.distances.tolist()


# ---- batch mode ----

@pytest.mark.parametrize("backend", ["jpeg", "array"])
def test_batch_mode_flags_sign_flip_and_rarely_stationary(rng, backend):
    detector = IBDD(window_length=100, image_backend=backend).fit(curves(rng, 2000))
    false_alarms = np.mean([detector.detect(curves(rng, 100)) for _ in range(200)])
    detections = np.mean([detector.detect(curves(rng, 100, flip=True)) for _ in range(100)])
    assert false_alarms <= 0.10
    assert detections == 1.0


def test_batch_mode_ignores_the_order_of_examples_in_a_bag(rng):
    # Bags sorted by class (as the synthetic generator writes them): without the
    # shuffle, the aligned class blocks alone make every bag look like drift.
    shapes = [SHAPE, np.cos(np.linspace(0, 2 * np.pi, D))]

    def sorted_bag(n):
        y = np.sort(rng.integers(0, 2, n))
        return (np.stack([shapes[c] for c in y]) * rng.uniform(0.5, 1.5, (n, 1))
                + rng.normal(scale=0.2, size=(n, D)))

    train = np.vstack([sorted_bag(100) for _ in range(10)])
    shuffled = IBDD(window_length=100).fit(train)
    in_order = IBDD(window_length=100, shuffle_batch=False).fit(train)
    bags = [sorted_bag(100) for _ in range(50)]
    assert np.mean([shuffled.detect(b) for b in bags]) <= 0.10
    assert np.mean([in_order.detect(b) for b in bags]) >= 0.90
    # Deterministic: the same batch always gets the same permutation.
    assert shuffled.statistic(bags[0]) == shuffled.statistic(bags[0])


def test_batch_mode_accepts_a_context_or_a_frame(rng):
    detector = IBDD(window_length=100).fit(curves(rng, 500))
    batch = curves(rng, 100)
    frame = pd.DataFrame(batch)
    frame['class'] = 1
    expected = detector.statistic(batch)
    assert detector.statistic(BatchContext(X=pd.DataFrame(batch))) == expected
    assert detector.statistic(frame) == expected


def test_batch_mode_rejects_other_batch_sizes(rng):
    detector = IBDD(window_length=100).fit(curves(rng, 500))
    with pytest.raises(ValueError):
        detector.detect(curves(rng, 99))


def test_batch_mode_requires_fit(rng):
    with pytest.raises(ValueError):
        IBDD(window_length=10).statistic(curves(rng, 10))


# ---- stream mode ----

def test_stream_mode_detects_an_abrupt_drift_quickly(rng):
    stream = np.vstack([curves(rng, 1000), curves(rng, 1000), curves(rng, 1000, flip=True)])
    detector = IBDD(window_length=300, consecutive_values=3).start_stream(stream[:1000])
    flags = [detector.update(x) for x in stream[1000:]]

    assert [i for i, flag in enumerate(flags) if flag] == detector.drift_points
    before = [p for p in detector.drift_points if p < 1000]
    after = [p for p in detector.drift_points if p >= 1000]
    assert len(before) <= 5
    assert after and after[0] - 1000 <= 30
    assert len(detector.msd_history) == 20 + 2000
    np.testing.assert_array_equal(detector.window, stream[-300:])
