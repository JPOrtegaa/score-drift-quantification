"""T2: this repository's IBDD against the authors' implementation on the paper's
benchmark datasets (plus the T3 check against the authors' Yoga notebook).

Slow (minutes per dataset); run with `pytest -m slow`. Needs
`python validation/ibdd/fetch.py all` and scikit-image (requirements-dev.txt).
The three largest benchmarks (Heartbeats, Insects, Posture) only run through
validation/ibdd/compare_ibdd.py.
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "validation", "ibdd"))

from compare_ibdd import NOTEBOOK_YOGA, dataset_files  # noqa: E402
from harness import load_split, run_ours  # noqa: E402
from original_compat import ORIGINAL_CODE  # noqa: E402

pytest.importorskip("skimage")


@pytest.mark.parametrize("name", ["Yoga", "UWave", "StarLightCurves"])
def test_same_drifts_accuracy_and_msd_as_the_original(name):
    train_file, test_file = dataset_files(name)
    if not (os.path.exists(train_file) and os.path.exists(ORIGINAL_CODE)):
        pytest.skip("run: python validation/ibdd/fetch.py all")
    from original_compat import run_original_ibdd

    train_X, train_y = load_split(train_file)
    test_X, test_y = load_split(test_file)
    window = len(train_y)

    ours = run_ours(train_X, train_y, test_X, test_y, window, 3)
    original = run_original_ibdd(train_file, test_file, window, 3)

    assert ours["drift_points"] == original["drift_points"]
    np.testing.assert_array_equal(ours["vet_acc"], original["vet_acc"])
    np.testing.assert_allclose(ours["msd_trace"], original["msd_trace"], rtol=0, atol=1e-9)

    if name == "Yoga":
        assert ours["drift_points"] == NOTEBOOK_YOGA["drift_points"]
        assert round(ours["mean_acc"], 2) == NOTEBOOK_YOGA["mean_acc"]
