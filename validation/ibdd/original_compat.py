"""Run the authors' IBDD (validation/ibdd/_external/code_site/detectors.py)
unmodified on the current library stack.

detectors.py was written for scikit-image < 0.18 and pandas < 2. Instead of
editing it, this module restores the few APIs it needs before importing it:
  * skimage.measure.compare_mse / compare_ssim, removed from scikit-image, are
    aliased to skimage.metrics.mean_squared_error / structural_similarity. The
    compare_mse alias also records every value, which exposes the full MSD trace
    of a run (the 20 calibration values followed by one per stream example).
  * DataFrame.append / Series.append, removed in pandas 2, are rebuilt on pd.concat.
  * plot_acc() is replaced by a no-op (it only draws a figure).
The original calls compare_mse exactly once per calibration run and once per
stream example, so the same alias also drives an optional tqdm progress bar.
Runs happen in a temporary directory because the original writes its window
images (w1.jpeg, w2.jpeg, ...) to the working directory.
"""
import contextlib
import importlib.util
import io
import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import skimage.measure
import skimage.metrics
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_CODE = os.path.join(HERE, "_external", "code_site", "detectors.py")

MSD_TRACE = []
_progress = None


def _compare_mse(image0, image1):
    value = skimage.metrics.mean_squared_error(image0, image1)
    MSD_TRACE.append(value)
    if _progress is not None:
        _progress.update()
    return value


def _append(self, other, ignore_index=False, verify_integrity=False, sort=False):
    return pd.concat([self, other], ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)


def load_original():
    if not os.path.exists(ORIGINAL_CODE):
        raise FileNotFoundError(f"{ORIGINAL_CODE} not found; run: python validation/ibdd/fetch.py code")

    skimage.measure.compare_mse = _compare_mse
    skimage.measure.compare_ssim = skimage.metrics.structural_similarity
    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = _append
    if not hasattr(pd.Series, "append"):
        pd.Series.append = _append

    spec = importlib.util.spec_from_file_location("ibdd_original_detectors", ORIGINAL_CODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.plot_acc = lambda *args, **kwargs: None
    return module


def run_original_ibdd(train_file, test_file, window_length, consecutive_values, progress=False, desc="original"):
    """Run the original IBDD() on a dataset. Returns a dict with drift_points,
    vet_acc (per-example correctness), mean_acc, execution_time and msd_trace.
    progress shows a tqdm bar over the MSD computations (20 calibration runs,
    then one per stream example)."""
    global _progress
    detectors = load_original()
    train_file, test_file = os.path.abspath(train_file), os.path.abspath(test_file)

    MSD_TRACE.clear()
    if progress:
        with open(test_file) as f:
            n_test = sum(1 for line in f if line.strip())
        _progress = tqdm(total=20 + n_test, desc=desc, unit="ex", mininterval=2)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as workdir:
        os.chdir(workdir)
        try:
            # The original prints a progress line per example; keep it quiet.
            with contextlib.redirect_stdout(io.StringIO()):
                drift_points, vet_acc, mean_acc, execution_time = detectors.IBDD(
                    train_file, test_file, window_length, consecutive_values
                )
        finally:
            os.chdir(cwd)
            if _progress is not None:
                _progress.close()
                _progress = None

    return {
        "drift_points": list(drift_points),
        "vet_acc": vet_acc,
        "mean_acc": mean_acc,
        "execution_time": execution_time,
        "msd_trace": list(MSD_TRACE),
    }
