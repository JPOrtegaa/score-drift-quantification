"""Shared setup for the CDT regression test and its golden-file generator.

The golden file was captured from the CDT implementation that lived in
methods/quantifiers_utils.py, before it moved to methods/drift_detectors/cdt.py.
Keep this module stable: changing the data or the parameters invalidates
tests/golden/cdt_golden.json.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CDT_PARAMS = dict(sizes=100, repetitions=2, pos_prev=np.linspace(0, 1, 5), measure="topsoe")
SEED = 7


def make_binary_df(n=600, d=4, seed=SEED):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, d)) + y[:, None] * 0.8
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df["class"] = y
    return df


def fit_cdt(cdt_cls):
    """Fit a CDT deterministically. _test_batch samples with the global numpy
    RNG (DataFrame.sample without random_state), so seed it right before fit."""
    cdt = cdt_cls(classifier=LogisticRegression(max_iter=1000, random_state=0), **CDT_PARAMS)
    np.random.seed(SEED)
    cdt.fit(make_binary_df())
    return cdt
