"""Drift detectors that decide, per test batch, whether QuaDapt should replace
the training scores with synthetic ones.

Every detector follows DriftDetector (base.py): fit(train) on the training data,
then detect(BatchContext) -> bool for each test batch. Add a new detector by
subclassing DriftDetector and registering it in DETECTORS below; the experiment
then exposes it through the {base}_{name} gated quantifiers.
"""
from .base import BatchContext, DriftDetector
from .cdt import CDT
from .ibdd import IBDD

DETECTORS = {cls.name: cls for cls in (CDT, IBDD)}


def build_detector(name, **kwargs):
    if name not in DETECTORS:
        raise ValueError(f"Unknown drift detector {name!r}; available: {sorted(DETECTORS)}")
    return DETECTORS[name](**kwargs)


__all__ = ["BatchContext", "DriftDetector", "CDT", "IBDD", "DETECTORS", "build_detector"]
