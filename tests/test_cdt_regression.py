"""T1: CDT behaves as it did in methods/quantifiers_utils.py before the move to
methods/drift_detectors (golden values captured from that version, on Windows)."""
import json
import os

import pytest

from cdt_fixture import fit_cdt
from methods.drift_detectors import CDT

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "cdt_golden.json")

# The golden values are bit-exact on the machine that captured them; another
# platform's math libraries round the last digits differently (~1e-18 here),
# so compare within a relative tolerance far below any real change.
RTOL = 1e-12


def test_cdt_matches_the_pre_refactor_golden_values():
    with open(GOLDEN) as f:
        golden = json.load(f)
    cdt = fit_cdt(CDT)
    assert cdt.thr_lower == pytest.approx(golden["thr_lower"], rel=RTOL)
    assert cdt.thr_upper == pytest.approx(golden["thr_upper"], rel=RTOL)
    assert [float(d) for d in cdt.distances] == pytest.approx(golden["distances"], rel=RTOL)
