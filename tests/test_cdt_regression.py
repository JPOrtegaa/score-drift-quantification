"""T1: CDT behaves exactly as it did in methods/quantifiers_utils.py before the
move to methods/drift_detectors (golden values captured from that version)."""
import json
import os

from cdt_fixture import fit_cdt
from methods.drift_detectors import CDT

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "cdt_golden.json")


def test_cdt_matches_the_pre_refactor_golden_values():
    with open(GOLDEN) as f:
        golden = json.load(f)
    cdt = fit_cdt(CDT)
    assert cdt.thr_lower == golden["thr_lower"]
    assert cdt.thr_upper == golden["thr_upper"]
    assert [float(d) for d in cdt.distances] == golden["distances"]
