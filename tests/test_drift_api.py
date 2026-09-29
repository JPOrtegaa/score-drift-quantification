"""T0: the detector interface shared by the experiments."""
import json
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest

from methods.drift_detectors import BatchContext, CDT, DETECTORS, DriftDetector, IBDD, build_detector


def test_registry_builds_every_detector():
    assert set(DETECTORS) == {"cdt", "ibdd"}
    assert isinstance(build_detector("cdt", measure="hellinger"), CDT)
    assert build_detector("ibdd", window_length=10).window_length == 10
    with pytest.raises(ValueError):
        build_detector("md3")


def test_scope_of_each_detector():
    assert CDT.per_class is True
    assert IBDD.per_class is False
    assert all(issubclass(cls, DriftDetector) for cls in DETECTORS.values())


def test_from_thresholds_and_two_sided_predict():
    detector = CDT.from_thresholds(0.5, 0.1)
    assert detector.predict(0.5) and detector.predict(0.1)
    assert not detector.predict(0.3)
    legacy = CDT.from_thresholds(0.5)
    assert legacy.thr_lower == -np.inf
    assert not legacy.predict(-100.0)


def test_cdt_statistic_is_the_dys_distance():
    from methods.quantifiers import DyS
    rng = np.random.default_rng(0)
    pos, neg, test = rng.beta(5, 2, 200), rng.beta(2, 5, 200), rng.beta(3, 3, 100)
    ctx = BatchContext(test_scores=test, pos_scores=pos, neg_scores=neg)
    _, expected = DyS(pos, neg, test, return_distance=True, measure="topsoe")
    detector = CDT.from_thresholds(expected + 1, expected - 1)
    assert detector.statistic(ctx) == expected
    assert detector.detect(ctx) is False


@pytest.mark.parametrize("dump", [pickle.dumps, lambda obj: pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)])
def test_fitted_detectors_survive_pickling(dump):
    rng = np.random.default_rng(0)
    ibdd = IBDD(window_length=50).fit(rng.normal(size=(300, 8)))
    cdt = CDT.from_thresholds(0.5, 0.1)
    ctx = BatchContext(X=pd.DataFrame(rng.normal(size=(50, 8))))

    ibdd_copy, cdt_copy = pickle.loads(dump(ibdd)), pickle.loads(dump(cdt))
    assert ibdd_copy.statistic(ctx) == ibdd.statistic(ctx)
    assert (ibdd_copy.thr_lower, ibdd_copy.thr_upper) == (ibdd.thr_lower, ibdd.thr_upper)
    assert (cdt_copy.thr_lower, cdt_copy.thr_upper) == (0.1, 0.5)


def test_detectors_run_in_joblib_workers():
    rng = np.random.default_rng(0)
    ibdd = IBDD(window_length=50).fit(rng.normal(size=(300, 8)))
    batches = [rng.normal(size=(50, 8)) for _ in range(4)]
    parallel = joblib.Parallel(n_jobs=2, backend="loky")(joblib.delayed(ibdd.statistic)(b) for b in batches)
    assert parallel == [ibdd.statistic(b) for b in batches]


def test_save_distances_schema(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "out" / "distances.csv"
    first = IBDD(window_length=20).fit(rng.normal(size=(100, 4)))
    second = IBDD(window_length=20).fit(rng.normal(size=(100, 4)))
    first.save_distances(str(path), model_id="a", overwrite=True)
    second.save_distances(str(path), model_id="b")

    df = pd.read_csv(path)
    assert list(df.columns) == ["model_id", "thr_lower", "thr_upper", "distances"]
    assert df["model_id"].tolist() == ["a", "b"]
    assert json.loads(df["distances"][0]) == first.distances.tolist()

    with pytest.raises(ValueError):
        CDT().save_distances(str(path))


def test_experiment_trace_keeps_the_detect_decision():
    # ovr.evaluate_detectors splits detect() into statistic + predict so the
    # statistic can be traced; the flag must stay the one detect() returns.
    import ovr
    rng = np.random.default_rng(0)
    ibdd = IBDD(window_length=50).fit(rng.normal(size=(300, 8)))
    cdt = CDT.from_thresholds(0.02, 0.001)
    for shift in (0.0, 3.0):
        ctx = BatchContext(X=pd.DataFrame(rng.normal(loc=shift, size=(50, 8))),
                           test_scores=rng.beta(3, 3, 50), pos_scores=rng.beta(5, 2, 200), neg_scores=rng.beta(2, 5, 200))
        detectors = {"ibdd": ibdd, "cdt": cdt}
        flags, rows = ovr.evaluate_detectors(detectors, ctx, model_id="m")
        assert flags == {name: d.detect(ctx) for name, d in detectors.items()}
        assert [(r["detector"], r["model_id"]) for r in rows] == [("ibdd", "m"), ("cdt", "m")]
        for row in rows:
            detector = detectors[row["detector"]]
            assert row["statistic"] == detector.statistic(ctx)
            assert (row["thr_lower"], row["thr_upper"], row["drift"]) == (detector.thr_lower, detector.thr_upper, flags[row["detector"]])


def test_cdt_fit_is_reproducible():
    # The calibration bags come from one seeded generator per fit: the same
    # random_state gives the same distances, a different one other bags.
    from sklearn.linear_model import LogisticRegression
    from cdt_fixture import CDT_PARAMS, make_binary_df

    def fit(random_state):
        cdt = CDT(classifier=LogisticRegression(max_iter=1000, random_state=0), random_state=random_state, **CDT_PARAMS)
        return cdt.fit(make_binary_df())

    first, second = fit(42), fit(42)
    assert first.distances.tolist() == second.distances.tolist()
    assert (first.thr_lower, first.thr_upper) == (second.thr_lower, second.thr_upper)
    assert fit(0).distances.tolist() != first.distances.tolist()
    # Bags within one fit are still distinct draws (not one bag reused).
    per_prev = first.distances.reshape(CDT_PARAMS["repetitions"], -1)
    assert not np.array_equal(per_prev[0], per_prev[1])
