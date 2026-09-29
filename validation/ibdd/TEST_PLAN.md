# IBDD validation test plan

This plan checks the IBDD implementation in `methods/drift_detectors/ibdd.py` against two references:
- the authors' code, `detectors.py` from the IBDD site;
- the published results in Souza, Chowdhury & Mueen, *Unsupervised Drift Detection on High-speed Data Streams*, IEEE BigData 2020.

It also checks that moving CDT out of `methods/quantifiers_utils.py` changed nothing, and that both detectors gate QuaDapt correctly in `ovr.py`.

Sources:
- IBDD site: https://sites.google.com/view/ibdd-paper
- Authors' code and datasets: `python validation/ibdd/fetch.py all`. They are downloaded, not committed. The code goes to `validation/ibdd/_external/` and the datasets to `datasets/ibdd/`.
- Dev dependencies: `pip install -r requirements-dev.txt` (pytest, and scikit-image to run the original).

## What the original does

The reference points in `detectors.py` are:
- **Image.** `plt.imsave(file.jpeg, window.T, cmap='Greys')`, then `skimage.io.imread`, then `compare_mse` for the MSD. The window is min-max normalized as a whole, run through the Greys colormap, and saved as a lossy JPEG.
- **Initial thresholds.** The reference is `X_train[-w:]`. There are 20 runs of `random.seed(i); shuffle(sequence)` (a cumulative, in-place shuffle). The thresholds are mean ± 2·std, and the lower one is clipped at 0.
- **Stream.**
  - The MSD history is seeded with the 20 calibration values.
  - Every 60 examples without a drift, the thresholds become mean ± 2·std of the last 50 MSDs.
  - A drift is flagged when *m* consecutive MSDs are beyond a threshold. The thresholds then jump:
    - `upper = last + std(last 49)`
    - `lower = last − mean(threshold gaps)`
    - and the mirror image for a drift below the lower threshold.
  - After a drift, the model is retrained on the window.
- **Evaluation.**
  - `RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)`.
  - Prequential accuracy.
  - w = |Train|, m = 3.

Our `IBDD` does the same image pipeline in memory (`image_backend="jpeg"`, the default). `start_stream()` + `update()` is a line-by-line port of the stream loop. `fit()` + `detect()` is the batch mode used to gate QuaDapt: fixed thresholds, one comparison per test batch, and each batch shuffled with a fixed seed so that the order of rows inside a bag doesn't matter (see T4).

## T0 – Unit tests (fast, no external data)

`pytest -m "not slow"`: `tests/test_ibdd_unit.py`, `tests/test_drift_api.py`

**Images**
- In-memory JPEG == `plt.imsave` to disk + `skimage.io.imread`.
- Shape `(features, examples, 3)`, dtype uint8.
- The array backend lies in [0, 1].

**MSD**
- 0 for identical images.
- Symmetric.
- == `skimage.metrics.mean_squared_error`.

**Calibration**
- == an independent recomputation using the global `random.seed/shuffle`.
- Lower threshold clipped at 0.
- Window capped at |Train|.
- The `class` column is ignored.

**Batch mode**
- ≤ 10% false alarms on stationary curves.
- 100% detection of sign flips on both backends.
- Class-sorted bags don't trigger false alarms (shuffle on); with the shuffle off they do (≥ 90%).
- A wrong batch size raises an error.

**Stream mode**
- An abrupt flip is detected within 30 examples.
- At most 5 false alarms before the flip.

**API**
- Registry.
- `per_class` scope.
- `from_thresholds` and the two-sided `predict`.
- CDT statistic == DyS distance.
- Pickle and joblib round-trips.
- `save_distances` schema.

## T1 – CDT refactor regression

- `tests/test_cdt_regression.py`: the moved CDT reproduces, bit for bit, the thresholds and distances captured from the pre-move version (`tests/golden/cdt_golden.json`, generated through `tests/cdt_fixture.py`).
- End to end: seeded `ovr.process_single_dataset("datasets/synthetic/label_shift_5,5.csv")` with CDT on, before (`RUN_CDT=True`) and after (`DRIFT_DETECTORS=["cdt"]`) the refactor. `distances.csv` and `label_shift_5,5_results.csv` must be byte-identical.

## T2 – Equivalence with the original implementation (hard gate)

**How the original is run.** `validation/ibdd/original_compat.py` runs the authors' `detectors.py` **unmodified**:
- `compare_mse` is aliased to `skimage.metrics.mean_squared_error`, which also records the full MSD trace;
- `DataFrame.append` and `Series.append` are rebuilt on `pd.concat`;
- `plot_acc` is a no-op;
- it runs in a temporary directory.

`validation/ibdd/harness.py` runs our IBDD under the same protocol.

**Pass criteria for each dataset (w = |Train|, m = 3, jpeg backend):**
1. identical drift points;
2. identical per-example correctness vector;
3. MSD traces equal within 1e-9 (calibration + every stream step).

**Where it runs:**
- `pytest -m slow`: Yoga, UWave, StarLightCurves.
- `python validation/ibdd/compare_ibdd.py --all --array`: all six. This writes `results/ibdd_validation/summary.csv` and `<dataset>_trace.csv`.

**Informational only.** `--array` also reports the lossless numpy backend.

## T3 – Reproduction of the published numbers

- **Yoga:** the authors' notebook output, 6 drifts at `[1076, 1093, 1144, 1236, 1267, 1322]` with 79.83% accuracy.
- **All six datasets:** Table II of the paper (IBDD and Baseline accuracies). Expect agreement within ±0.5 pp. The datasets are the KAIS journal release, and library versions differ from 2020, so small deviations are possible even when T2 passes.

## T4 – QuaDapt integration (`ovr.py`)

Run `ovr.py` with `DRIFT_DETECTORS=["cdt","ibdd"]` on three datasets:
- `datasets/synthetic/global_covariate_shift2.csv`: temporal bags of 1000, 2 features;
- `datasets/ours/Mfeat_icdm21.csv`: UPP batches of 100;
- `datasets/kaggle/IRIS.csv`: 75 training rows, fewer than one batch, so IBDD must be skipped with a warning and no `*_ibdd` rows written.

**Checks:**
- every class prediction of a `{base}_{det}` row equals that class's prediction in the `{base}` or `{base}_syn` row of its batch. The check is per class because with One-vs-Rest each class has its own binary model:
  - a dataset-level detector (IBDD) flags every class at once, so its rows must also match as a whole;
  - a `per_class` detector (CDT) flags each class separately, so on multiclass data a row can mix base and synthetic values;
- the IBDD flag rate per bag (synthetic) and per UPP prevalence bin (real).

## Results

Environment:
- T2–T4 ran on sabethes, quadapt conda env: python 3.11.2, numpy 2.2.2, pandas 2.2.3, scikit-learn 1.6.1, matplotlib 3.10.0, Pillow 11.1.0, scikit-image 0.26.0.
- T0–T1 ran in the repo `venv`: python 3.13.5, same package versions.

Outputs are in `results/ibdd_validation/` (git-ignored): `<dataset>/summary.csv`, `<dataset>/<dataset>_trace.csv` and `run.log` for T2/T3, `integration/` for T4. One tmux window per dataset (`launch.sh`), run on 2026-09-25.

### T0 – Unit tests: PASS

`pytest -m "not slow"`: 27 passed, 3 deselected (the slow ones).

### T1 – CDT refactor regression: golden values PASS, end to end NOT RUN

- `tests/test_cdt_regression.py` passes, but only within `rel=1e-12`. The golden values were captured on Windows. On Linux the distances differ in the last digits (about 1.7e-17 on the first distance) because the math libraries round differently. The thresholds and every distance agree to 1e-12.
- The end-to-end byte-identical comparison (`RUN_CDT=True` before the refactor vs `DRIFT_DETECTORS=["cdt"]` after it, on `label_shift_5,5`) has not been run.

### T2 – Equivalence with the original: PASS on all six

w = |Train|, m = 3, jpeg backend. Times are wall-clock seconds on one pinned core.

| Dataset | w | Test examples | Same drift points | Same correctness vector | Max \|ΔMSD\| | Drifts | Time ours / original |
|---|---|---|---|---|---|---|---|
| Yoga | 300 | 3,000 | yes | yes | 0.0 | 6 | 9 / 27 |
| UWave | 896 | 3,582 | yes | yes | 0.0 | 36 | 43 / 78 |
| StarLightCurves | 1000 | 8,236 | yes | yes | 0.0 | 67 | 370 / 376 |
| Posture | 2000 | 162,860 | yes | yes | 0.0 | 186 | 106 / 743 |
| Heartbeats | 500 | 77,404 | yes | yes | 0.0 | 1010 | 356 / 722 |
| Insects | 1000 | 85,400 | yes | yes | 0.0 | 696 | 626 / 940 |

- The MSD traces are identical, not just within 1e-9.
- `pytest -m slow` was not run separately. It checks the same three criteria on Yoga, UWave and StarLightCurves, which the table above already covers.

### T3 – Reproduction of the published numbers: PASS

Accuracies in %. The largest deviation from the paper is 0.01 pp.

| Dataset | IBDD (ours) | IBDD (paper) | Baseline (ours) | Baseline (paper) | Array backend: drifts / accuracy |
|---|---|---|---|---|---|
| Yoga | 79.83 | 79.83 | 56.23 | 56.23 | 6 / 76.97 |
| UWave | 55.08 | 55.08 | 19.29 | 19.29 | 33 / 54.08 |
| StarLightCurves | 91.96 | 91.96 | 23.22 | 23.22 | 67 / 92.03 |
| Posture | 55.09 | 55.09 | 46.26 | 46.26 | 199 / 55.14 |
| Heartbeats | 98.71 | 98.72 | 96.67 | 96.67 | 974 / 98.72 |
| Insects | 95.09 | 95.09 | 61.28 | 61.29 | 619 / 95.12 |

- **Yoga:** the drift points `[1076, 1093, 1144, 1236, 1267, 1322]` and the 79.83% accuracy are the notebook's exactly.
- **Array backend (informational):** accuracy is within 0.1 pp of the jpeg backend on four datasets, and lower on UWave (−1.0 pp) and Yoga (−2.9 pp, with the same number of drifts). The drift counts move by up to 11% (Insects 619 vs 696).

### T4 – QuaDapt integration: PASS

| Dataset | CDT: class predictions from base or syn | CDT: rows mixing base and syn | IBDD: class predictions from base or syn | IBDD: whole rows from base or syn | Run time |
|---|---|---|---|---|---|
| global_covariate_shift2 (binary) | 780/780 | 0/390 | 780/780 | 390/390 | 85 s |
| Mfeat_icdm21 (3 classes) | 30000/30000 | 2382/10000 | 30000/30000 | 10000/10000 | 2272 s |
| IRIS (3 classes) | 30000/30000 | 88/10000 | skipped | skipped | 2263 s |

- **IRIS:** IBDD is skipped with `WARNING: skipping IBDD, 75 training rows < window of 100`, and no `*_ibdd` rows are written.
- **Mixed rows (CDT):** these are batches where the per-class CDTs disagreed about drift. For example, on Mfeat batch 10, `DyS_cdt` takes the synthetic value for c4 and the base value for c2 and c5.
- **global_covariate_shift2:**
  - thresholds [3393.3, 4866.9], w = 1000;
  - flag rate 0/39 bags;
  - bag MSDs range from 3650 to 4731, all inside the thresholds.
  - The covariate shift in this dataset does not reach IBDD's ±2σ calibration band.
- **Mfeat_icdm21:**
  - thresholds [3846.6, 7820.8], w = 100;
  - flag rate 6.1% (61/1000 batches);
  - over the 100 UPP prevalence points (10 batches each), 60 points have no flag and the highest per-point rate is 0.4.
  - The flag rate grows with the L1 distance between the batch prevalence and the training prevalence (0.375, 0.375, 0.25), with correlation 0.32. The rows below are grouped by that distance:

| L1 distance to training prevalence | Points | Flag rate | Mean MSD |
|---|---|---|---|
| ≤ 0.25 | 9 | 0.033 | 5931 |
| 0.25–0.5 | 24 | 0.029 | 5887 |
| 0.5–0.75 | 44 | 0.057 | 5890 |
| 0.75–1.0 | 14 | 0.079 | 5934 |
| > 1.0 | 9 | 0.167 | 6325 |
