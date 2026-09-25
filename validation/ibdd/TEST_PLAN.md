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
- every `{base}_{det}` row equals the `{base}` or `{base}_syn` row of its batch;
- the IBDD flag rate per bag (synthetic) and per UPP prevalence bin (real).

## Results

Environment: python 3.13.7, numpy 2.2.2, pandas 2.2.3, scikit-learn 1.6.1, matplotlib 3.10.0, Pillow 11.1.0, scikit-image 0.26.0.

RESULTS_PLACEHOLDER
