"""Compare this repository's IBDD with the authors' implementation on the IBDD
benchmark datasets (tests T2 and T3 of validation/ibdd/TEST_PLAN.md).

    python validation/ibdd/compare_ibdd.py Yoga UWave        # selected datasets
    python validation/ibdd/compare_ibdd.py --all             # all six benchmarks
    python validation/ibdd/compare_ibdd.py --all --array     # also run the numpy image backend

Needs `python validation/ibdd/fetch.py all` first. For each dataset writes
<out>/<dataset>_trace.csv (MSD and drift flag per step, both implementations;
negative steps are the calibration permutations) and updates <out>/summary.csv.
"""
import argparse
import os
import platform
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from harness import load_split, run_baseline, run_ours  # noqa: E402

DATA_DIR = os.path.join(ROOT, "datasets", "ibdd", "KAIS_IBDD_Datasets", "benchmark_real")
DEFAULT_OUT = os.path.join(ROOT, "results", "ibdd_validation")

# Souza et al. (BigData 2020), Table II: accuracy (%) of the classifier updated
# by IBDD and of the static Baseline, window = |Train|, m = 3.
PAPER = {
    "Heartbeats":      {"ibdd": 98.72, "baseline": 96.67},
    "Insects":         {"ibdd": 95.09, "baseline": 61.29},
    "Posture":         {"ibdd": 55.09, "baseline": 46.26},
    "StarLightCurves": {"ibdd": 91.96, "baseline": 23.22},
    "UWave":           {"ibdd": 55.08, "baseline": 19.29},
    "Yoga":            {"ibdd": 79.83, "baseline": 56.23},
}
# Output of the authors' notebook "How to run IBDD - Yoga example.ipynb".
NOTEBOOK_YOGA = {"drift_points": [1076, 1093, 1144, 1236, 1267, 1322], "mean_acc": 79.83}


def dataset_files(name):
    return (os.path.join(DATA_DIR, f"{name}_TRAIN.data"), os.path.join(DATA_DIR, f"{name}_TEST.data"))


def library_versions():
    import matplotlib, PIL, sklearn, skimage
    return (f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
            f"sklearn {sklearn.__version__}, matplotlib {matplotlib.__version__}, "
            f"Pillow {PIL.__version__}, scikit-image {skimage.__version__}")


def compare(name, consecutive_values, run_original, run_array, out_dir):
    train_file, test_file = dataset_files(name)
    train_X, train_y = load_split(train_file)
    test_X, test_y = load_split(test_file)
    window = len(train_y)  # the paper's window = |Train|
    row = {"dataset": name, "window": window, "m": consecutive_values, "n_test": len(test_y)}
    print(f"\n== {name}: train {train_X.shape}, test {test_X.shape}, w={window}, m={consecutive_values}")

    ours = run_ours(train_X, train_y, test_X, test_y, window, consecutive_values, "jpeg",
                    progress=True, desc=f"{name} ours")
    row.update(ours_drifts=len(ours["drift_points"]), ours_acc=ours["mean_acc"], ours_time=ours["execution_time"])
    print(f"ours:     {len(ours['drift_points'])} drifts, acc {ours['mean_acc']:.2f}%, {ours['execution_time']:.0f}s")
    trace = {"step": np.arange(len(ours["msd_trace"])) - (len(ours["msd_trace"]) - len(test_y)),
             "msd_ours": ours["msd_trace"]}
    trace["drift_ours"] = np.isin(trace["step"], ours["drift_points"])

    if run_original:
        from original_compat import run_original_ibdd
        orig = run_original_ibdd(train_file, test_file, window, consecutive_values,
                                 progress=True, desc=f"{name} original")
        same_trace_length = len(orig["msd_trace"]) == len(ours["msd_trace"])
        row.update(
            orig_drifts=len(orig["drift_points"]), orig_acc=orig["mean_acc"], orig_time=orig["execution_time"],
            same_drift_points=orig["drift_points"] == ours["drift_points"],
            same_vet_acc=bool(np.array_equal(orig["vet_acc"], ours["vet_acc"])),
            max_abs_msd_diff=(float(np.max(np.abs(np.subtract(orig["msd_trace"], ours["msd_trace"]))))
                              if same_trace_length else np.nan),
        )
        print(f"original: {len(orig['drift_points'])} drifts, acc {orig['mean_acc']:.2f}%, {orig['execution_time']:.0f}s")
        print(f"T2 same drift points: {row['same_drift_points']}, same per-example correctness: "
              f"{row['same_vet_acc']}, max |MSD diff|: {row['max_abs_msd_diff']}")
        if same_trace_length:
            trace["msd_original"] = orig["msd_trace"]
        trace["drift_original"] = np.isin(trace["step"], orig["drift_points"])

    if run_array:
        array = run_ours(train_X, train_y, test_X, test_y, window, consecutive_values, "array",
                         progress=True, desc=f"{name} array")
        row.update(array_drifts=len(array["drift_points"]), array_acc=array["mean_acc"])
        print(f"array backend: {len(array['drift_points'])} drifts, acc {array['mean_acc']:.2f}%")

    row["baseline_acc"] = run_baseline(train_X, train_y, test_X, test_y)
    if name in PAPER:
        row.update(paper_acc=PAPER[name]["ibdd"], paper_baseline_acc=PAPER[name]["baseline"])
        print(f"T3 paper: IBDD {PAPER[name]['ibdd']}% (ours {row['ours_acc']:.2f}%), "
              f"baseline {PAPER[name]['baseline']}% (ours {row['baseline_acc']:.2f}%)")
    if name == "Yoga" and consecutive_values == 3:
        row["same_as_notebook"] = ours["drift_points"] == NOTEBOOK_YOGA["drift_points"]
        print(f"T3 notebook: drift points {NOTEBOOK_YOGA['drift_points']} -> ours {ours['drift_points']}")
    row["drift_points_ours"] = " ".join(map(str, ours["drift_points"]))
    row["versions"] = library_versions()

    pd.DataFrame(trace).to_csv(os.path.join(out_dir, f"{name}_trace.csv"), index=False)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help=f"any of {sorted(PAPER)}")
    parser.add_argument("--all", action="store_true", help="run all six benchmark datasets")
    parser.add_argument("-m", "--consecutive-values", type=int, default=3)
    parser.add_argument("--skip-original", action="store_true", help="only run this repository's IBDD")
    parser.add_argument("--array", action="store_true", help="also run the numpy (no JPEG) image backend")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    names = sorted(PAPER) if args.all else args.datasets
    if not names:
        parser.error("name at least one dataset or pass --all")
    os.makedirs(args.out, exist_ok=True)

    summary_path = os.path.join(args.out, "summary.csv")
    for name in names:
        row = compare(name, args.consecutive_values, not args.skip_original, args.array, args.out)
        # One row per (dataset, m); rerunning a dataset replaces its row.
        summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
        if not summary.empty:
            summary = summary[~((summary["dataset"] == name) & (summary["m"] == args.consecutive_values))]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
        summary.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
