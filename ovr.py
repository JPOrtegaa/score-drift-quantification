import pandas as pd
import numpy as np
import sys
import os
import json
import warnings

# Suppress all warnings from sklearn
warnings.filterwarnings('ignore')

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

from mlquantify.model_selection import UPP
from mlquantify.adjust_counting import AC, PAC, FM
from mlquantify.neighbors import KDEyHD, KDEyCS, KDEyML
from mlquantify.likelihood import EMQ
from mlquantify.mixture import HDx
from mlquantify.neighbors import PWK

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import pdb

from tqdm import tqdm
from joblib import Parallel, delayed

# Import preprocessing functions from different dataset folders
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'datasets', 'kaggle'))
import datasets.kaggle.preprocess as kaggle_preprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'datasets', 'openml'))
import datasets.openml.preprocess as openml_preprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'datasets', 'ours'))
import datasets.ours.preprocess as ours_preprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'datasets', 'schumacher'))
import datasets.schumacher.preprocess as schumacher_preprocess

from methods.quantifiers import (
    CC,
    CC2,
    ACC,
    PCC,
    PACC,
    T50,
    MAX,
    MS,
    MS2,
    SMM,
    DyS,
    DySyn,
    HDy,
    X,
)
from methods.quadapt import (
    ACCSyn,
    PACCSyn,
    XSyn,
    MAXSyn,
    T50Syn,
    MSSyn,
    MS2Syn,
    SMMSyn,
    HDySyn,
)
from methods.quantifiers_utils import getTPRandFPRbyThreshold, CDT
import math
import sys
import argparse

# Root folder for all per-dataset output (results CSV, distances.csv,
# test_scores/, distributions). Each dataset gets its own subfolder underneath.
RESULTS_ROOT = "ovr_results_corrected_topsoe_binrange"

# Toggle persistence of score distributions: training distributions, per-batch
# test scores, and the DySyn selected distributions (the <RESULTS_ROOT>/<dataset>/
# {test_scores,<class>,multiclass} file structure). Set to False to skip writing
# them; the final <dataset>_results.csv is always written.
SAVE_DISTRIBUTIONS = False

# Toggle the whole CDT pipeline. When False no CDT is trained or loaded and the
# CDT-gated quantifiers ({base}_cdt) are dropped from the binary list, so a run
# only exercises the base and synthetic quantifiers.
RUN_CDT = False

# Toggle the CDT threshold source. When True, reuse the pre-computed CDT
# thresholds from an existing <RESULTS_ROOT>/<dataset>/distances.csv instead of
# training a new CDT per binary classifier (no distances file is rewritten).
# When False, train a fresh CDT per binary classifier and persist its DyS
# distances/thresholds to distances.csv. The distances file carries two-sided
# thresholds (thr_lower/thr_upper); a legacy file with a single `thr` column is
# read as the upper bound only (lower = None).
USE_PRECOMPUTED_CDT_DISTANCES = False

# Toggle the multiclass (aggregate) quantifiers. When False the multiclass list
# is emptied and none of the machinery behind it runs: no PWK/HDx fit, no extra
# multiclass classifier with its own 10-fold cross-validation, no per-batch
# multiclass prediction. Only the binary and synthetic quantifiers are evaluated,
# and the results CSV simply has no GAC/GPAC/EMQ/KDEy*/FM/HDx/PWK/CC2 rows.
RUN_MULTICLASS = False

# Distance measure used by every synthetic (MoSS-based) quantifier: DySyn and
# the {base}_syn family. Keep it in one place so the standalone DySyn call in
# test_classifier cannot drift away from the quantifier map.
SYN_MEASURE = "topsoe"

# Index of the last training bag for the synthetic datasets (see
# datasets/synthetic/). Those files carry their own temporal batching in the `t`
# column: 50 bags at t = linspace(0, 1, 50). Bags 0..SYNTHETIC_TRAIN_LAST_BAG
# (inclusive) form the training set, every later bag becomes one test batch.
SYNTHETIC_TRAIN_LAST_BAG = 10

def serialize_scores(scores):
    if scores is None:
        return ""

    return json.dumps(np.asarray(scores).tolist())

def sanitize_model_name(model_id):
    return str(model_id).replace(os.sep, "_").replace("/", "_").replace(" ", "_")

def build_dataset_output_dirs(dataset_name):
    dataset_dir = os.path.join(".", RESULTS_ROOT, dataset_name)
    test_scores_dir = os.path.join(dataset_dir, "test_scores")
    os.makedirs(dataset_dir, exist_ok=True)
    if SAVE_DISTRIBUTIONS:
        os.makedirs(test_scores_dir, exist_ok=True)
    return dataset_dir, test_scores_dir

def persist_training_distributions(dataset_dir, classifiers, validation_scores):
    if not SAVE_DISTRIBUTIONS:
        return
    for cls, model_data in classifiers.items():
        class_dir = os.path.join(dataset_dir, sanitize_model_name(cls))
        os.makedirs(class_dir, exist_ok=True)
        training_row = {
            "model_id": cls,
            "pos_scores": serialize_scores(model_data["pos_scores"]),
            "neg_scores": serialize_scores(model_data["neg_scores"]),
        }
        pd.DataFrame([training_row]).to_csv(os.path.join(class_dir, "training_distributions.csv"), index=False)

    multiclass_dir = os.path.join(dataset_dir, "multiclass")
    os.makedirs(multiclass_dir, exist_ok=True)
    multiclass_row = {
        "model_id": "multiclass",
        "training_scores": serialize_scores(validation_scores),
    }
    pd.DataFrame([multiclass_row]).to_csv(os.path.join(multiclass_dir, "training_distributions.csv"), index=False)

def persist_batch_scores(test_scores_dir, batch_index, model_id, incoming_test_scores, selected_p_scores=None, selected_n_scores=None):
    if not SAVE_DISTRIBUTIONS:
        return
    file_name = f"batch_{batch_index:04d}_{sanitize_model_name(model_id)}.csv"
    file_path = os.path.join(test_scores_dir, file_name)
    row = {
        "batch_index": batch_index,
        "model_id": model_id,
        "incoming_test_scores": serialize_scores(incoming_test_scores),
        "selected_p_scores": serialize_scores(selected_p_scores),
        "selected_n_scores": serialize_scores(selected_n_scores),
    }
    pd.DataFrame([row]).to_csv(file_path, index=False)

def get_quantifier_map(test_scores, tpr_fpr, pos_scores, neg_scores):
    """Returns a dictionary mapping quantifier names to their callable functions."""
    measure = SYN_MEASURE
    return {
        "CC": lambda: CC(test_scores, thr=0.5),
        "PCC": lambda: PCC(test_scores),
        "ACC": lambda: ACC(test_scores, tpr_fpr, thr=0.5),
        "PACC": lambda: PACC(test_scores, pos_scores, neg_scores),
        "T50": lambda: T50(test_scores, tpr_fpr),
        "MAX": lambda: MAX(test_scores, tpr_fpr),
        "MS": lambda: MS(test_scores, tpr_fpr),
        "MS2": lambda: MS2(test_scores, tpr_fpr),
        "X": lambda: X(test_scores, tpr_fpr),
        "SMM": lambda: SMM(pos_scores, neg_scores, test_scores),
        "DyS": lambda: DyS(pos_scores, neg_scores, test_scores),
        "DySyn": lambda: DySyn(test_scores, measure=measure)[0],
        "HDy": lambda: HDy(pos_scores, neg_scores, test_scores)[0],
        "ACC_syn": lambda: ACCSyn(test_scores, measure=measure, MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "PACC_syn": lambda: PACCSyn(test_scores, measure=measure, MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "X_syn": lambda: XSyn(test_scores, measure=measure, MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "MAX_syn": lambda: MAXSyn(test_scores, measure=measure, MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "T50_syn": lambda: T50Syn(test_scores, measure=measure, MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "MS_syn": lambda: MSSyn(test_scores, measure=measure, MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "MS2_syn": lambda: MS2Syn(test_scores, measure=measure, MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "SMM_syn": lambda: SMMSyn(test_scores, measure=measure, MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "HDy_syn": lambda: HDySyn(test_scores, MF=np.arange(0.1, 1.0, 0.2))[0],
    }


def get_multiclass_quantifier_map(y_train, X_test, priors, posteriors, qnt_models, classifier):
    """Returns a dictionary mapping multiclass quantifier names to their callable functions."""

    # Initialize quantifiers that use aggregate pattern
    gac = AC()
    gpac = PAC()
    emq = EMQ(max_iter=2000)
    kde = KDEyHD()
    kde_cs = KDEyCS()
    kde_ml = KDEyML()
    fm = FM()
    hdx = qnt_models["HDx"]
    pwk = qnt_models["PWK"]

    # Get hard predictions for GAC and map to actual class labels
    train_pred_indices = np.argmax(priors, axis=1)
    test_pred_indices = np.argmax(posteriors, axis=1)
    
    # Map indices to actual class labels using classifier's classes_
    train_preds = classifier.classes_[train_pred_indices]
    test_preds = classifier.classes_[test_pred_indices]

    return {
        "GAC": lambda: gac.aggregate(train_predictions=train_preds, predictions=test_preds, y_train=y_train),
        "GPAC": lambda: gpac.aggregate(train_predictions=priors, predictions=posteriors, y_train=y_train),
        "EMQ": lambda: emq.aggregate(predictions=posteriors, y_train=y_train),
        "KDEyHD": lambda: kde.aggregate(train_predictions=priors, predictions=posteriors, y_train=y_train),
        "KDEyCS": lambda: kde_cs.aggregate(train_predictions=priors, predictions=posteriors, y_train=y_train),
        "KDEyML": lambda: kde_ml.aggregate(train_predictions=priors, predictions=posteriors, y_train=y_train),
        "FM": lambda: fm.aggregate(train_predictions=priors, predictions=posteriors, y_train=y_train),
        "HDx": lambda: hdx.predict(X=X_test),
        "PWK": lambda: pwk.predict(X=X_test),
        "CC2": lambda: CC2(posteriors),
    }

def scale_dataset(df, scaler=None):
    df_scaled = df.copy()
    
    # Get feature columns (all except 'class')
    feature_cols = [col for col in df_scaled.columns if col != 'class']
    
    # Check if scaling is needed (if data is not already scaled)
    needs_scaling = False

    # More lenient check - consider range and variance
    if scaler is None:  # Only check on training data
        for col in feature_cols:
            col_range = df_scaled[col].max() - df_scaled[col].min()
            col_std = df_scaled[col].std()
            # If range > 10 or std differs significantly from 1, scale it
            if col_range > 10 or abs(col_std - 1.0) > 0.5:
                needs_scaling = True
                break

    if needs_scaling or scaler is not None:
        if scaler is None:
            scaler = StandardScaler()
            df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
        else:
            # If scaler is provided, use it
            df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])
    
    scaled_flag = needs_scaling or (scaler is not None)

    return df_scaled, scaler, scaled_flag

def binarize_dataset(train_df):
    trains = {}

    classes = train_df['class'].unique()

    # Already-binary dataset: One-vs-Rest is unnecessary, a single classifier
    # for the positive class carries all the information. The caller reads the
    # positive prevalence directly and derives the negative one as its
    # complement.
    if len(classes) == 2:
        pos_cls = max(classes)
        bin_train_df = train_df.copy()
        bin_train_df['class'] = (bin_train_df['class'] == pos_cls).astype(int)
        return {pos_cls: bin_train_df}

    for cls in classes:
        bin_train_df = train_df.copy()
        bin_train_df['class'] = (bin_train_df['class'] == cls).astype(int)
        trains[cls] = bin_train_df

    return trains

def train_quantifiers(train_df):

    pwk = PWK(n_neighbors=11, n_jobs=-1)
    hdx = HDx(bins_size=np.linspace(10, 110, 11))

    X_train, y_train = train_df.drop(columns=['class']), train_df['class']
    pwk.fit(X_train, y_train)
    hdx.fit(X_train, y_train)

    models = {
        "PWK": pwk,
        "HDx": hdx,
    }

    return models

def train_classifier(train_df, fit_cdt=True):
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class']

    # CDT calculation
    cdt = None
    if fit_cdt and len(y_train.unique()) == 2:  # Binary only
        cdt = CDT(classifier=RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), measure="topsoe")
        cdt.fit(train_df)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    clf = LogisticRegression(random_state=42, max_iter=2000, n_jobs=-1)
    
    fold_scores = []
    fold_labels = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        clf.fit(X_fold_train, y_fold_train)
        proba = clf.predict_proba(X_fold_val)
        fold_scores.append(np.column_stack((proba, y_fold_val)))
        fold_labels.append(y_fold_val.to_numpy())

    clf.fit(X_train, y_train)
    validation_scores = np.vstack(fold_scores)
    # Labels in the same (shuffled) row order as validation_scores. The folds are
    # concatenated in split order, which is a permutation of train_df, so the
    # quantifiers that pair each validation prediction with its true label must
    # use these and not train_df['class'].
    y_validation = np.concatenate(fold_labels)

    if len(y_train.unique()) > 2: # Multiclass
        validation_scores = validation_scores[:, :-1]  # Remove last column (labels)
        tpr_fpr = None
        pos_scores = None
        neg_scores = None
        
    else: # Binary
        tpr_fpr = getTPRandFPRbyThreshold(validation_scores)
        pos_scores = validation_scores[validation_scores[:, 2] == 1, 1].astype(float)
        neg_scores = validation_scores[validation_scores[:, 2] == 0, 1].astype(float)

    return clf, tpr_fpr, pos_scores, neg_scores, validation_scores, cdt, y_validation

def train_one_vs_rest_classifiers(trains, dataset_dir=None):
    classifiers = {}

    distances_path = os.path.join(dataset_dir, "distances.csv") if dataset_dir else None

    # When reusing pre-computed thresholds, load them from distances.csv up
    # front (no distances file is rewritten). The file carries two-sided
    # thresholds (thr_lower/thr_upper); a legacy file with a single `thr` column
    # is read as the upper bound only (lower = None).
    thr_by_model = {}
    if RUN_CDT and USE_PRECOMPUTED_CDT_DISTANCES and distances_path is not None and os.path.exists(distances_path):
        thr_df = pd.read_csv(distances_path, usecols=lambda c: c != "distances")
        has_two_sided = "thr_upper" in thr_df.columns
        for r in thr_df.itertuples(index=False):
            model = str(r.model_id)
            if has_two_sided:
                lower = getattr(r, "thr_lower", None)
                thr_by_model[model] = {
                    "lower": None if lower is None or pd.isna(lower) else float(lower),
                    "upper": float(r.thr_upper),
                }
            else:
                thr_by_model[model] = {"lower": None, "upper": float(r.thr)}

    first_saved = True
    for cls, bin_train_df in trains.items():
        # Only train a fresh CDT when we are not reusing pre-computed thresholds.
        fit_cdt = RUN_CDT and not USE_PRECOMPUTED_CDT_DISTANCES
        clf, tpr_fpr, pos_scores, neg_scores, _, cdt, _ = train_classifier(bin_train_df, fit_cdt=fit_cdt)

        if not RUN_CDT:
            cdt_thr = None
        elif USE_PRECOMPUTED_CDT_DISTANCES:
            cdt_thr = thr_by_model.get(str(cls), None)
        elif cdt is not None:
            cdt_thr = {"lower": cdt.thr_lower, "upper": cdt.thr_upper}
            # Persist the freshly-computed distances/thresholds so they can be
            # reused on a later run. Overwrite for the first classifier of the
            # dataset, then append one row per subsequent classifier.
            if distances_path is not None:
                cdt.save_distances(distances_path, model_id=cls, overwrite=first_saved)
                first_saved = False
        else:
            cdt_thr = None

        classifiers[cls] = {
            "model": clf,
            "tpr_fpr": tpr_fpr,
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "cdt_thr": cdt_thr,
        }

    return classifiers

def test_classifier(batch, classifier, quantifiers, validation_scores=None, qnt_models=None, y_validation=None, batch_index=None, model_id=None):
    # The binary quantifiers return [positive_prevalence, 1 - positive]; index 0
    # (the positive prevalence) is always stored, so the value is the prevalence
    # of the class the classifier treats as positive -- the same class whose true
    # prevalence is written to c{cls}_real.

    if isinstance(classifier, dict): # Binary
        clf = classifier["model"]
        tpr_fpr = classifier["tpr_fpr"]
        pos_scores = classifier["pos_scores"]
        neg_scores = classifier["neg_scores"]
        cdt_thr = classifier.get("cdt_thr", None)

        X_test = batch.drop(columns=['class'])
        test_scores = clf.predict_proba(X_test)[:, 1]

        # Get quantifier map with all necessary parameters
        quantifier_map = get_quantifier_map(test_scores, tpr_fpr, pos_scores, neg_scores)

        results = {}
        selected_p_scores = None
        selected_n_scores = None

        # CDT drift detector (binary only): set up once with the trained
        # thresholds and the DyS distance of this batch (reused for every
        # synthetic quantifier). cdt_thr carries the two-sided bounds; a missing
        # lower bound (legacy files) becomes -inf so only the upper bound gates.
        cdt = None
        dys_distance = None
        if cdt_thr is not None:
            cdt = CDT(classifier=None)
            cdt.thr_upper = cdt_thr["upper"]
            cdt.thr_lower = cdt_thr["lower"] if cdt_thr["lower"] is not None else -np.inf
            _, dys_distance = DyS(pos_scores, neg_scores, test_scores, return_distance=True, measure="topsoe")

        for q in quantifiers:
            # CDT-gated variant ({base}_cdt): drift (distance outside
            # [thr_lower, thr_upper]) -> synthetic version, otherwise fall back
            # to the base (non-synthetic) quantifier.
            if q.endswith("_cdt"):
                base_q = q[:-4]
                syn_q = "DySyn" if base_q == "DyS" else f"{base_q}_syn"
                if cdt is None:
                    raise ValueError(f"CDT threshold unavailable for gated quantifier: {q}")
                chosen = syn_q if cdt.predict(dys_distance) else base_q
                if chosen not in quantifier_map:
                    raise ValueError(f"Unknown quantifier for CDT gating: {chosen}")
                # Reuse the already-computed prevalence when available, otherwise
                # compute it on demand so this does not depend on list ordering.
                results[q] = results[chosen] if chosen in results else quantifier_map[chosen]()[0]
                continue

            if q not in quantifier_map:
                raise ValueError(f"Unknown quantifier: {q}")
            if q == "DySyn":
                qnt_result = DySyn(
                    test_scores,
                    measure=SYN_MEASURE,
                    write_distribution=False,
                    return_metadata=True,
                )
                results[q] = qnt_result[0][0]
                selected_p_scores = qnt_result[3]["selected_p_scores"]
                selected_n_scores = qnt_result[3]["selected_n_scores"]
            else:
                results[q] = quantifier_map[q]()[0]

        metadata = {
            "batch_index": batch_index,
            "model_id": model_id,
            "incoming_test_scores": test_scores,
            "selected_p_scores": selected_p_scores,
            "selected_n_scores": selected_n_scores,
        }

    else: # Multiclass
        # y_validation holds the training labels in the same row order as the
        # cross-validated scores in validation_scores; the aggregate-style
        # quantifiers pair the two row by row.
        X_test = batch.drop(columns=['class'])
        test_scores = classifier.predict_proba(X_test)

        multiclass_map = get_multiclass_quantifier_map(y_validation, X_test, validation_scores, test_scores, qnt_models, classifier)

        results = {}
        for q in quantifiers:
            if q not in multiclass_map:
                raise ValueError(f"Unknown quantifier: {q}")
            results[q] = multiclass_map[q]()

        metadata = {
            "batch_index": batch_index,
            "model_id": model_id,
            "incoming_test_scores": test_scores,
            "selected_p_scores": None,
            "selected_n_scores": None,
        }

    return results, metadata

def handle_batch_results(batch_result, multiclass_result, batch_df, quantifiers, batch_index, classes):
    # Get real prevalence from batch
    real_prevalence = batch_df['class'].value_counts(normalize=True).sort_index().to_dict()

    # Binary datasets are handled by a single classifier (no One-vs-Rest), which
    # already reports the positive prevalence; the negative one is its
    # complement, so the pair sums to 1 and needs no normalization.
    is_binary = len(classes) == 2 and len(batch_result) == 1
    pos_cls = max(batch_result.keys()) if is_binary else None
    neg_cls = min(cls for cls in classes if cls != pos_cls) if is_binary else None

    # Restructure results: one row per quantifier
    quantifier_results = {}
    for q in quantifiers:
        # Collect predictions for each class (positive class probability)
        if is_binary:
            pos_pred = batch_result[pos_cls][q]
            class_predictions = {pos_cls: pos_pred, neg_cls: 1 - pos_pred}
            normalized_predictions = dict(class_predictions)
        else:
            class_predictions = {}
            for cls in sorted(batch_result.keys()):
                class_predictions[cls] = batch_result[cls][q]

            # Normalize predictions to sum to 1
            total = sum(class_predictions.values())
            normalized_predictions = {cls: pred / total if total > 0 else 0 for cls, pred in class_predictions.items()}

        quantifier_results[q] = {
            'predictions': class_predictions,
            'normalized_predictions': normalized_predictions,
            'real_prevalence': real_prevalence,
            'batch_index': batch_index,
        }

    # Add multiclass quantifiers
    for q in multiclass_result:
        if isinstance(multiclass_result[q], dict):
            predictions = multiclass_result[q]
        else:
            # Transform array into dict based on the sorted class list (the same
            # order mlquantify uses, i.e. classifier.classes_)
            ordered_classes = [int(cls) if isinstance(cls, (int, np.integer)) else cls for cls in sorted(classes)]
            predictions = {cls: float(multiclass_result[q][i]) for i, cls in enumerate(ordered_classes)}

        quantifier_results[q] = {
            'predictions': predictions,
            'normalized_predictions': predictions,  # Assuming multiclass predictions are already normalized
            'real_prevalence': real_prevalence,
            'batch_index': batch_index,
        }
    
    return quantifier_results

def process_single_batch(batch_index, idx, y_validation, tests, test_df, classifiers, quantifiers, classifier, validation_scores, qnt_models, test_scores_dir, classes):
    """Process a single batch - designed for parallel execution."""
    batch_result = {}
    for cls in tests:
        binary_test = tests[cls]
        binary_classifier = classifiers[cls]
        binary_batch = binary_test.iloc[idx]
        batch_result[cls], score_metadata = test_classifier(
            binary_batch,
            binary_classifier,
            quantifiers['binary'],
            batch_index=batch_index,
            model_id=cls,
        ) # put binary quantifiers only.
        persist_batch_scores(
            test_scores_dir,
            batch_index,
            cls,
            score_metadata["incoming_test_scores"],
            score_metadata["selected_p_scores"],
            score_metadata["selected_n_scores"],
        )
    
    # Also get multiclass quantifiers (skipped entirely when RUN_MULTICLASS is
    # off -- handle_batch_results then only emits the binary quantifier rows).
    multiclass_result = {}
    if RUN_MULTICLASS:
        batch = test_df.iloc[idx]
        multiclass_result, multiclass_metadata = test_classifier(
            batch,
            classifier,
            quantifiers['multiclass'],
            validation_scores,
            qnt_models,
            y_validation,
            batch_index=batch_index,
            model_id="multiclass",
        ) # multiclass quantifiers
        persist_batch_scores(
            test_scores_dir,
            batch_index,
            "multiclass",
            multiclass_metadata["incoming_test_scores"],
            multiclass_metadata["selected_p_scores"],
            multiclass_metadata["selected_n_scores"],
        )


    result = handle_batch_results(batch_result, multiclass_result, test_df.iloc[idx], quantifiers['binary'], batch_index, classes)

    return result


def test_one_vs_rest_classifiers(y_validation, tests, test_df, classifiers, quantifiers, classifier, validation_scores, qnt_models, test_scores_dir, classes, batch_indices=None, n_jobs=-1):
    """
    Test classifiers using parallel processing.

    Parameters:
    -----------
    batch_indices : list of positional index arrays, default=None
        Pre-built test batches. When None, artificial batches are drawn with UPP.
        The synthetic datasets pass their own temporal bags here instead.
    n_jobs : int, default=-1
        Number of CPU cores to use. -1 means all available cores.
    """
    if batch_indices is None:
        upp = UPP(batch_size=100, n_prevalences=100, repeats=10, random_state=42)

        X = test_df.drop(columns=['class'])
        y = test_df['class']

        # Collect all batch indices first
        batch_indices = list(upp.split(X, y))

    total_batches = len(batch_indices)

    # Process batches in parallel
    all_results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_single_batch)(batch_index, idx, y_validation, tests, test_df, classifiers, quantifiers, classifier, validation_scores, qnt_models, test_scores_dir, classes)
        for batch_index, idx in enumerate(tqdm(batch_indices, total=total_batches, desc="Processing batches", unit="batch"))
    )

    return all_results

def is_synthetic_dataset(dataset_path):
    """True for the generated drift datasets living in datasets/synthetic/."""
    return 'datasets/synthetic' in str(dataset_path).replace(os.sep, '/')

def preprocess_synthetic(df):
    """Adapt a synthetic_generator CSV (x1, x2, label, t, blob) to the shared
    convention. The `t` column is kept here and consumed by
    split_synthetic_by_time; `blob` is generator metadata and is dropped."""
    df = df.rename(columns={'label': 'class'})
    return df.drop(columns=['blob'], errors='ignore')

def split_synthetic_by_time(df, train_last_bag=SYNTHETIC_TRAIN_LAST_BAG):
    """Split a synthetic dataset along its own temporal bags.

    Bags 0..train_last_bag (inclusive) become the training set; every later bag
    becomes one test batch, in temporal order. Returns the two frames without
    the `t` column (scale_dataset treats every non-'class' column as a feature),
    plus the positional index array of each test bag.
    """
    bags = sorted(df['t'].unique())
    train_bags, test_bags = bags[:train_last_bag + 1], bags[train_last_bag + 1:]

    train_df = df[df['t'].isin(train_bags)]
    test_df = df[df['t'].isin(test_bags)]

    test_t = test_df['t'].to_numpy()
    batch_indices = [np.flatnonzero(test_t == bag) for bag in test_bags]

    train_df = train_df.drop(columns=['t'])
    test_df = test_df.drop(columns=['t'])

    return train_df, test_df, batch_indices, test_bags

def pre_process_dts(df, dataset_name, dataset_path):
    # Mapping of dataset names to preprocessing functions
    kaggle_datasets = {
        'cirrhosis': kaggle_preprocess.preprocess_cirrhosis,
        'predictive_maintenance': kaggle_preprocess.preprocess_predictive_maintenance,
        'star_classification': kaggle_preprocess.preprocess_star_classification,
        'Student_performance_data': kaggle_preprocess.preprocess_student_performance,
        'zoo': kaggle_preprocess.preprocess_zoo,
        'healthcare': kaggle_preprocess.preprocess_healthcare,
        'music_genre': kaggle_preprocess.preprocess_music_genre,
        'customer_segmentation': kaggle_preprocess.preprocess_customer_segmentation,
        'fashion-mnist': kaggle_preprocess.preprocess_fashion_mnist,
        'zoo2': kaggle_preprocess.preprocess_zoo2,
        'zoo3': kaggle_preprocess.preprocess_zoo3
    }
    
    openml_datasets = {
        'dataset_313_spectrometer': openml_preprocess.preprocess_spectrometer,
        'dataset_4552_BachChoralHarmony': openml_preprocess.preprocess_bach_choral_harmony,
        'dataset_1457_amazon-commerce-reviews': openml_preprocess.preprocess_amazon_commerce_reviews,
        'fabert': openml_preprocess.preprocess_fabert,
        'dataset_44478_amazon-commerce-reviews_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True': openml_preprocess.preprocess_amazon_commerce_reviews_subset,
        'dataset_44479_amazon-commerce-reviews_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True': openml_preprocess.preprocess_amazon_commerce_reviews_subset,
        'dataset_44480_amazon-commerce-reviews_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True': openml_preprocess.preprocess_amazon_commerce_reviews_subset,
        'dataset_44481_amazon-commerce-reviews_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True': openml_preprocess.preprocess_amazon_commerce_reviews_subset,
        'dataset_44482_amazon-commerce-reviews_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True': openml_preprocess.preprocess_amazon_commerce_reviews_subset,
        'fars': openml_preprocess.preprocess_fars,
    }

    schumacher_datasets = {
        'bike_sharing_data': schumacher_preprocess.preprocess_bike,
        'blog_feedback_data': schumacher_preprocess.preprocess_blog_feedback,
        'concrete_data': schumacher_preprocess.preprocess_concrete,
        'contraceptive_data': schumacher_preprocess.preprocess_contraceptive,
        'diamonds_data': schumacher_preprocess.preprocess_diamonds,
        'drugs_data': schumacher_preprocess.preprocess_drugs,
        'energy_data': schumacher_preprocess.preprocess_energy,
        'fifa19_data': schumacher_preprocess.preprocess_fifa19,
        'news_popularity_data': schumacher_preprocess.preprocess_news_popularity,
        'skillcraft_data': schumacher_preprocess.preprocess_skillcraft,
        'superconductor_data': schumacher_preprocess.preprocess_superconductor,
        'turk_student_eval_data': schumacher_preprocess.preprocess_turk_student_eval,
        'video_game_sales_data': schumacher_preprocess.preprocess_video_game_sales,
        'yeast_data': schumacher_preprocess.preprocess_yeast,
        'theorem_data': schumacher_preprocess.preprocess_theorem,
    }

    # Apply preprocessing based on dataset name
    if is_synthetic_dataset(dataset_path):
        df = preprocess_synthetic(df)
    elif dataset_name in kaggle_datasets:
        df = kaggle_datasets[dataset_name](df)
    elif dataset_name in openml_datasets:
        df = openml_datasets[dataset_name](df)
    elif dataset_name in schumacher_datasets:
        df = schumacher_datasets[dataset_name](df)

    # Keep existing preprocessing for other datasets
    elif dataset_name == 'Avila':
        df = df.rename(columns={'V11': 'class'})
    elif dataset_name == 'Chessgame':
        df = df.rename(columns={'game': 'class'})
        # Transform categorical columns (chess board letters) to numerical
        categorical_cols = ['white_king_col', 'white_rook_col', 'black_king_col']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ord(x.lower()) - ord('a') + 1 if isinstance(x, str) else x)
    elif dataset_name == 'HAR':
        df = ours_preprocess.preprocess_har(df)
    elif dataset_name == 'Covertype':
        df = ours_preprocess.preprocess_covertype(df)
    elif dataset_name == 'Dermatology':
        df = ours_preprocess.preprocess_dermatology(df)
    elif dataset_name == 'Mosquitoes':
        df = ours_preprocess.preprocess_mosquitoes(df)
    elif dataset_name == 'Land-use' or dataset_name == 'Walking':
        df = df.rename(columns={'Class': 'class'})
    elif dataset_name == 'Nursery':
        feature_cols = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']
        categories = [
            ['usual', 'pretentious', 'great_pret'],
            ['proper', 'less_proper', 'improper', 'critical', 'very_crit'],
            ['foster', 'incomplete', 'complete', 'completed'],
            ['1', '2', '3', 'more'],
            ['convenient', 'less_conv', 'critical'],
            ['convenient', 'inconv'],
            ['nonprob', 'slightly_prob', 'problematic'],
            ['not_recom', 'recommended', 'priority']
        ]
        ordinal_encoder = OrdinalEncoder(categories=categories)
        df[feature_cols] = ordinal_encoder.fit_transform(df[feature_cols])
        df = df[df['class'] != 'recommend']
    elif dataset_name == 'Walking':
        df = df.rename(columns={'Class': 'class'})

    # Drop missing values from the entire dataset (both features and target)
    df = df.dropna()

    if df['class'].dtype == 'float64' or df['class'].dtype == 'int64':
        df['class'] = df['class'].astype(int)
    
    return df

def process_single_dataset(dataset_path):
    """Process a single dataset and save results."""
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    dataset_dir, test_scores_dir = build_dataset_output_dirs(dataset_name)
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(dataset_path)
    df = pre_process_dts(df, dataset_name, dataset_path)

    # The synthetic datasets carry their own temporal batching in the `t`
    # column: train on the first bags and use every later bag as one test batch,
    # preserving the drift the generator encodes. Every other dataset keeps the
    # random 50/50 split plus UPP-sampled batches.
    if is_synthetic_dataset(dataset_path):
        train_df, test_df, batch_indices, test_bags = split_synthetic_by_time(df)
        print(f"Synthetic split: train on bags 0..{SYNTHETIC_TRAIN_LAST_BAG} ({len(train_df)} rows), "
              f"{len(batch_indices)} test batches ({len(test_df)} rows)")
    else:
        train_df, test_df = train_test_split(df, test_size=0.5, stratify=df['class'], random_state=42)
        batch_indices, test_bags = None, None

    train_df, train_scaler, scaled_flag = scale_dataset(train_df)
    if scaled_flag:
        test_df, _, _ = scale_dataset(test_df, scaler=train_scaler)

    trains = binarize_dataset(train_df)
    tests = binarize_dataset(test_df)

    classes = sorted(train_df['class'].unique())
    # Every binary quantifier returns [positive_prevalence, 1 - positive], where
    # "positive" is the class its OvR classifier was trained to detect. Always
    # store the positive entry (index 0) so the stored value is the estimated
    # prevalence of that class -- the same class whose true prevalence is written
    # to c{cls}_real. handle_batch_results then normalizes the K per-class
    # estimates onto the simplex (multiclass) or pairs the single estimate with
    # its complement (binary). Reading index 1 instead stored 1 - p_cls, the
    # prevalence of "not this class", leaving prediction and truth on opposite
    # sides of the OvR problem.
    is_binary_dataset = len(classes) == 2

    classifiers = train_one_vs_rest_classifiers(trains, dataset_dir)

    # The plain (non-OvR) classifier and its cross-validated scores only feed the
    # multiclass quantifiers, so its 10-fold fit is skipped along with them.
    classifier, validation_scores, y_validation, priors = None, None, None, None
    if RUN_MULTICLASS:
        classifier, _, _, _, validation_scores, _, y_validation = train_classifier(train_df, fit_cdt=RUN_CDT)

        # train_classifier only strips the label column from the validation scores
        # on the multiclass branch; the multiclass quantifiers need the posteriors
        # alone, so strip it here for binary datasets too.
        priors = validation_scores[:, :-1] if is_binary_dataset else validation_scores

    quantifiers = {
        "binary": [
            "PCC",
            "ACC",
            "PACC",
            "T50",
            "MAX",
            "MS",
            "MS2",
            "X",
            "SMM",
            "DyS",
            "DySyn",
            "HDy",
            "ACC_syn",
            "PACC_syn",
            "X_syn",
            "MAX_syn",
            "T50_syn",
            "MS_syn",
            "MS2_syn",
            "SMM_syn",
            "HDy_syn",
        ],
        "multiclass": [
            "CC2",
            "PWK",
            "HDx",
            "GAC",
            "GPAC",
            "FM",
            "EMQ",
            "KDEyHD",
            "KDEyCS",
            "KDEyML",
        ] if RUN_MULTICLASS else []
    }

    # The CDT-gated variants only exist when the CDT pipeline is on.
    if RUN_CDT:
        quantifiers["binary"] += [
            "DyS_cdt",
            "ACC_cdt",
            "PACC_cdt",
            "X_cdt",
            "MAX_cdt",
            "T50_cdt",
            "MS_cdt",
            "MS2_cdt",
            "SMM_cdt",
            "HDy_cdt",
        ]

    # PWK and HDx are only ever called through the multiclass map, so fitting
    # them (HDx in particular is expensive on wide datasets) is skipped too.
    qnt_models = train_quantifiers(train_df) if RUN_MULTICLASS else None
    persist_training_distributions(dataset_dir, classifiers, validation_scores)

    results = test_one_vs_rest_classifiers(
        y_validation,
        tests,
        test_df,
        classifiers,
        quantifiers,
        classifier,
        priors,
        qnt_models,
        test_scores_dir,
        classes,
        batch_indices=batch_indices,
        n_jobs=1,
    )

    # Flatten results into rows for CSV
    rows = []
    for batch_result in results:
        for quantifier, data in batch_result.items():
            row = {
                'qnt': quantifier,
                'batch_index': data['batch_index'],
            }
            # Synthetic runs: keep the bag's timestamp alongside the batch index
            # so drift can be plotted against t directly.
            if test_bags is not None:
                row['t'] = test_bags[data['batch_index']]

            row_classes = sorted(data['predictions'].keys())

            # Add predictions
            for cls in row_classes:
                row[f'c{cls}_p'] = data['predictions'][cls]

            # Add normalized predictions
            for cls in row_classes:
                row[f'c{cls}_p_normalized'] = data['normalized_predictions'][cls]

            # Add real prevalence
            for cls in row_classes:
                row[f'c{cls}_real'] = data['real_prevalence'].get(cls, 0)

            rows.append(row)

    # Write to CSV
    results_df = pd.DataFrame(rows)
    # Round the prevalences only: `t` identifies the bag and must keep its
    # precision (consecutive bags are ~0.02 apart).
    round_cols = [col for col in results_df.columns if col != 't']
    results_df[round_cols] = results_df[round_cols].round(2)
    output_path = os.path.join(dataset_dir, f'{dataset_name}_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='One-vs-Rest Quantification')
    parser.add_argument('-dts', '--dataset', required=False, help='Path to the dataset CSV file')
    parser.add_argument('-exp', '--experiments', required=False, help='Path to the experiments TXT file')

    args = parser.parse_args()
    
    # Check that at least one argument is provided
    if not args.dataset and not args.experiments:
        parser.error("At least one of -dts/--dataset or -exp/--experiments is required")
    
    # Process experiments file if provided
    if args.experiments:
        with open(args.experiments, 'r') as f:
            commands = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        print(f"Found {len(commands)} datasets to process")
        
        for i, command in enumerate(commands, 1):
            # Parse the command to extract the dataset path
            # Expected format: python ovr.py -dts datasets/kaggle/cirrhosis.csv
            parts = command.split()
            if '-dts' in parts:
                dts_index = parts.index('-dts')
                if dts_index + 1 < len(parts):
                    dataset_path = parts[dts_index + 1]
                    print(f"\n[{i}/{len(commands)}] Processing: {dataset_path}")
                    try:
                        process_single_dataset(dataset_path)
                    except Exception as e:
                        print(f"ERROR processing {dataset_path}: {e}")
                        continue
        
        print(f"\n{'='*60}")
        print(f"Completed processing all {len(commands)} datasets")
        print(f"{'='*60}")
    
    # Process single dataset if provided
    elif args.dataset:
        process_single_dataset(args.dataset)
