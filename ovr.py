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

# Toggle persistence of score distributions: training distributions, per-batch
# test scores, and the DySyn selected distributions (the cdt_results/<dataset>/
# {test_scores,<class>,multiclass} file structure). Set to False to skip writing
# them; the final <dataset>_results.csv is always written.
SAVE_DISTRIBUTIONS = False

def serialize_scores(scores):
    if scores is None:
        return ""

    return json.dumps(np.asarray(scores).tolist())

def sanitize_model_name(model_id):
    return str(model_id).replace(os.sep, "_").replace("/", "_").replace(" ", "_")

def build_dataset_output_dirs(dataset_name):
    dataset_dir = os.path.join(".", "cdt_results", dataset_name)
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
        "DySyn": lambda: DySyn(test_scores, measure="hellinger")[0],
        "HDy": lambda: HDy(pos_scores, neg_scores, test_scores)[0],
        "ACC_syn": lambda: ACCSyn(test_scores, measure="hellinger", MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "PACC_syn": lambda: PACCSyn(test_scores, measure="hellinger", MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "X_syn": lambda: XSyn(test_scores, measure="hellinger", MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "MAX_syn": lambda: MAXSyn(test_scores, measure="hellinger", MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "T50_syn": lambda: T50Syn(test_scores, measure="hellinger", MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "MS_syn": lambda: MSSyn(test_scores, measure="hellinger", MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "MS2_syn": lambda: MS2Syn(test_scores, measure="hellinger", MF_dysyn=np.arange(0.1, 1.0, 0.2)),
        "SMM_syn": lambda: SMMSyn(test_scores, measure="hellinger", MF_dysyn=np.arange(0.1, 1.0, 0.2)),
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

def train_classifier(train_df):
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class']

    # CDT calculation
    cdt = None
    if len(y_train.unique()) == 2:  # Binary only
        cdt = CDT(classifier=RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        cdt.fit(train_df)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    clf = LogisticRegression(random_state=42, max_iter=2000, n_jobs=-1)
    
    fold_scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        clf.fit(X_fold_train, y_fold_train)
        proba = clf.predict_proba(X_fold_val)
        fold_scores.append(np.column_stack((proba, y_fold_val)))
    
    clf.fit(X_train, y_train)
    validation_scores = np.vstack(fold_scores)

    if len(y_train.unique()) > 2: # Multiclass
        validation_scores = validation_scores[:, :-1]  # Remove last column (labels)
        tpr_fpr = None
        pos_scores = None
        neg_scores = None
        
    else: # Binary
        tpr_fpr = getTPRandFPRbyThreshold(validation_scores)
        pos_scores = validation_scores[validation_scores[:, 2] == 1, 1].astype(float)
        neg_scores = validation_scores[validation_scores[:, 2] == 0, 1].astype(float)

    return clf, tpr_fpr, pos_scores, neg_scores, validation_scores, cdt

def train_one_vs_rest_classifiers(trains, dataset_dir=None):
    classifiers = {}

    # All binary classifiers of the dataset share a single distances.csv,
    # storing the DyS distances measured inside each CDT. The first classifier
    # rewrites the file from scratch (so rows from a previous run don't
    # accumulate and duplicate each class); the rest append to it.
    distances_path = os.path.join(dataset_dir, "distances.csv") if dataset_dir else None
    first_save = True

    for cls, bin_train_df in trains.items():
        clf, tpr_fpr, pos_scores, neg_scores, _, cdt = train_classifier(bin_train_df)

        cdt_thr = None
        if cdt is not None:
            cdt_thr = cdt.thr
            if distances_path is not None:
                cdt.save_distances(distances_path, model_id=cls, overwrite=first_save)
                first_save = False

        classifiers[cls] = {
            "model": clf,
            "tpr_fpr": tpr_fpr,
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "cdt_thr": cdt_thr,
        }

    return classifiers

def test_classifier(batch, classifier, quantifiers, validation_scores=None, qnt_models=None, train_df=None, batch_index=None, model_id=None):

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

        # CDT drift detector (binary only): set up once with the trained threshold
        # and the DyS distance of this batch (reused for every synthetic quantifier).
        cdt = None
        dys_distance = None
        if cdt_thr is not None:
            cdt = CDT(classifier=None)
            cdt.thr = cdt_thr
            _, dys_distance = DyS(pos_scores, neg_scores, test_scores, return_distance=True)

        for q in quantifiers:
            # CDT-gated variant ({base}_cdt): drift (distance >= thr) -> synthetic
            # version, otherwise fall back to the base (non-synthetic) quantifier.
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
                results[q] = results[chosen] if chosen in results else quantifier_map[chosen]()[1]
                continue

            if q not in quantifier_map:
                raise ValueError(f"Unknown quantifier: {q}")
            if q == "DySyn":
                qnt_result = DySyn(
                    test_scores,
                    measure="hellinger",
                    write_distribution=False,
                    return_metadata=True,
                )
                results[q] = qnt_result[0][1]
                selected_p_scores = qnt_result[3]["selected_p_scores"]
                selected_n_scores = qnt_result[3]["selected_n_scores"]
            else:
                results[q] = quantifier_map[q]()[1]

        metadata = {
            "batch_index": batch_index,
            "model_id": model_id,
            "incoming_test_scores": test_scores,
            "selected_p_scores": selected_p_scores,
            "selected_n_scores": selected_n_scores,
        }

    else: # Multiclass
        y_train = train_df['class']
        X_test = batch.drop(columns=['class'])
        test_scores = classifier.predict_proba(X_test)

        multiclass_map = get_multiclass_quantifier_map(y_train, X_test, validation_scores, test_scores, qnt_models, classifier)

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

def handle_batch_results(batch_result, multiclass_result, batch_df, quantifiers, batch_index):
    # Get real prevalence from batch
    real_prevalence = batch_df['class'].value_counts(normalize=True).sort_index().to_dict()

    # Restructure results: one row per quantifier
    quantifier_results = {}
    for q in quantifiers:
        # Collect predictions for each class (positive class probability)
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
            # Transform array into dict based on sorted class keys
            classes = sorted(batch_result.keys())
            classes = [int(cls) if isinstance(cls, (int, np.integer)) else cls for cls in classes]
            predictions = {cls: float(multiclass_result[q][i]) for i, cls in enumerate(classes)}

        quantifier_results[q] = {
            'predictions': predictions,
            'normalized_predictions': predictions,  # Assuming multiclass predictions are already normalized
            'real_prevalence': real_prevalence,
            'batch_index': batch_index,
        }
    
    return quantifier_results

def process_single_batch(batch_index, idx, train_df, tests, test_df, classifiers, quantifiers, classifier, validation_scores, qnt_models, test_scores_dir):
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
    
    batch = test_df.iloc[idx]
    # Also get multiclass quantifiers
    multiclass_result, multiclass_metadata = test_classifier(
        batch,
        classifier,
        quantifiers['multiclass'],
        validation_scores,
        qnt_models,
        train_df,
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
    
    result = handle_batch_results(batch_result, multiclass_result, test_df.iloc[idx], quantifiers['binary'], batch_index)

    return result


def test_one_vs_rest_classifiers(train_df, tests, test_df, classifiers, quantifiers, classifier, validation_scores, qnt_models, test_scores_dir, n_jobs=-1):
    """
    Test classifiers using parallel processing.
    
    Parameters:
    -----------
    n_jobs : int, default=-1
        Number of CPU cores to use. -1 means all available cores.
    """
    upp = UPP(batch_size=100, n_prevalences=100, repeats=10, random_state=42)

    X = test_df.drop(columns=['class'])
    y = test_df['class']

    # Calculate total number of batches
    total_batches = upp.n_prevalences * upp.repeats
    
    # Collect all batch indices first
    batch_indices = list(upp.split(X, y))
    
    # Process batches in parallel
    all_results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_single_batch)(batch_index, idx, train_df, tests, test_df, classifiers, quantifiers, classifier, validation_scores, qnt_models, test_scores_dir)
        for batch_index, idx in enumerate(tqdm(batch_indices, total=total_batches, desc="Processing batches", unit="batch"))
    )

    return all_results

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
    
    # Apply preprocessing based on dataset name
    if dataset_name in kaggle_datasets:
        df = kaggle_datasets[dataset_name](df)
    elif dataset_name in openml_datasets:
        df = openml_datasets[dataset_name](df)
        
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

    train_df, test_df = train_test_split(df, test_size=0.5, stratify=df['class'], random_state=42)
    train_df, train_scaler, scaled_flag = scale_dataset(train_df)
    if scaled_flag:
        test_df, _, _ = scale_dataset(test_df, scaler=train_scaler)
    
    trains = binarize_dataset(train_df)
    tests = binarize_dataset(test_df)

    classifiers = train_one_vs_rest_classifiers(trains, dataset_dir)
    classifier, _, _, _, validation_scores, _ = train_classifier(train_df)
    
    quantifiers = {
        "binary": [
            "CC",
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
        ]
    }

    qnt_models = train_quantifiers(train_df)
    persist_training_distributions(dataset_dir, classifiers, validation_scores)

    results = test_one_vs_rest_classifiers(
        train_df,
        tests,
        test_df,
        classifiers,
        quantifiers,
        classifier,
        validation_scores,
        qnt_models,
        test_scores_dir,
        n_jobs=23,
    )

    # Flatten results into rows for CSV
    rows = []
    for batch_result in results:
        for quantifier, data in batch_result.items():
            row = {
                'qnt': quantifier,
                'batch_index': data['batch_index'],
            }
            classes = sorted(data['predictions'].keys())
            
            # Add predictions
            for cls in classes:
                row[f'c{cls}_p'] = data['predictions'][cls]
            
            # Add normalized predictions
            for cls in classes:
                row[f'c{cls}_p_normalized'] = data['normalized_predictions'][cls]
            
            # Add real prevalence
            for cls in classes:
                row[f'c{cls}_real'] = data['real_prevalence'].get(cls, 0)
            
            rows.append(row)

    # Write to CSV
    results_df = pd.DataFrame(rows).round(2)
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
