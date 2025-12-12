import pandas as pd
import numpy as np

from mlquantify.model_selection import UPP

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression

import pdb

from methods.Quantifiers import (
    CC,
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
from methods.QuaDapt import (
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
from methods.Quantifiers_Utils import getTPRandFPRbyThreshold
import math

def get_quantifier_map(test_scores, tpr_fpr, pos_scores, neg_scores):
    """Returns a dictionary mapping quantifier names to their callable functions."""
    return {
        "CC": lambda: CC(test_scores, thr=0.5),
        "PCC": lambda: PCC(test_scores),
        "ACC": lambda: ACC(test_scores, tpr_fpr, thr=0.5),
        "PACC": lambda: PACC(test_scores, tpr_fpr, thr=0.5),
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

def scale_dataset(df, scaler=None):
    df_scaled = df.copy()
    
    # Get feature columns (all except last)
    feature_cols = df_scaled.columns[:-1]
    
    if scaler is None:
        scaler = StandardScaler()
        # Fit and transform, assign as float
        df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
    else:
        # Transform only, assign as float
        df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])

    return df_scaled, scaler

def binarize_dataset(train_df):
    trains = {}

    classes = train_df['class'].unique()
    for cls in classes:
        bin_train_df = train_df.copy()
        bin_train_df['class'] = (bin_train_df['class'] == cls).astype(int)
        trains[cls] = bin_train_df

    return trains

def train_classifier(train_df):
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class']
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    clf = LogisticRegression(random_state=42)
    
    fold_scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        clf.fit(X_fold_train, y_fold_train)
        proba = clf.predict_proba(X_fold_val)
        fold_scores.append(np.column_stack((proba, y_fold_val)))
    
    clf.fit(X_train, y_train)
    validation_scores = np.vstack(fold_scores)
    tpr_fpr = getTPRandFPRbyThreshold(validation_scores)
    pos_scores = validation_scores[validation_scores[:, 2] == 1, 1].astype(float)
    neg_scores = validation_scores[validation_scores[:, 2] == 0, 1].astype(float)

    return clf, tpr_fpr, pos_scores, neg_scores

def train_one_vs_rest_classifiers(trains):
    classifiers = {}

    for cls, bin_train_df in trains.items():
        clf, tpr_fpr, pos_scores, neg_scores = train_classifier(bin_train_df)
        classifiers[cls] = {
            "model": clf,
            "tpr_fpr": tpr_fpr,
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
        }

    return classifiers

def test_classifier(binary_batch, classifier, quantifiers):
    clf = classifier["model"]
    tpr_fpr = classifier["tpr_fpr"]
    pos_scores = classifier["pos_scores"]
    neg_scores = classifier["neg_scores"]

    X_test = binary_batch.drop(columns=['class'])
    test_scores = clf.predict_proba(X_test)[:, 1]

    # Get quantifier map with all necessary parameters
    quantifier_map = get_quantifier_map(test_scores, tpr_fpr, pos_scores, neg_scores)

    results = {}
    for q in quantifiers:
        if q not in quantifier_map:
            raise ValueError(f"Unknown quantifier: {q}")
        results[q] = quantifier_map[q]()[1]

    return results

def handle_batch_results(batch_result, batch_df, quantifiers):
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
        normalized_predictions = {cls: pred / total for cls, pred in class_predictions.items()}

        # Compute normalized cross-entropy
        eps = 1e-10
        cross_entropy = -sum(real_prevalence.get(cls, 0) * math.log(normalized_predictions[cls] + eps) 
                             for cls in normalized_predictions)
        max_entropy = math.log(len(normalized_predictions))
        normalized_ce = cross_entropy / max_entropy if max_entropy > 0 else 0
        
        quantifier_results[q] = {
            'predictions': class_predictions,
            'normalized_predictions': normalized_predictions,
            'real_prevalence': real_prevalence
        }
    return quantifier_results

def test_one_vs_rest_classifiers(tests, test_df, classifiers, quantifiers):

    upp = UPP(batch_size=100, n_prevalences=2, repeats=2, random_state=42)

    X = test_df.drop(columns=['class'])
    y = test_df['class']

    all_results = []
    for idx in upp.split(X, y):
        batch_result = {}
        for cls in tests:
            test = tests[cls]
            classifier = classifiers[cls]
            binary_batch = test.iloc[idx]
            batch_result[cls] = test_classifier(binary_batch, classifier, quantifiers)

        result = handle_batch_results(batch_result, test_df.iloc[idx], quantifiers)
        all_results.append(result)

    return all_results


if __name__ == "__main__":
    df = pd.read_csv("./datasets/Dermatology.csv")

    train_df, test_df = train_test_split(df, test_size=0.5, stratify=df['class'], random_state=42)
    train_df, train_scaler = scale_dataset(train_df)
    test_df, _ = scale_dataset(test_df, scaler=train_scaler)
    
    trains = binarize_dataset(train_df)
    tests = binarize_dataset(test_df)

    classifiers = train_one_vs_rest_classifiers(trains)
    quantifiers = [
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
    ]
    results = test_one_vs_rest_classifiers(tests, test_df, classifiers, quantifiers)

    # Flatten results into rows for CSV
    rows = []
    for batch_result in results:
        for quantifier, data in batch_result.items():
            row = {'qnt': quantifier}
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
    results_df.to_csv('results.csv', index=False)
