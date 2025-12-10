import pandas as pd
import numpy as np

import mlquantify as mlq
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pdb
from sklearn.model_selection import StratifiedKFold

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
    
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        clf.fit(X_fold_train, y_fold_train)
        score = clf.predict_proba(X_fold_val)
        scores.append(score)
        # pdb.set_trace()
    
    clf.fit(X_train, y_train)
    scores = np.vstack(scores) # transform into 1D array

    return clf, scores

def train_one_vs_rest_classifiers(trains):
    classifiers = {}

    for cls, bin_train_df in trains.items():
        clf, scores = train_classifier(bin_train_df)
        classifiers[cls] = (clf, scores)

    return classifiers

def test_classifier(test_df, classifier):
    X_test = test_df.drop(columns=['class'])
    y_test = test_df['class']

    y_pred = classifier.predict_proba(X_test)[:, 1]

    return y_pred


def test_one_vs_rest_classifiers(test_df, classifiers):
    results = {}

    X_test = test_df.drop(columns=['class'])
    y_test = test_df['class']

    scaler = classifiers[list(classifiers.keys())[0]][2]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    for cls, (clf, _, scaler) in classifiers.items():
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        y_pred = clf.predict_proba(X_test_scaled)[:, 1]
        results[cls] = y_pred

    return results

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

if __name__ == "__main__":
    df = pd.read_csv("./datasets/Dermatology.csv")

    train_df, test_df = train_test_split(df, test_size=0.5, stratify=df['class'], random_state=42)
    train_dff, train_scaler = scale_dataset(train_df)
    test_df, _ = scale_dataset(test_df, scaler=train_scaler)
    pdb.set_trace()
    
    trains = binarize_dataset(train_df)

    classifiers = train_one_vs_rest_classifiers(trains)



    # X = test_df.drop(columns=['class'])
    # y = test_df['class']

    # upp = mlq.model_selection.UPP(batch_size=100, n_prevalences=1, repeats=10)

    # for idx in upp.split(X, y):
    #     selected_df = test_df.iloc[idx]
    #     print(selected_df.head(10))
    #     print(selected_df['class'].value_counts(normalize=True))
