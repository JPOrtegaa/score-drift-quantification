import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# from methods.quantifiers import DyS

def getTPRandFPRbyThreshold(validation_scores):
    unique_scores = np.arange(0.01, 1.00, 0.01)
    arrayOfTPRandFPRByTr = []

    total_positive = np.sum(validation_scores[:, 2] == 1)
    total_negative = np.sum(validation_scores[:, 2] == 0)

    for threshold in unique_scores:
        fp = np.sum((validation_scores[:, 0] > threshold) & (validation_scores[:, 2] == 0))
        tp = np.sum((validation_scores[:, 0] > threshold) & (validation_scores[:, 2] == 1))
        tpr = tp / total_positive if total_positive > 0 else 0
        fpr = fp / total_negative if total_negative > 0 else 0

        arrayOfTPRandFPRByTr.append([round(threshold, 2), tpr, fpr])

    return np.array(arrayOfTPRandFPRByTr, dtype=object)

def DySyn_distance(x, method="hellinger"):
    if method == "ord":
        x_dif = x[0, :] - x[1, :]
        acum = 0
        aux = 0
        for val in x_dif:
            aux += val
            acum += aux
        return abs(acum)

    if method == "topsoe":
        return sum(x[0, i] * np.log((2 * x[0, i]) / (x[0, i] + x[1, i])) +
                   x[1, i] * np.log((2 * x[1, i]) / (x[1, i] + x[0, i])) for i in range(x.shape[1]))

    if method == "jensen_difference":
        return sum(((x[0, i] * np.log(x[0, i]) + x[1, i] * np.log(x[1, i])) / 2) -
                   ((x[0, i] + x[1, i]) / 2) * np.log((x[0, i] + x[1, i]) / 2) for i in range(x.shape[1]))

    if method == "taneja":
        return sum(((x[0, i] + x[1, i]) / 2) * np.log((x[0, i] + x[1, i]) / (2 * np.sqrt(x[0, i] * x[1, i])))
                   for i in range(x.shape[1]))

    if method == "hellinger":
        return 2 * np.sqrt(1 - sum(np.sqrt(x[0, i] * x[1, i]) for i in range(x.shape[1])))

    if method == "prob_symm":
        return 2 * sum(((x[0, i] - x[1, i]) ** 2) / (x[0, i] + x[1, i]) for i in range(x.shape[1]))

    raise ValueError("measure argument must be a valid option")

def getHist(scores, nbins):
    breaks = np.linspace(0, 1, nbins + 1)
    breaks[-1] = 1.1
    re = np.full(len(breaks) - 1, 1 / (len(breaks) - 1))

    for i in range(1, len(breaks)):
        re[i - 1] = (re[i - 1] + np.sum((scores >= breaks[i - 1]) & (scores < breaks[i]))) / (len(scores) + 1)

    return re

def TernarySearch(left, right, f, eps=1e-4):
    while True:
        if abs(left - right) < eps:
            return (left + right) / 2

        leftThird = left + (right - left) / 3
        rightThird = right - (right - left) / 3

        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird

def MoSS(n, alpha, m):
    p_score = np.random.uniform(size=int(n * alpha)) ** m
    n_score = 1 - (np.random.uniform(size=int(round(n * (1 - alpha), 0))) ** m)
    scores = np.column_stack((np.concatenate((p_score, n_score)), np.concatenate((p_score, n_score)), np.concatenate((np.ones(len(p_score)), np.full(len(n_score), 2)))))
    return scores

# Concept Distance Threshold class implementation (CDT)
class CDT:

    def __init__(self, classifier, sizes=1000, repetitions=10, pos_prev = np.linspace(0, 1, 100)):
        self.classifier = classifier
        # Allow a single batch size (scalar) or multiple sizes (iterable)
        self.sizes = [sizes] if np.isscalar(sizes) else sizes
        self.repetitions = repetitions
        self.pos_prev = pos_prev
        self.distances = None
        self.thr = None

    # Train classifier with k-fold cross-validation and store the positive and negative scores.
    def _train_classifier(self, train):
        X_train = train.drop(columns=['class'])
        y_train = train['class']

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        fold_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            self.classifier.fit(X_fold_train, y_fold_train)
            proba = self.classifier.predict_proba(X_fold_val)
            fold_scores.append(np.column_stack((proba, y_fold_val)))

        train_scores = np.vstack(fold_scores)
        pos_scores = train_scores[train_scores[:, 2] == 1, 1].astype(float)
        neg_scores = train_scores[train_scores[:, 2] == 0, 1].astype(float)

        self.classifier.fit(train.drop(columns=['class']), train['class'])

        return pos_scores, neg_scores
    
    def _split_train(self, train):
        # 50% train, 50% validation, stratified by class
        train_data, validation_data = train_test_split(
            train, test_size=0.5, stratify=train['class'], random_state=42
        )
        return train_data, validation_data
    
    def _test_batch(self, validation, n_pos, size):
        val_pos = validation[validation['class'] == 1]
        val_neg = validation[validation['class'] == 0]

        batch = pd.concat([val_pos.sample(n=min(n_pos, len(val_pos))), val_neg.sample(n=min(size - n_pos, len(val_neg)))])
        test_scores = self.classifier.predict_proba(batch.drop(columns=['class']))[:, 1]

        return test_scores

    # Fit Concept Distance Threshold (CDT)
    def fit(self, train):
        from .quantifiers import DyS  # local import to avoid circular import

        distances = []

        # Split train into training and validation sets
        train, validation = self._split_train(train)
        pos_scores, neg_scores = self._train_classifier(train)

        # APP sampling and distance extraction
        for _ in range(self.repetitions):
            for size in self.sizes:
                for prev in self.pos_prev:
                    n_pos = int(round(size * prev))

                    test_scores = self._test_batch(validation, n_pos, size)

                    _, distance = DyS(pos_scores, neg_scores, test_scores, return_distance=True)
                    distances.append(distance)

        # Threshold calculation based on the extracted distances (mean + 2*std)
        self.distances = np.array(distances)
        self.thr = np.mean(distances) + (2 * np.std(distances))

    # Return if the test mean has concept drift based on the threshold.
    def predict(self, distance):
        return (distance >= self.thr)