import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..quantifiers import DyS
from .base import DriftDetector


# Concept Distance Threshold class implementation (CDT)
class CDT(DriftDetector):
    name = "cdt"
    per_class = True

    def __init__(self, classifier=None, sizes=1000, repetitions=10, pos_prev = np.linspace(0, 1, 100), measure="topsoe"):
        self.classifier = classifier
        # Allow a single batch size (scalar) or multiple sizes (iterable)
        self.sizes = [sizes] if np.isscalar(sizes) else sizes
        self.repetitions = repetitions
        self.pos_prev = pos_prev
        self.measure = measure
        self.distances = None
        self.thr_upper, self.thr_lower = None, None

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

                    _, distance = DyS(pos_scores, neg_scores, test_scores, return_distance=True, measure=self.measure)
                    distances.append(distance)

        # Threshold calculation based on the extracted distances (mean + 2*std)
        self.distances = np.array(distances)
        sd = np.std(distances)
        self.thr_upper = np.mean(distances) + (2 * sd)
        self.thr_lower = np.mean(distances) - (2 * sd)
        return self

    # DyS distance between the training score distributions and the test batch.
    def statistic(self, ctx):
        _, distance = DyS(ctx.pos_scores, ctx.neg_scores, ctx.test_scores, return_distance=True, measure=self.measure)
        return distance
