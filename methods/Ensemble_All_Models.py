from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y
from sklearn.utils import resample
import numpy as np
from scipy.stats import mode
import numpy as np


class Ensemble_All_Models(ClassifierMixin, BaseEnsemble):
    
    def __init__(self, base_estimator=None, max_bootstraps=10, predict_decision="MV"):
        """Initialization."""
        self.base_estimator = base_estimator
        self.max_bootstraps = max_bootstraps
        self.predict_decision = predict_decision

    def fit(self, X, y):
        self.ensemble = []
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Create bootstraps samples
        X_b = []
        y_b = []
        for random_state in range(self.max_bootstraps):
            Xy_bootstrap = resample(X, y, replace=True, random_state=random_state)
            X_b.append(Xy_bootstrap[0])
            y_b.append(Xy_bootstrap[1])

        # Train estimators based on each bootstrapped dataset
        for X_b_sample, y_b_sample in zip(X_b, y_b):
            candidate = clone(self.base_estimator).fit(X_b_sample, y_b_sample)
            # Add all models to ensemble
            self.ensemble.append(candidate)
        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble])
    
    def predict(self, X):
        # Prediction based on the Average Support Vectors
        if self.predict_decision == "ASV":
            ens_sup_matrix = self.ensemble_support_matrix(X)
            average_support = np.mean(ens_sup_matrix, axis=0)
            prediction = np.argmax(average_support, axis=1)
        # Prediction based on the Majority Voting
        elif self.predict_decision == "MV":
            predictions = np.array([member_clf.predict(X) for member_clf in self.ensemble])
            prediction = np.squeeze(mode(predictions, axis=0)[0])
        return self.classes_[prediction]

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble]
        return np.average(probas_, axis=0)
    