# Reference methods based on 
# Mohammed, A.M., Onieva, E., Wo ́zniak, M., Mart ́ınez-Mu ̃noz, G.: An analysis of heuristic metrics for classifier ensemble pruning based on ordered aggregation. Pattern Recognition 124, 108493 (2022). https://doi.org/https://doi.org/10.1016/j.patcog.2021.108493, https://www.sciencedirect.com/science/article/pii/S0031320321006695
# and PyPruning library
# https://github.com/sbuschjaeger/PyPruning


from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy.stats import mode
import numpy as np
from PyPruning.GreedyPruningClassifier import GreedyPruningClassifier, margin_distance, reduced_error, complementariness


class OrderBasePruningEnsemble(ClassifierMixin, BaseEnsemble):
    
    def __init__(self, base_estimator=None, max_bootstraps=10, K_steps_optimization=10, predict_decision="MV", holdout=False, metric_name="margin", alpha=0.5):
        """Initialization."""
        self.base_estimator = base_estimator
        self.max_bootstraps = max_bootstraps
        self.K_steps_optimization = K_steps_optimization
        self.predict_decision = predict_decision
        self.holdout = holdout
        self.metric_name = metric_name
        self.alpha = alpha

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

        # train-test split = holdout cross-validation
        if self.holdout == True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            # Create bootstraps samples
            X_b = []
            y_b = []
            for random_state in range(self.max_bootstraps):
                Xy_bootstrap = resample(X_train, y_train, replace=True, random_state=random_state)
                X_b.append(Xy_bootstrap[0])
                y_b.append(Xy_bootstrap[1])
        else:
            X_test = X
            y_test = y
            # Create bootstraps samples
            X_b = []
            y_b = []
            for random_state in range(self.max_bootstraps):
                Xy_bootstrap = resample(X, y, replace=True, random_state=random_state)
                X_b.append(Xy_bootstrap[0])
                y_b.append(Xy_bootstrap[1])

        # Train estimators based on each bootstrapped dataset
        candidate_models = []
        for X_b_sample, y_b_sample in zip(X_b, y_b):
            candidate = clone(self.base_estimator).fit(X_b_sample, y_b_sample)
            candidate_models.append(candidate)

        metric = {
            'margin': self.margin,
            'diversity': self.diversity,
            'margin_diversity': self.margin_and_diversity,
            'reduced_error': reduced_error,
            'complementariness': complementariness,
            'margin_org': margin_distance,
        }


        prune_model = GreedyPruningClassifier(metric=metric[self.metric_name], n_estimators=self.K_steps_optimization)
        self.ensemble = prune_model.prune(X, y, candidate_models).estimators_

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
            self.predictions = predictions
            prediction = np.squeeze(mode(predictions, axis=0)[0])
        return self.classes_[prediction]
    
    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble]
        return np.average(probas_, axis=0)
    

    # funkcja margin - trochę zmieniona ze względu na użycie biblioteki PyPruning
    def margin(self, i, ensemble_proba, selected_models, target):
        # policz ile głosów z każdego modelu zgadza się z prawdziwą klasą z y dla każdej próbki w X i zapisz to do listy

        # przetworzenie danych wejściowych
        if len(selected_models) == 0:
            val = np.random.uniform(0, 1)
            return val
        else:
            selected_proba = np.concatenate((ensemble_proba[selected_models,:,:],ensemble_proba[[i],:,:]), axis=0)
        y = target
        predictions = np.argmax(selected_proba, axis=2)
        average_support = np.mean(selected_proba, axis=0)
        y_pred = np.argmax(average_support, axis=1)
        ensemble_size = len(selected_models)

        binary_votes = []
        votes_pred_sum_0 = []
        votes_pred_sum_1 = []
        for i in range(predictions.shape[0]):
            vote_line = []
            votes_pred_sum_0_line = []
            votes_pred_sum_1_line = []
            for j, yj in zip(range(predictions.shape[1]), y):
                vote_line.append(predictions[i][j] == yj)
                votes_pred_sum_0_line.append(predictions[i][j] == 0)
                votes_pred_sum_1_line.append(predictions[i][j] == 1)
            binary_votes.append(vote_line)
            votes_pred_sum_0.append(votes_pred_sum_0_line)
            votes_pred_sum_1.append(votes_pred_sum_1_line)
        # zsumuj głosy z każdego modelu dla każdej próbki w X
        votes_true = np.sum(binary_votes, axis=0)
        # oblicz sumę głosów dla każdej próbki w X osobno dla każdej klasy i zapisz to do list
        votes_pred_sum_0 = np.sum(votes_pred_sum_0, axis=0)
        votes_pred_sum_1 = np.sum(votes_pred_sum_1, axis=0)

        # oblicz różnicę między głosami dla prawdziwej klasy a głosami dla klasy przewidzianej przez model
        votes_subtract = []
        for i in range(len(votes_pred_sum_0)):
            if y_pred[i] == 0:
                votes_subtract.append(votes_true[i] - votes_pred_sum_1[i])
            elif y_pred[i] == 1:
                votes_subtract.append(votes_true[i] - votes_pred_sum_0[i])
        # podziel każdy element listy votes_subtract przez liczbę klasyfikatorów w ensemble
        votes_divide = np.array(votes_subtract)/ensemble_size
        # jeśli jakakolwiek wartość w votes jest równa 0 to zamień ją na 0,000001, żeby nie było logarytmu z 0
        votes_divide = [0.000001 if x==0 else x for x in votes_divide]
        # zrób wartość bezwględną w votes_divide, żeby były tylko nieujemne wartości
        votes_divide = np.absolute(votes_divide)
        # policz logarytm z każdego elementu listy votes_divide
        log_votes = np.log10(votes_divide)
        margin = np.sum(log_votes)

        return margin
    
    # funkcja diversity - trochę zmieniona ze względu na użycie biblioteki PyPruning
    def diversity(self, i, ensemble_proba, selected_models, target):
        # przetworzenie danych wejściowych
        if len(selected_models) == 0:
            val = np.random.uniform(0, 1)
            return val
        else:
            selected_proba = np.concatenate((ensemble_proba[selected_models,:,:],ensemble_proba[[i],:,:]), axis=0)
        y = target
        predictions = np.argmax(selected_proba, axis=2)
        ensemble_size = len(selected_models)

        diversity = 0
        # policz ile głosów z każdego modelu zgadza się z prawdziwą klasą z y dla każdej próbki w X i zapisz to do listy
        binary_votes = []
        for i in range(predictions.shape[0]):
            vote_line = []
            for j, yj in zip(range(predictions.shape[1]), y):
                vote_line.append(predictions[i][j] == yj)
            binary_votes.append(vote_line)

        # zsumuj głosy z każdego modelu dla każdej próbki w X
        votes = np.sum(binary_votes, axis=0)

        # jeśli jakakolwiek wartość w votes jest równa 0 to zamień ją na 0,000001, żeby nie było logarytmu z 0
        votes = [0.000001 if x==0 else x for x in votes]
        # podziel każdy element listy votes przez liczbę klasyfikatorów w ensemble
        votes_divide = np.array(votes)/ensemble_size
        log_votes = np.log10(votes_divide)
        diversity = np.sum(log_votes)
        return diversity
    
    # metryka margin&diversity z parametrem alpha - agregacja
    def margin_and_diversity(self, i, ensemble_proba, selected_models, target):

        margin = self.margin(i, ensemble_proba, selected_models, target)
        diversity = self.diversity(i, ensemble_proba, selected_models, target)

        return (margin*self.alpha+diversity*(1-self.alpha))