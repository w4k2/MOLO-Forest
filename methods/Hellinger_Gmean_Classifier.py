from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y
from sklearn.metrics import recall_score
from sklearn.utils import resample
from imblearn.metrics import specificity_score
import numpy as np
from math import sqrt
from scipy.stats import mode
from sklearn.metrics import precision_score, recall_score
import numpy as np
from itertools import compress
import math
import matplotlib.pyplot as plt
import os
from random import randint
from sklearn.model_selection import train_test_split


class Hellinger_Gmean_Classifier(ClassifierMixin, BaseEnsemble):
    
    def __init__(self, base_estimator=None, max_bootstraps=10, K_steps_optimization=10, predict_decision="MV", scatter_plots=False, plot_iterator=0, pruning_ens_value=10, holdout=False):
        """Initialization."""
        self.base_estimator = base_estimator
        self.max_bootstraps = max_bootstraps
        self.K_steps_optimization = K_steps_optimization
        self.predict_decision = predict_decision
        self.scatter_plots = scatter_plots
        self.plot_iterator = plot_iterator
        self.pruning_ens_value = pruning_ens_value
        self.holdout = holdout

        self.all_ens_dict = {}
        self.pareto_ensemble_dict = {}
        self.plot_random_id = randint(0, 100)

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
        # utwórz słownik, w którym klucze to indeksy z candidate_models, a wartości to modele (pamiętaj, że klucz/indeksy zaczynają się od zera)
        candidate_models_dict = dict(enumerate(candidate_models))

        # Local optimization
        # jest tyle modeli, ile próbek bootstrapu
        for K_step in range(1, self.K_steps_optimization+1):
            print("K = ", K_step)
            if K_step == 1:
                metrics = []
                for model in candidate_models:
                    ensemble = []
                    ensemble.append(model)
                    y_pred = self._predict(X_test, ensemble)
                    metrics.append([
                        self.hellinger_distance(y_test, y_pred), 
                        self.geometric_mean_score_pr(recall_score(y_test, y_pred),precision_score(y_test, y_pred))])
                    
                    # Znalezienie niezdominowanych rozwiązań, tylko wtedy, gdy osiągnięty jest ostatni element listy candidate_models
                    if model == candidate_models[-1]:
                        non_dominant_solutions = self.is_pareto_efficient(np.array(metrics))
                        # słownik, który przechowuje liczbę danego kroku K, a w nim kolejne cyfry dla modeli znajdujących się w ensemble oraz jakość tego ensemble
                        pareto_ensemble_dict = {f"K{K_step}": {}}
                        # indeksy z non_dominant_solutions, które są True
                        non_dominant_indexes = list(compress(range(len(non_dominant_solutions)), non_dominant_solutions))
                        # uzupełnienie słownika numerami modeli, które są na pareto oraz jakością tego rozwiązania
                        for model_id in non_dominant_indexes:
                            # do słownika dopisz odpowiadającą wartość jakości dla danego modelu
                            pareto_ensemble_dict[f"K{K_step}"][f"{model_id}"] = metrics[model_id]

                        # PRUNING ENSEMBLI
                        # usuń z pareto_ensemble_dict te modele, które mają najgorszą jakość, a zostaw tylko pruning_ens_value najlepszych rozwiązań na pareto, czyli zmniejsz rozmiar słownika, tam gdzie jest f"E{ensemble_index}" - jest to tak jakby Beam Search, ale bez gotowej funkcji, tak samo jest w pozostałych krokach K
                        if len(pareto_ensemble_dict[f"K{K_step}"]) > self.pruning_ens_value:
                            # Tu lista z wynikami jest najpierw sortowana wg. drugiej metryki, a potem brane są tylko elementy, do momentu jak wskazuje zmienna self.pruning_ens_value i robiony jest z tego słownik
                            pareto_ensemble_dict[f"K{K_step}"] = dict(sorted(pareto_ensemble_dict[f"K{K_step}"].items(), key=lambda item: item[1][0])[:self.pruning_ens_value])

                # Wykres scatter plot z wszystkimi rozwiązaniami i rozwiązaniami niezdominowanymi
                if self.scatter_plots:
                    self.scatter_plot(metrics, non_dominant_solutions)

            if K_step == 2:
                metrics = []
                ensemble_index = 0
                all_ens_dict = {f"K{K_step}": {f"E{ensemble_index}": ""}}
                for model_id, model in enumerate(candidate_models):
                    for previous_model_id in pareto_ensemble_dict[f"K{K_step-1}"]:
                        # w związku z tym, że tu są 2 pętle, te modele mogą się powtarzać, bo jeśli model_id=0 i previous_model_id=4, a potem model_id=4 i previous_model_id=0, to wtedy w słowniku all_ens_dict będzie 0;4 i 4;0, a to jest to samo, więc trzeba dać jakiś warunek, żeby tego ponownie nie obliczać
                        if f"{previous_model_id};{model_id}" in all_ens_dict[f"K{K_step}"].values() or f"{model_id};{previous_model_id}" in all_ens_dict[f"K{K_step}"].values():
                            continue

                        ensemble = []
                        # warunek, że nie mogą to być takie same modele, jak były w poprzednim ensemble
                        if model_id != previous_model_id:
                            # to działa, jeśli ten klucz to jedna cyfra
                            ensemble.append(candidate_models_dict[int(previous_model_id)])
                            ensemble.append(model)
                            y_pred = self._predict(X_test, ensemble)
                            metrics.append([
                                self.hellinger_distance(y_test, y_pred), 
                                self.geometric_mean_score_pr(recall_score(y_test, y_pred),precision_score(y_test, y_pred))])
                        else:
                            # print("MODEL TAKI SAM")
                            metrics.append([0, 0])
                        # Zapisuj wszystkie przetestowane kombinacje do słownika, które znalazły się w ensemble, zapisz ich indeksy
                        all_ens_dict[f"K{K_step}"][f"E{ensemble_index}"] = f"{previous_model_id};{model_id}"
                        ensemble_index += 1

                    # Znalezienie niezdominowanych rozwiązań, tylko wtedy, gdy jest ostatni element listy candidate_models
                    if model == candidate_models[-1]:
                        non_dominant_solutions = self.is_pareto_efficient(np.array(metrics))
                        # słownik, który przechowuje liczbę danego kroku K, a w nim kolejne cyfry dla modeli znajdujących się w ensemble oraz jakość tego ensemble
                        pareto_ensemble_dict[f"K{K_step}"] = {}
                        # uzyskaj indeksy z non_dominant_solutions, które są True
                        non_dominant_ensembles_indexes = list(compress(range(len(non_dominant_solutions)), non_dominant_solutions))
                        # uzupełnienie słownika numerami modeli, które są na pareto oraz jakością tego rozwiązania
                        for ens_id in non_dominant_ensembles_indexes:
                            # do słownika dopisz odpowiadającą wartość jakości dla danego modelu
                            pareto_ensemble_dict[f"K{K_step}"][f"{ens_id}"] = metrics[ens_id]
                        
                        # PRUNING ENSEMBLI
                        if len(pareto_ensemble_dict[f"K{K_step}"]) > self.pruning_ens_value:
                            pareto_ensemble_dict[f"K{K_step}"] = dict(sorted(pareto_ensemble_dict[f"K{K_step}"].items(), key=lambda item: item[1][0])[:self.pruning_ens_value])

                # Wykres scatter plot z wszystkimi rozwiązaniami i rozwiązaniami niezdominowanymi
                if self.scatter_plots:
                    self.scatter_plot(metrics, non_dominant_solutions)

            if K_step > 2:
                metrics = []
                all_ens_dict[f"K{K_step}"] = {}
                ensemble_index = 0
                for model_id, model in enumerate(candidate_models):
                    for previous_ens_id in pareto_ensemble_dict[f"K{K_step-1}"]:
                        # w związku z tym, że tu są 2 pętle, te modele mogą się powtarzać, bo jeśli model_id=0 i previous_model_id=4, a potem model_id=4 i previous_model_id=0, to wtedy w słowniku all_ens_dict będzie 0;4 i 4;0, a to jest to samo, więc trzeba dać jakiś warunek, żeby tego ponownie nie obliczać
                        if f"{previous_model_id};{model_id}" in all_ens_dict[f"K{K_step}"].values() or f"{model_id};{previous_model_id}" in all_ens_dict[f"K{K_step}"].values():
                            continue

                        ensemble = []
                        # warunek, że nie mogą to być takie same modele, jak były w poprzednim ensemble
                        if model_id != previous_model_id:
                            prev_models_id_list = all_ens_dict[f"K{K_step-1}"][f"E{previous_ens_id}"].split(";")
                            for prev_m_id in prev_models_id_list:
                                ensemble.append(candidate_models_dict[int(prev_m_id)])
                            ensemble.append(model)
                            y_pred = self._predict(X_test, ensemble)
                            metrics.append([
                                self.hellinger_distance(y_test, y_pred), 
                                self.geometric_mean_score_pr(recall_score(y_test, y_pred),precision_score(y_test, y_pred))])
                        else:
                            # print("MODEL TAKI SAM")
                            metrics.append([0, 0])
                        # Zapisuj wszystkie przetestowane kombinacje do słownika, które znalazły się w ensemble, zapisz indeksy tych modeli, modeli jest tyle co K_step
                        value = ""
                        for previous_model_id in prev_models_id_list:
                            # dodaj kolejne modele do stringa, który jest wartością słownika
                            value += f"{previous_model_id};" 
                        value += f"{model_id}"
                        all_ens_dict[f"K{K_step}"][f"E{ensemble_index}"] = str(value)

                        ensemble_index += 1

                    # Znalezienie niezdominowanych rozwiązań, tylko wtedy, gdy jest ostatni element listy candidate_models
                    if model == candidate_models[-1]:
                        non_dominant_solutions = self.is_pareto_efficient(np.array(metrics))
                        # słownik, który przechowuje liczbę danego kroku K, a w nim kolejne cyfry dla modeli znajdujących się w ensemble oraz jakość tego ensemble
                        pareto_ensemble_dict[f"K{K_step}"] = {}
                        # uzyskaj indeksy z non_dominant_solutions, które są True
                        non_dominant_ensembles_indexes = list(compress(range(len(non_dominant_solutions)), non_dominant_solutions))
                        # uzupełnienie słownika numerami modeli, które są na pareto oraz jakością tego rozwiązania
                        for ens_id in non_dominant_ensembles_indexes:
                            # do słownika dopisz odpowiadającą wartość jakości dla danego modelu
                            pareto_ensemble_dict[f"K{K_step}"][f"{ens_id}"] = metrics[ens_id]
                        
                        # PRUNING ENSEMBLI
                        if len(pareto_ensemble_dict[f"K{K_step}"]) > self.pruning_ens_value:
                            pareto_ensemble_dict[f"K{K_step}"] = dict(sorted(pareto_ensemble_dict[f"K{K_step}"].items(), key=lambda item: item[1][0])[:self.pruning_ens_value])
                        
                if self.scatter_plots:
                    self.scatter_plot(metrics, non_dominant_solutions)

        # wybierz jeden ensemble wg. metryki gmean(największa wartość) z pareto_ensemble_dict z ostatniego kroku K
        ensemble = []
        final_ensemble = dict(sorted(pareto_ensemble_dict[f"K{K_step}"].items(), key=lambda item: item[1][0])[:1])
        final_ensemble_id = list(final_ensemble.keys())[0]
        final_models_id_list = all_ens_dict[f"K{K_step}"][f"E{final_ensemble_id}"].split(";")
        for fin_m_id in final_models_id_list:
            ensemble.append(candidate_models_dict[int(fin_m_id)])
        self.ensemble = ensemble

        # do zapisania wyników do pliku w eksperymencie głównym
        self.all_ens_dict = all_ens_dict
        self.pareto_ensemble_dict = pareto_ensemble_dict

        return self
            

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble])
    
    # Predict for inside use
    def _predict(self, X, ensemble):
        # Prediction based on the Majority Voting
        predictions = np.array([member_clf.predict(X) for member_clf in ensemble])
        prediction = np.squeeze(mode(predictions, axis=0)[0])
        return self.classes_[prediction]
    
    def predict(self, X):
        # Prediction based on the Average Support Vectors
        if self.predict_decision == "ASV":
            ens_sup_matrix = self.ensemble_support_matrix(X)
            average_support = np.mean(ens_sup_matrix, axis=0)
            prediction = np.argmax(average_support, axis=1)
        # Prediction based on the Majority Voting
        elif self.predict_decision == "MV":
            # member_clf.predict(X) dotyczy funkcji predict bazowego klasyfikatora (tu Decision Tree), a nie mojej zdefiniowanej funkcji
            predictions = np.array([member_clf.predict(X) for member_clf in self.ensemble])
            prediction = np.squeeze(mode(predictions, axis=0)[0])
        return self.classes_[prediction]

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble]
        return np.average(probas_, axis=0)
    
    # Calculate Hellinger distance based on ref. [2] 
    def hellinger_distance(self, y, y_pred):
        # TPR - True Positive Rate (or sensitivity, recall, hit rate)
        tprate = recall_score(y, y_pred)
        # TNR - True Negative Rate (or specificity, selectivity)
        tnrate = specificity_score(y, y_pred)
        # FPR - False Positive Rate (or fall-out)
        fprate = 1 - tnrate 
        # FNR - False Negative Rate (or miss rate)
        fnrate = 1 - tprate
        # Calculate Hellinger distance
        if tprate > fnrate:
            hd = sqrt((sqrt(tprate)-sqrt(fprate))**2 + (sqrt(1-tprate)-sqrt(1-fprate))**2)
        else:
            hd = 0
        return hd
    
    # Calculate geometric_mean_score based on Precision and Recall
    def geometric_mean_score_pr(self, recall, precision):
        return math.sqrt(recall*precision)
    
    # coś z tego: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    def is_pareto_efficient(self, costs, return_mask = True):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs>=costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient
        
    def scatter_plot(self, metrics_data, non_dominant_data):
        experiment_name = "experiment1"
        filename_pareto_chart = "results/%s/scatter_plots/idr%d_scatter%d" % (experiment_name, self.plot_random_id, self.plot_iterator)
        if not os.path.exists("results/%s/scatter_plots/" % (experiment_name)):
            os.makedirs("results/%s/scatter_plots/" % (experiment_name))

        self.plot_iterator += 1

        plt.grid(True, color="silver", linestyle=":", axis='both')
        metrics_data = np.array(metrics_data)
        # solutions metrics data
        sol_x = metrics_data[:, 0]
        sol_y = metrics_data[:,1]
        plt.scatter(sol_x, sol_y, color='darkgray', marker="o", label="ALL")

        # pareto front solution
        non_dominant = np.array(list(compress(metrics_data, non_dominant_data)))
        pf_x = non_dominant[:, 0]
        pf_y = non_dominant[:,1]
        plt.scatter(pf_x, pf_y, color='black', marker="o", label="PF")

        plt.xlabel('Hellinger', fontsize=12)
        plt.ylabel('G_mean', fontsize=12)
        # plt.xlim([0, 1.1])
        # plt.ylim([0, 1.1])
        plt.legend(loc="best")
        plt.gcf().set_size_inches(9, 6)
        plt.savefig(filename_pareto_chart+".png", bbox_inches='tight')
        plt.savefig(filename_pareto_chart+".eps", format='eps', bbox_inches='tight')
        plt.clf()
        plt.close()
