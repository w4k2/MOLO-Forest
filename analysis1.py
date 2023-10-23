import json
import numpy as np
import os
from itertools import compress
from pathlib import Path
from utils.load_datasets import load_dataset
from utils.plots import ensemble_plot_final_K, scatter_plot


from utils.wilcoxon_ranking import pairs_metrics_multi_grid_all, pairs_metrics_multi_line
from utils.datasets_table_description import make_description_table
from utils.load_datasets import load_dataset
from utils.datasets_table_description import calc_imbalance_ratio
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from methods.Hellinger_Gmean_Classifier import Hellinger_Gmean_Classifier
from methods.Margin_Diversity_Classifier import Margin_Diversity_Classifier
from methods.Ensemble_All_Models import Ensemble_All_Models
from utils.plots import ensemble_plot

from methods.Margin_Diversity_Hellinger_Classifier import Margin_Diversity_Hellinger_Classifier


base_estimator = DecisionTreeClassifier(random_state=1234)
# max_bootstrap is the same as number of classfiers in ensemble
max_bootstraps = 100
# 100
# K_steps_optimization is how many iterations is in creation of ensemble
K_steps_optimization = 20
# 20
methods = {
    "DT":
        DecisionTreeClassifier(random_state=1234),
    "Margin_Diversity_Hellinger_Classifier":
        Margin_Diversity_Hellinger_Classifier(base_estimator=base_estimator, max_bootstraps=max_bootstraps, K_steps_optimization=K_steps_optimization, predict_decision="MV", scatter_plots=False, plot_iterator=0, pruning_ens_value=10),
    "Margin_Diversity_Hellinger_Classifier_h":
        Margin_Diversity_Hellinger_Classifier(base_estimator=base_estimator, max_bootstraps=max_bootstraps, K_steps_optimization=K_steps_optimization, predict_decision="MV", scatter_plots=False, plot_iterator=0, pruning_ens_value=10, holdout=True),
    "Hellinger_Gmean_Classifier_h":
        Hellinger_Gmean_Classifier(base_estimator=base_estimator, max_bootstraps=max_bootstraps, K_steps_optimization=K_steps_optimization, predict_decision="MV", scatter_plots=False, plot_iterator=0, pruning_ens_value=10, holdout=True),
    "Hellinger_Gmean_Classifier":
        Hellinger_Gmean_Classifier(base_estimator=base_estimator, max_bootstraps=max_bootstraps, K_steps_optimization=K_steps_optimization, predict_decision="MV", scatter_plots=False, plot_iterator=0, pruning_ens_value=10),
    "Margin_Diversity_Classifier_h":
        Margin_Diversity_Classifier(base_estimator=base_estimator, max_bootstraps=max_bootstraps, K_steps_optimization=K_steps_optimization, predict_decision="MV", scatter_plots=False, plot_iterator=0, pruning_ens_value=10, holdout=True),
    "Ensemble_All_Models":
        Ensemble_All_Models(base_estimator=base_estimator, max_bootstraps=max_bootstraps, predict_decision="MV"),
    "RF":
        RandomForestClassifier(random_state=1234, bootstrap=False, n_estimators=max_bootstraps),
    "RF_b":
        RandomForestClassifier(random_state=1234, bootstrap=True, n_estimators=max_bootstraps),
}
method_names = list(methods.keys())

metrics_alias = [
    "BAC",
    "F1score",
    "Specificity",
    "Recall",
    "Precision",
    "Gmean"]

# Load datasets and names
DATASETS_DIR = "datasets_60/"
dataset_paths = []
dataset_names = []
imbalance_ratios = []
for root, _, files in os.walk(DATASETS_DIR):
    # print(root, files)
    for filename in filter(lambda _: _.endswith('.dat'), files):
        dataset_paths.append(os.path.join(root, filename))
        dataset_path = os.path.join(root, filename)
        dataset_name = Path(dataset_path).stem
        dataset_names.append(dataset_name)
        X, y = load_dataset(dataset_path)
        IR = calc_imbalance_ratio(X, y)
        imbalance_ratios.append(IR)

n_folds = 10
n_methods = len(methods)
n_metrics = len(metrics_alias)
n_datasets = len(dataset_paths)
# Load data from file
data_np = np.zeros((n_datasets, n_metrics, n_methods, n_folds))
mean_scores = np.zeros((n_datasets, n_metrics, n_methods))
stds = np.zeros((n_datasets, n_metrics, n_methods))
sum_times = np.zeros((n_datasets, len(methods)))

for dataset_id, dataset_path in enumerate(dataset_paths):
    dataset_name = Path(dataset_path).stem
    for clf_id, clf_name in enumerate(methods):
        for metric_id, metric in enumerate(metrics_alias):
            try:
                filename = "results/experiment1/raw_results/%s/%s/%s.csv" % (metric, dataset_name, clf_name)
                if not os.path.isfile(filename):
                    print("File not exist - %s" % filename)
                    # continue
                scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                data_np[dataset_id, metric_id, clf_id] = scores
                mean_score = np.mean(scores)
                mean_scores[dataset_id, metric_id, clf_id] = mean_score
                std = np.std(scores)
                stds[dataset_id, metric_id, clf_id] = std
            except:
                print("Error loading data!", dataset_name, clf_name, metric)
        try:
            filename = "results/experiment1/time_results/%s/%s_time.csv" % (dataset_name, clf_name)
            if not os.path.isfile(filename):
                # print("File not exist - %s" % filename)
                continue
            times = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
            sum_times[dataset_id, clf_id] = sum(times)
        except:
            print("Error loading time data!", dataset_name, clf_name)

# All datasets with description in the table
# make_description_table(DATASETS_DIR)

experiment_name = "experiment1"
# Results in form of one .tex table of each metric
# result_tables(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)

# Results in form of one .tex table of each metric sorted by IR
# result_tables_IR(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)

# Results in form of one .tex table of each metric sorted by number of features
# result_tables_features(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)

# Wilcoxon ranking grid - statistic test for all methods
# pairs_metrics_multi_grid_all(method_names=method_names, data_np=data_np, experiment_name=experiment_name, dataset_paths=dataset_paths, metrics=metrics_alias, filename="ex1_wilcoxon_all", ref_methods=list(method_names)[0:2], offset=-10)

# Wilcoxon ranking line - statistic test for my method vs the remaining methods
# pairs_metrics_multi_line(method_names=list(method_names), data_np=data_np, experiment_name=experiment_name, dataset_paths=dataset_paths, metrics=metrics_alias, filename="ex1_wilcoxon", ref_methods=list(method_names))

# Time results in form of .tex table
# result_tables_for_time(dataset_names, imbalance_ratios, sum_times, methods, experiment_name)

### Wykresy z rozwiązań dict
pareto_ens_dict_all = {}
solution_dict_final_K = {}
annotations = {}
for dataset_id, dataset_path in enumerate(dataset_paths):
    dataset_name = Path(dataset_path).stem
    pareto_ens_dict_all[f"{dataset_name}"] = {}
    solution_dict_final_K[f"{dataset_name}"] = {}
    annotations[f"{dataset_name}"] = {}
    for fold_id in range(n_folds):
        try:
            filename = "results/experiment1/dict/%s/pareto_ens_dict_fold%d.json" % (dataset_name, fold_id)
            if not os.path.isfile(filename):
                print("File not exist - %s" % filename)
            # wczytaj dane z formatu json z pliku i zapisz do dict
            pareto_ens_dict = json.load(open(filename))
            # zapisz pareto_ens_dict do słownika
            pareto_ens_dict_all[f"{dataset_name}"][f"{fold_id}"] = pareto_ens_dict
            solution_dict_final_K[f"{dataset_name}"][f"{fold_id}"] = pareto_ens_dict[f"K{K_steps_optimization}"]
            annotations[f"{dataset_name}"][f"{fold_id}"] = list(pareto_ens_dict[f"K{K_steps_optimization}"].keys())
        except:
            print("Error loading dict data!", dataset_name)
print(pareto_ens_dict_all)
# K_steps_optimization, gmean - druga metryka
# ensemble_plot(datasets=dataset_names, n_folds=n_folds, experiment_name=experiment_name, solution_dict=pareto_ens_dict_all, K_steps_optimization=K_steps_optimization)

# Wykres, który zawiera wszystkie ensemble z ostatniej iteracji (K_steps_optimization), oś x - jedna metryka (gmean), oś y - druga metryka (hellinger)
# ensemble_plot_final_K(datasets=dataset_names, n_folds=n_folds, experiment_name=experiment_name, solution_dict_final_K=solution_dict_final_K, K_steps_optimization=K_steps_optimization, annotations=annotations)

