import json
import numpy as np
import os
import time
from joblib import Parallel, delayed
import logging
import traceback
from pathlib import Path
import warnings
from imblearn.metrics import specificity_score

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone

from methods.Hellinger_Gmean_Classifier import Hellinger_Gmean_Classifier
from methods.Ensemble_All_Models import Ensemble_All_Models
from methods.Margin_Diversity_Classifier import Margin_Diversity_Classifier
from methods.Margin_Diversity_Hellinger_Classifier import Margin_Diversity_Hellinger_Classifier
from utils.load_datasets import load_dataset
import math
from methods.ref_methods import OrderBasePruningEnsemble
from PyPruning.GreedyPruningClassifier import GreedyPruningClassifier, margin_distance


# Calculate geometric_mean_score based on Precision and Recall
def geometric_mean_score_pr(recall, precision):
    return math.sqrt(recall*precision)

"""
Datasets are from KEEL repository.
"""

# Calculate results for reference methods

base_estimator = DecisionTreeClassifier(random_state=1234)
# max_bootstrap is the same as number of classfiers in ensemble
max_bootstraps = 100
# 100
# K_steps_optimization is how many iterations is in creation of ensemble
K_steps_optimization = 20
# 20
methods = {
    "OBPE-margin":
        OrderBasePruningEnsemble(base_estimator=base_estimator, max_bootstraps=max_bootstraps, K_steps_optimization=K_steps_optimization, predict_decision="MV", holdout=False, metric_name="margin", alpha=0.5),
    
    "OBPE-diversity":
        OrderBasePruningEnsemble(base_estimator=base_estimator, max_bootstraps=max_bootstraps, K_steps_optimization=K_steps_optimization, predict_decision="MV", holdout=False, metric_name="diversity", alpha=0.5),

    "OBPE-mardiv":
        OrderBasePruningEnsemble(base_estimator=base_estimator, max_bootstraps=max_bootstraps, K_steps_optimization=K_steps_optimization, predict_decision="MV", holdout=False, metric_name="margin_diversity", alpha=0.5),
    
    "OBPE-reduced_error":
        OrderBasePruningEnsemble(base_estimator=base_estimator, max_bootstraps=max_bootstraps, K_steps_optimization=K_steps_optimization, predict_decision="MV", holdout=False, metric_name="reduced_error", alpha=0.5),

    "OBPE-complementariness":
        OrderBasePruningEnsemble(base_estimator=base_estimator, max_bootstraps=max_bootstraps, K_steps_optimization=K_steps_optimization, predict_decision="MV", holdout=False, metric_name="complementariness", alpha=0.5),
    
    "OBPE-margin_org":
        OrderBasePruningEnsemble(base_estimator=base_estimator, max_bootstraps=max_bootstraps, K_steps_optimization=K_steps_optimization, predict_decision="MV", holdout=False, metric_name="margin_org", alpha=0.5),

}

# Repeated Stratified K-Fold cross validator
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=111)
n_folds = n_splits * n_repeats

DATASETS_DIR = "datasets_60/"
# DATASETS_DIR = "datasets_test/"
dataset_paths = []
for root, _, files in os.walk(DATASETS_DIR):
    for filename in filter(lambda _: _.endswith('.dat'), files):
        dataset_paths.append(os.path.join(root, filename))

metrics = [
    balanced_accuracy_score,
    f1_score,
    specificity_score,
    recall_score,
    precision_score,
    geometric_mean_score_pr # Gmean based on Precision and Recall
    ]
metrics_alias = [
    "BAC",
    "F1score",
    "Specificity",
    "Recall",
    "Precision",
    "Gmean"]

if not os.path.exists("textinfo"):
    os.makedirs("textinfo")
logging.basicConfig(filename='textinfo/experiment3.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
logging.info("--------------------------------------------------------------------------------")
logging.info("-------                        NEW EXPERIMENT                            -------")
logging.info("--------------------------------------------------------------------------------")


def compute(dataset_id, dataset_path):
    logging.basicConfig(filename='textinfo/experiment3.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
    try:
        warnings.filterwarnings("ignore")

        X, y = load_dataset(dataset_path)
        # Normalization - transform data to [0, 1]
        X = MinMaxScaler().fit_transform(X, y)
        scores = np.zeros((len(metrics), len(methods), n_folds))
        # diversity = np.zeros((len(methods), n_folds, 4))
        time_for_all = np.zeros((len(methods), n_folds))
        dataset_name = Path(dataset_path).stem

        print("START: %s" % (dataset_path))
        logging.info("START - %s" % (dataset_path))
        start = time.time()

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            for clf_id, clf_name in enumerate(methods):
                start_method = time.time()
                clf = clone(methods[clf_name])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Scores for each metric
                for metric_id, metric in enumerate(metrics):
                    if metric_id==5: # if Gmean
                        scores[metric_id, clf_id, fold_id] = metric(scores[3, clf_id, fold_id], scores[4, clf_id, fold_id])
                    else:
                        scores[metric_id, clf_id, fold_id] = metric(y_test, y_pred)

                end_method = time.time() - start_method
                logging.info("DONE METHOD %s - %s fold: %d (Time: %f [s])" % (clf_name, dataset_path, fold_id, end_method))
                print("DONE METHOD %s - %s fold: %d (Time: %.2f [s])" % (clf_name, dataset_path, fold_id, end_method))

                time_for_all[clf_id, fold_id] = end_method

        # Save results to csv
        for clf_id, clf_name in enumerate(methods):
            # Save metric results
            for metric_id, metric in enumerate(metrics_alias):
                filename = "results/experiment3/raw_results/%s/%s/%s.csv" % (metric, dataset_name, clf_name)
                if not os.path.exists("results/experiment3/raw_results/%s/%s/" % (metric, dataset_name)):
                    os.makedirs("results/experiment3/raw_results/%s/%s/" % (metric, dataset_name))
                np.savetxt(fname=filename, fmt="%f", X=scores[metric_id, clf_id, :])
            # Save time
            filename = "results/experiment3/time_results/%s/%s_time.csv" % (dataset_name, clf_name)
            if not os.path.exists("results/experiment3/time_results/%s/" % (dataset_name)):
                os.makedirs("results/experiment3/time_results/%s/" % (dataset_name))
            np.savetxt(fname=filename, fmt="%f", X=time_for_all[clf_id, :])

        end = time.time() - start
        logging.info("DONE - %s (Time: %d [s])" % (dataset_path, end))
        print("DONE - %s (Time: %d [s])" % (dataset_path, end))

    except Exception as ex:
        logging.exception("Exception in %s" % (dataset_path))
        print("ERROR: %s" % (dataset_path))
        traceback.print_exc()
        print(str(ex))


# Multithread; n_jobs - number of threads, where -1 all threads, safe for my computer 2
Parallel(n_jobs=-1)(
                delayed(compute)
                (dataset_id, dataset_path)
                for dataset_id, dataset_path in enumerate(dataset_paths)
                )
