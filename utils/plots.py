import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import rankdata
from scipy import stats
import Orange
from .load_datasets import load_dataset
from .datasets_table_description import calc_imbalance_ratio

# Plot scatter of pareto front solutions and all methods
def scatter_plot(datasets, n_folds, experiment_name, data_solutions):
    for dataset_id, dataset in enumerate(datasets):
        print(dataset)
        for fold_id in range(n_folds):
            filename_pareto_chart = "results/%s/scatter_plots/%s/scatter_%s_fold%d" % (experiment_name, dataset, dataset, fold_id)
            if not os.path.exists("results/%s/scatter_plots/%s/" % (experiment_name, dataset)):
                os.makedirs("results/%s/scatter_plots/%s/" % (experiment_name, dataset))

            moo_x = []
            moo_y = []
            for solution in data_solutions:
                moo_x.append(solution[0])
                moo_y.append(solution[1])
            moo_x = np.array(moo_x)
            moo_y = np.array(moo_y)

            plt.grid(True, color="silver", linestyle=":", axis='both')

            # MOOforest pareto
            plt.scatter(moo_x, moo_y, color='darkgray', marker="o", label="MOOforest PF")


    n_rows_p = 1000
    for dataset_id, dataset in enumerate(datasets):
        print(dataset)
        for fold_id in range(n_folds):
            solutions_moo = []
            for sol_id in range(n_rows_p):
                try:
                    filename_pareto_semoos = "results/%s/pareto_raw/%s/MOOforest/fold%d/sol%d.csv" % (experiment_name, dataset, fold_id, sol_id)
                    solution_moo = np.genfromtxt(filename_pareto_semoos, dtype=np.float32)
                    solution_moo = solution_moo.tolist()
                    solution_moo[0] = solution_moo[0] * (-1)
                    solution_moo[1] = solution_moo[1] * (-1)
                    solutions_moo.append(solution_moo)
                except IOError:
                    pass
            if solutions_moo:
            #  and solutions_semoosb and solutions_semoosbp:
                filename_pareto_chart = "results/%s/scatter_plots/%s/scatter_%s_fold%d" % (experiment_name, dataset, dataset, fold_id)
                if not os.path.exists("results/%s/scatter_plots/%s/" % (experiment_name, dataset)):
                    os.makedirs("results/%s/scatter_plots/%s/" % (experiment_name, dataset))

                moo_x = []
                moo_y = []
                for solution in solutions_moo:
                    moo_x.append(solution[0])
                    moo_y.append(solution[1])
                moo_x = np.array(moo_x)
                moo_y = np.array(moo_y)
                plt.grid(True, color="silver", linestyle=":", axis='both')

                # MOOforest pareto
                plt.scatter(moo_x, moo_y, color='darkgray', marker="o", label="MOOforest PF")
                # Precision
                moo_precision = raw_data[dataset_id, 4, 0, fold_id]
                # Recall
                moo_recall = raw_data[dataset_id, 3, 0, fold_id]
                plt.scatter(moo_precision, moo_recall, color='black', marker="o", label="MOOforest")

                # DT
                plt.scatter(raw_data[dataset_id, 4, 1, fold_id], raw_data[dataset_id, 3, 1, fold_id], color='tab:pink', marker=">", label="DT")
                # RF
                plt.scatter(raw_data[dataset_id, 4, 2, fold_id], raw_data[dataset_id, 3, 2, fold_id], color='tab:blue', marker="v", label="RF")
                # RF_b
                plt.scatter(raw_data[dataset_id, 4, 3, fold_id], raw_data[dataset_id, 3, 3, fold_id], color='tab:orange', marker="v", label="RF_b")
                # DE_Forest
                plt.scatter(raw_data[dataset_id, 4, 4, fold_id], raw_data[dataset_id, 3, 4, fold_id], color='tab:blue', marker="+", label="DE_Forest")
                # RandomFS
                plt.scatter(raw_data[dataset_id, 4, 5, fold_id], raw_data[dataset_id, 3, 5, fold_id], color='tab:red', marker="^", label="RandomFS")
                # RandomFS_b
                plt.scatter(raw_data[dataset_id, 4, 6, fold_id], raw_data[dataset_id, 3, 6, fold_id], color='tab:purple', marker="<", label="RandomFS_b")

                # plt.title("Objective Space", fontsize=12)
                plt.xlabel('Precision', fontsize=12)
                plt.ylabel('Recall', fontsize=12)
                plt.xlim([0, 1.1])
                plt.ylim([0, 1.1])
                plt.legend(loc="best")
                plt.gcf().set_size_inches(9, 6)
                plt.savefig(filename_pareto_chart+".png", bbox_inches='tight')
                plt.savefig(filename_pareto_chart+".eps", format='eps', bbox_inches='tight')
                plt.clf()
                plt.close()


# Ensemble plot of ensemble size and G-mean
def ensemble_plot(datasets, n_folds, experiment_name, solution_dict, K_steps_optimization):
    # print(solution_dict)
    for dataset_id, dataset in enumerate(datasets):
        for fold_id in range(n_folds):
            plot_filename = "results/%s/ensemble_plots/%s/ens_plot_%s_fold%d" % (experiment_name, dataset, dataset, fold_id)
            if not os.path.exists("results/%s/ensemble_plots/%s/" % (experiment_name, dataset)):
                os.makedirs("results/%s/ensemble_plots/%s/" % (experiment_name, dataset))
            
            data_x = []
            data_y = []
            for ensemble_size in range(1, K_steps_optimization+1):
                data_x.append(ensemble_size)
                # wybierz jeden najlepszy ensemble z danego datasetu i folda według najwyższej metryki gmean i zapisz do listy data_y
                values = solution_dict[f"{dataset}"][f"{fold_id}"][f"K{ensemble_size}"].values()
                chosen_gmean = max(values, key=lambda item: item[1])[1]
                data_y.append(chosen_gmean)
            # print(data_x)
            # print(data_y)

            plt.scatter(data_x, data_y, color='tab:blue', marker="o", label="Ensemble")
            plt.title("Dependence of ensemble size on G-mean", fontsize=12)
            plt.xlabel('Ensemble size', fontsize=12)
            plt.ylabel('G-mean', fontsize=12)
            # plt.xlim([1, K_steps_optimization+1])
            plt.ylim([0, 1.1])
            plt.xticks(np.arange(0, K_steps_optimization+1, step=1))
            # plt.legend(loc="best")
            plt.grid(True, color="silver", linestyle=":", axis='both')
            plt.gcf().set_size_inches(9, 5)
            plt.savefig(plot_filename+".png", bbox_inches='tight')
            plt.savefig(plot_filename+".eps", format='eps', bbox_inches='tight')
            plt.clf()
            plt.close()


# Wykres, który zawiera wszystkie ensemble z ostatniej iteracji (K_steps_optimization), oś x - jedna metryka (gmean), oś y - druga metryka (hellinger)
def ensemble_plot_final_K(datasets, n_folds, experiment_name, solution_dict_final_K, K_steps_optimization, annotations):
    for dataset_id, dataset in enumerate(datasets):
        for fold_id in range(n_folds):
            plot_filename = "results/%s/ensemble_plots_final_K/%s/ens_plot_final_K_%s_fold%d" % (experiment_name, dataset, dataset, fold_id)
            if not os.path.exists("results/%s/ensemble_plots_final_K/%s/" % (experiment_name, dataset)):
                os.makedirs("results/%s/ensemble_plots_final_K/%s/" % (experiment_name, dataset))
            
            data_x = []
            data_y = []
            values = list(solution_dict_final_K[f"{dataset}"][f"{fold_id}"].values())
            # gmean
            data_x.append([i[1] for i in values])
            # hellinger
            data_y.append([i[0] for i in values])
            data_x = data_x[0]
            data_y = data_y[0]
            # print("X", data_x, "Y", data_y)

            plt.scatter(data_x, data_y, color='tab:blue', marker="o", label="Ensemble")

            # Loop for annotation of all points
            off = 0
            for i, xi, yi, text in zip(range(len(data_x)), data_x, data_y, annotations[f"{dataset}"][f"{fold_id}"]):
                # zaznacz na wykresie najlepszy ensemble, tylko pierwszy dopasowany
                if data_x.index(max(data_x)) == i:
                    plt.annotate(text,
                        xy=(xi, yi), xycoords='data',
                        xytext=(1.5, 1.5+off), textcoords='offset points',
                        color="tab:red")
                    off += -10.0
                # warunek gdy xi jest różne od yi
                else:
                    plt.annotate(text,
                        xy=(xi, yi), xycoords='data',
                        xytext=(1.5, 1.5+off), textcoords='offset points',
                        color="tab:blue")
                    off += -10.0
                
            plt.title("All ensembles", fontsize=12)
            plt.xlabel('G-mean', fontsize=12)
            plt.ylabel('Hellinger', fontsize=12)
            # plt.xlim([0, 1.1])
            # plt.ylim([0, 1.1])
            # plt.legend(loc="best")
            plt.grid(True, color="silver", linestyle=":", axis='both')
            plt.gcf().set_size_inches(9, 5)
            plt.savefig(plot_filename+".png", bbox_inches='tight')
            plt.savefig(plot_filename+".eps", format='eps', bbox_inches='tight')
            plt.clf()
            plt.close()


# Dla każdej metryki osobno zrób wykres jakości od ilości bootstrapów, uśrednienie po datasetach i foldach, z odchyleniem standardowym
def plot_bootstraps(mean_scores, metrics_alias, max_bootstraps_list, experiment_name):
    mean_scores_ds = np.mean(mean_scores, axis=0)
    stds_ds = np.std(mean_scores, axis=0)
    for metric_id, metric in enumerate(metrics_alias):
        fig, ax = plt.subplots()
        # mam 4 różne typy metod
        mean_scores_4methods = {}
        stds_4methods = {}
        for main_method_id, main_name in enumerate(("Margin_Diversity_Classifier", "Hellinger_Gmean_Classifier", "Margin_Diversity_Classifier_h", "Hellinger_Gmean_Classifier_h")):
            mean_scores_4methods[main_name] = []
            stds_4methods[main_name] = []
            for id in range(main_method_id, 32, 4):
                mean_scores_4methods[main_name].append(mean_scores_ds[metric_id, id])
                stds_4methods[main_name].append(stds_ds[metric_id, id])
        
        for main_method_id, main_name in enumerate(("Margin_Diversity_Classifier", "Hellinger_Gmean_Classifier", "Margin_Diversity_Classifier_h", "Hellinger_Gmean_Classifier_h")):
            ax.errorbar(max_bootstraps_list, mean_scores_4methods[main_name], stds_4methods[main_name], fmt='o', linewidth=2, capsize=6, label=main_name)
            ax.set(xlim=(0, 200), xticks=np.arange(0, 250, 25), ylim=(0, 1), yticks=np.arange(0, 1.2, 0.1))

            # Add legend to the plot
            ax.legend(loc='best')
            ax.grid(True, color="silver", linestyle=":", axis='both')

        plt.xlabel('Number of bootstraps', fontsize=12)
        plt.ylabel(f"{metric}", fontsize=12)

        if not os.path.exists("results/%s/bootstraps/" % (experiment_name)):
            os.makedirs("results/%s/bootstraps/" % (experiment_name))
        filepath = "results/%s/bootstraps/plot_bootstraps_%s" % (experiment_name, metric)
        plt.savefig(filepath + ".png", bbox_inches='tight')
        plt.savefig(filepath + ".eps", format='eps', bbox_inches='tight')
        plt.clf()
        plt.close()


# Dla każdej metryki osobno i dla każdego datasetu zrób wykres jakości od ilości bootstrapów, uśrednienie po foldach, z odchyleniem standardowym
def plot_bootstraps_for_dataset(mean_scores, stds, metrics_alias, max_bootstraps_list, experiment_name, datasets):
    # mean_scores_ds = np.mean(mean_scores, axis=0)
    # stds_ds = np.std(mean_scores, axis=0)
    for dataset_id, dataset in enumerate(datasets):
        for metric_id, metric in enumerate(metrics_alias):
            fig, ax = plt.subplots()
            # mam 4 różne typy metod
            mean_scores_4methods = {}
            stds_4methods = {}
            for main_method_id, main_name in enumerate(("Margin_Diversity_Classifier", "Hellinger_Gmean_Classifier", "Margin_Diversity_Classifier_h", "Hellinger_Gmean_Classifier_h")):
                mean_scores_4methods[main_name] = []
                stds_4methods[main_name] = []
                for id in range(main_method_id, 32, 4):
                    mean_scores_4methods[main_name].append(mean_scores[dataset_id, metric_id, id])
                    stds_4methods[main_name].append(stds[dataset_id, metric_id, id])
            
            for main_method_id, main_name in enumerate(("Margin_Diversity_Classifier", "Hellinger_Gmean_Classifier", "Margin_Diversity_Classifier_h", "Hellinger_Gmean_Classifier_h")):
                ax.errorbar(max_bootstraps_list, mean_scores_4methods[main_name], stds_4methods[main_name], fmt='o', linewidth=2, capsize=6, label=main_name)
                ax.set(xlim=(0, 200), xticks=np.arange(0, 250, 25), ylim=(0, 1), yticks=np.arange(0, 1.2, 0.1))

                # Add legend to the plot
                ax.legend(loc='best')
                ax.grid(True, color="silver", linestyle=":", axis='both')

            plt.xlabel('Number of bootstraps', fontsize=12)
            plt.ylabel(f"{metric}", fontsize=12)

            if not os.path.exists("results/%s/bootstraps_ds/%s/" % (experiment_name, dataset)):
                os.makedirs("results/%s/bootstraps_ds/%s/" % (experiment_name, dataset))
            filepath = "results/%s/bootstraps_ds/%s/plot_bootstraps_%s_%s" % (experiment_name, dataset, dataset, metric)
            plt.savefig(filepath + ".png", bbox_inches='tight')
            plt.savefig(filepath + ".eps", format='eps', bbox_inches='tight')
            plt.clf()
            plt.close()


# Zebranie wyników z plików .csv do jednego pliku .csv w formie tabeli, wiersze to datasety, a kolumny to uśrednione wyniki dla każdej metody, dla każdej metryki osobno. jest to potrzebne do importowania wyników do narzędzia KEEL do wygenerowania testów statystycznych (Friedman + post-hoc) i analizy
def collect_results_for_keel(dataset_names, metrics_alias, mean_scores, method_names, experiment_name):
    # print(mean_scores)
    for metric_id, metric_name in enumerate(metrics_alias):
        # print(metric_name)
        # print(mean_scores[:, metric_id, :])
        # print(mean_scores[:, metric_id, :].shape)

        mean_scores_df = pd.DataFrame(mean_scores[:, metric_id, :], columns=method_names, index=dataset_names)
        # print(mean_scores_df)

        filename = "results/%s/keel/tables/%s.csv" % (experiment_name, metric_name)
        if not os.path.exists("results/%s/keel/tables/" % (experiment_name)):
            os.makedirs("results/%s/keel/tables/" % (experiment_name))
        mean_scores_df.to_csv(filename, sep=',', encoding='utf-8', index=True, header=True)

# Calculate ranks for every method based on mean_scores; the higher the rank, the better the method
def calc_ranks(mean_scores, metric_id):
    ranks = []
    for ms in mean_scores[metric_id]:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    # print("\nRanks for", metric_a, ": ", ranks, "\n")
    mean_ranks = np.mean(ranks, axis=0)
    # print("\nMean ranks for", metric_a, ": ", mean_ranks, "\n")
    return(ranks, mean_ranks)


# Calculate Friedman statistics
def friedman_test(clf_names, mean_ranks, n_streams, critical_difference):
    N_ = n_streams
    k_ = len(clf_names)
    p_value = 0.05
    
    friedman = (12*N_/(k_*(k_+1)))*(np.sum(mean_ranks**2)-(k_*(k_+1)**2)/4)
    print("Friedman", friedman)
    iman_davenport = ((N_-1)*friedman)/(N_*(k_-1)-friedman)
    print("Iman-davenport", iman_davenport)
    f_dist = stats.f.ppf(1-p_value, k_-1, (k_-1)*(N_-1))
    print("F-distribution", f_dist)
    if f_dist < iman_davenport:
        print("Reject hypothesis H0")
    
    print("Critical difference", critical_difference)
    print(mean_ranks)

# Friedman test and post-hoc test Nemenyi - plots
def friedman_plot(metrics_alias, mean_scores, n_datasets, method_names, methods_alias, experiment_name):
    for metric_id, metric_a in enumerate(metrics_alias):
        ranks, mean_ranks = calc_ranks(mean_scores, metric_id)
        critical_difference = Orange.evaluation.compute_CD(mean_ranks, n_datasets, test='nemenyi')

        # Friedman test, implementation from Demsar2006
        print("\n", metric_a)
        friedman_test(method_names, mean_ranks, n_datasets, critical_difference)

        # CD diagrams to compare base classfiers with each other based on Nemenyi test (post-hoc)
        fnames = [('results/%s/keel/plot_ranks/cd_%s.png' % (experiment_name, metric_a)), ('results/%s/keel/plot_ranks/cd_%s.eps' % (experiment_name, metric_a))]
        if not os.path.exists('results/%s/keel/plot_ranks/' % experiment_name):
            os.makedirs('results/%s/keel/plot_ranks/' % experiment_name)
        for fname in fnames:
            Orange.evaluation.graph_ranks(mean_ranks, methods_alias, cd=critical_difference, width=7, textspace=1, filename=fname)



def result_tables_IR(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name):
    imbalance_ratios = []
    for dataset_path in dataset_paths:
        X, y = load_dataset(dataset_path)
        IR = calc_imbalance_ratio(X, y)
        imbalance_ratios.append(IR)
    IR_argsorted = np.argsort(imbalance_ratios)
    for metric_id, metric in enumerate(metrics_alias):
        if not os.path.exists("results/%s/tables_IR/" % experiment_name):
            os.makedirs("results/%s/tables_IR/" % experiment_name)
        with open("results/%s/tables_IR/results_%s_%s.tex" % (experiment_name, metric, experiment_name), "w+") as file:
            print("\\begin{table}[!ht]", file=file)
            print("\\centering", file=file)
            print("\\caption{%s}" % (metric), file=file)
            columns = "r"
            for i in methods:
                columns += " c"
            print("\\scalebox{0.4}{", file=file)
            print("\\begin{tabular}{%s}" % columns, file=file)
            print("\\hline", file=file)
            columns_names = "\\textbf{ID} &"
            for name in methods:
                name = name.replace("_", "-")
                columns_names += f'\\textbf{{{name}}} & '
            columns_names = columns_names[:-3]
            columns_names += "\\\\"
            print(columns_names, file=file)
            print("\\hline", file=file)
            for id, arg in enumerate(IR_argsorted):
                id += 1
                line = "%d" % (id)
                # lineir = "$%s$" % (dataset_paths[arg])
                # print(line, lineir)
                line_values = []
                line_values = mean_scores[arg, metric_id, :]
                max_value = np.amax(line_values)
                for clf_id, clf_name in enumerate(methods):
                    if mean_scores[arg, metric_id, clf_id] == max_value:
                        line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                    else:
                        line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                line += " \\\\"
                print(line, file=file)
            print("\\end{tabular}}", file=file)
            print("\\end{table}", file=file)

