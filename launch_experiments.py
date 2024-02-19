# This script is used to launch the experiment.
import argparse

from skada.model_selection import RandomShuffleDomainAwareSplit
from skada import make_da_pipeline
from skada.utils import source_target_split, source_target_merge

from utils import (seed_everything, fetch_all_classes,
                   save_results, generate_bibtext)
from base_bench_classes import BaseBenchClasses

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import time

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datasets", help = "Directory containing the datasets", type = str, default = "datasets")
parser.add_argument("-e", "--estimators", help = "Directory containing the estimators", type = str, default = "estimators")
parser.add_argument("-s", "--scorers", help = "Directory containing the scorers", type = str, default = "scorers")
parser.add_argument("-r", "--rerun", help = "Rerun all the experiments", type = bool, default = False)
parser.add_argument("-c", "--config", help = "YAML file containing the experiment configuration", type = str, default = "experiments.yaml")


def launch_experiments(args):
    # Fetch the datasets
    dataset_classes = fetch_all_classes(args.datasets, BaseBenchClasses.dataset.value)

    # Fetch the estimators
    estimator_classes = fetch_all_classes(args.estimators, BaseBenchClasses.estimator.value)

    # Fetch the scoring functions
    scorer_classes = fetch_all_classes(args.scorers, BaseBenchClasses.scorer.value)

    # Load the experiment configurations from the YAML file
    #experiment_configs = load_experiment_config(args.config)

    # Each error catched will be stored in this dict
    # and printed at the end of the experiment
    error_list = {}

    scorers = {}
    for scorer_class in scorer_classes:
        scorer_instance = scorer_class()
        scorer_name = str(scorer_instance)
        scorers[scorer_name] = scorer_instance.get_scorer()



    for dataset_class in dataset_classes:
        dataset = dataset_class()

        X, y, sample_domain = dataset.pack_train()
        _, target_labels, _ = dataset.pack_test()

        (X_train, X_test,
         y_train, y_test,
         target_labels_train, target_labels_test,
         sample_domain_train, sample_domain_test
        ) = train_test_split(X, y, target_labels, sample_domain, test_size=0.3)

        param_da_train = {'sample_domain': sample_domain_train, 'target_labels': target_labels_train}

        for estimator_class in estimator_classes:
            #estimator = estimator_class().set_fit_request(sample_domain=True)
            bench_estimator = estimator_class()


            cv = RandomShuffleDomainAwareSplit(test_size=0.3, random_state=0, n_splits=5)

            param_grid = bench_estimator.parameters
            print('Parameters:', param_grid)

            pipe = make_da_pipeline(
                dataset.get_prepocessor(),
                bench_estimator.get_base_estimator(),
                dataset.get_classifier()
            )
        
            clf = GridSearchCV(
                estimator = pipe,
                scoring=scorers,
                param_grid = param_grid,
                cv=cv,
                refit=False,
            )
            
            try:
                clf.fit(X_train, y_train, **param_da_train)
            except ValueError as e:
                print(f'Error: {e}')
                print(f'Dataset: {str(dataset)}, Estimator: {str(bench_estimator)}')
                print('Skipping...')
                error_list[str(dataset) + '-' + str(bench_estimator)] = e
                continue

            best_params_per_scorer = get_best_params_per_scorer(clf)
            print(best_params_per_scorer)
  
            # Now we retrain the model with the best parameters
            for scorer in best_params_per_scorer.keys():
                best_params = best_params_per_scorer[scorer]
                pipe.set_params(**best_params)

                # TRAIN THE MODEL
                #TODO: Wont be accurate if we're using a GPU
                t0 = time.time()
                pipe.fit(X_train, y_train, **{'sample_domain': sample_domain_train})
                t1 = time.time()
                training_time = t1 - t0

                # TEST THE MODEL
                (X_test_source, X_test_target,
                 target_labels_test_source, target_labels_test_target,
                 ) = source_target_split(X_test, target_labels_test, sample_domain=sample_domain_test)

                # Source accuracy
                y_pred_test_source = pipe.predict(X_test_source)
                source_acc = accuracy_score(target_labels_test_source, y_pred_test_source)
                source_f1 = f1_score(target_labels_test_source, y_pred_test_source, average='weighted')

                # Target accuracy
                y_pred_test_target = pipe.predict(X_test_target)
                target_acc = accuracy_score(target_labels_test_target, y_pred_test_target)
                target_f1 = f1_score(target_labels_test_target, y_pred_test_target, average='weighted')

                print('------------------------------------')
                print('Source Accuracy:', source_acc)
                print('Target Accuracy:', target_acc)
                print('Score:', clf.cv_results_['mean_test_' + scorer])

                # Compute the time to predict for one sample
                t2 = time.time()
                _ = pipe.predict(X_test_target[0:1])
                t3 = time.time()
                prediction_time = t3 - t2
                
                
                print('------------------------------------')
                print('\n\n\n')
                # Save results
                scores = {
                    'mean_test_score': clf.cv_results_['mean_test_' + scorer],
                    'std_test_score': clf.cv_results_['std_test_' + scorer],
                    'source_acc': source_acc,
                    'source_f1': source_f1,
                    'target_acc': target_acc,
                    'target_f1': target_f1,
                    'training_time': training_time,
                    'prediction_time': prediction_time,
                }
                
                save_results(dataset, bench_estimator, scorer, best_params, scores)

    generate_bibtext(dataset_classes + estimator_classes + scorer_classes)

    if error_list:
        print('The following errors were catched during the experiment:')
        for key, value in error_list.items():
            print(f'{key}: {value}')



            

def get_best_params_per_scorer(clf):
    best_params_per_scorer = {}
    for scorer_name in clf.scoring:
        best_index = np.argmax(clf.cv_results_['mean_test_' + scorer_name])
        best_params = clf.cv_results_['params'][best_index]
        best_params_per_scorer[scorer_name] = best_params
    return best_params_per_scorer




if __name__ == "__main__":
    args = parser.parse_args()

    seed_everything()

    launch_experiments(args)
