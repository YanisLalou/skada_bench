# This script is used to launch the experiment.
import argparse

from skada.model_selection import RandomShuffleDomainAwareSplit
from skada import make_da_pipeline
from skada.utils import source_target_split, source_target_merge

from utils import (seed_everything, fetch_all_classes,
                   save_results)
from base_bench_classes import BaseBenchClasses

from sklearn.model_selection import GridSearchCV
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

    scorers = {}
    for scorer_class in scorer_classes:
        scorer_instance = scorer_class()
        scorer_name = str(scorer_instance)
        scorers[scorer_name] = scorer_instance.get_scorer()



    for dataset_class in dataset_classes:
        dataset = dataset_class()

        X, y, sample_domain = dataset.pack_train()
        _, target_labels, _ = dataset.pack_test()

        # y_source, y_target = source_target_split(y, sample_domain=sample_domain)
        # target_labels, _ = source_target_merge(y_source, y_test, sample_domain = sample_domain)
        param_da = {'sample_domain': sample_domain, 'target_labels': target_labels}

        for estimator_class in estimator_classes:
            #estimator = estimator_class().set_fit_request(sample_domain=True)
            bench_estimator = estimator_class()


            cv = RandomShuffleDomainAwareSplit(test_size=0.3, random_state=0, n_splits=5)

            #scorer = accuracy_score
            #scorer = scorer_class().get_scorer()
            #scorer = SupervisedScorer()

            # For the case where the scorer requires the target labels
            # if 'target_labels' in inspect.signature(scorer._score).parameters:
            #     y_source, y_target = source_target_split(y, sample_domain=sample_domain)
            #     target_labels, _ = source_target_merge(y_source, y_target, sample_domain = sample_domain)
            #     params = {'sample_domain': sample_domain, 'target_labels': target_labels}
            # else:
            #     params = {'sample_domain': sample_domain}

            # NOT USEFUL WHEN WE DO A PACK_LODO()
            #y_source, y_target = source_target_split(y, sample_domain=sample_domain)
            #target_labels, _ = source_target_merge(y_source, y_target, sample_domain = sample_domain)
            # y_source, y_target = source_target_split(y, sample_domain=sample_domain)
            # target_labels, _ = source_target_merge(y_source, y_test, sample_domain = sample_domain)
            # param_da = {'sample_domain': sample_domain, 'target_labels': target_labels}

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
            
            #clf.fit(X, y, sample_domain=sample_domain)
            clf.fit(X, y, **param_da)

            best_params_per_scorer = get_best_params_per_scorer(clf)
            
            print(best_params_per_scorer)
  
            # Now we retrain the model with the best parameters
            for scorer in best_params_per_scorer.keys():
                best_params = best_params_per_scorer[scorer]
                pipe.set_params(**best_params)

                # To train the model
                X, y, sample_domain = dataset.pack_train()

                #TODO: Wont be accurate if we're using a GPU
                t0 = time.time()
                pipe.fit(X, y, **{'sample_domain': sample_domain})
                t1 = time.time()
                training_time = t1 - t0

                # To test the model
                X, y, sample_domain = dataset.pack_test()
                X_source, X_target, y_source, y_target = source_target_split(X, y, sample_domain=sample_domain)

                y_pred_source = pipe.predict(X_source)
                source_acc = accuracy_score(y_source, y_pred_source)
                source_f1 = f1_score(y_source, y_pred_source, average='weighted')

                
                t2 = time.time()
                y_pred_target = pipe.predict(X_target)
                t3 = time.time()
                prediction_time = t3 - t2
                target_acc = accuracy_score(y_target, y_pred_target)
                target_f1 = f1_score(y_target, y_pred_target, average='weighted')

                print('------------------------------------')
                print('Source Accuracy:', source_acc)
                print('Target Accuracy:', target_acc)
                print('Score:', clf.cv_results_['mean_test_' + scorer])
                
                
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



            
        

    #     save_results(dataset, pipe, scorer, dataset_params, estimator_params_list, scorer_params, scores)

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
