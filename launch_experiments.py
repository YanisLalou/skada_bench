# This script is used to launch the experiment.

import os
import argparse
import pickle
from sklearn.model_selection import cross_validate
from skada.model_selection import LeaveOneDomainOut

parser = argparse.ArgumentParser()

# To chose the model

parser.add_argument("-d", "--datasets", help = "Directory containing the datasets", type = str, default = "datasets")
parser.add_argument("-e", "--estimators", help = "Directory containing the estimators", type = str, default = "estimators")
parser.add_argument("-s", "--scorers", help = "Directory containing the scorers", type = str, default = "scorers")
parser.add_argument("-r", "rerun", help = "Rerun all the experiments", type = bool, default = False)


def launch_experiments(args):
    # Fetch the datasets
    datasets = fetch_datasets(args.datasets)

    # Fetch the estimators
    estimators = fetch_estimators(args.estimators)

    # Fetch the scoring functions
    scorers = fetch_scorers(args.scorers)
    
    # TODO: Add a function to chech the results folder and see if the results are already there
    
    for dataset in datasets:
        for estimator in estimators:
            for scorer in scorers:
                # Launch the experiment    
                print("Launching experiment for dataset {}, estimator {} and scorer {}".format(dataset, estimator, scorer))
                X, y, sample_domain = dataset.pack_lodo()

                cv = LeaveOneDomainOut(max_n_splits=len(dataset.domain_names_), test_size=0.3, random_state=0)
                scores = cross_validate(
                    estimator,
                    X,
                    y,
                    cv=cv,
                    params={'sample_domain': sample_domain},
                    scoring=scorer,
                )

                # Save the results
                results = {
                    'dataset': str(dataset),
                    'estimator': str(estimator),
                    'scorer': str(scorer),
                    'scores': scores,
                }

                results_folder = os.path.join('results', str(dataset), str(estimator))
                os.makedirs(results_folder, exist_ok=True)

                results_filename = os.path.join(results_folder, str(scorer) + '.pkl')
                with open(results_filename, 'wb') as f:
                    pickle.dump(results, f)
    

def fetch_datasets(dataset_folder):
    dataset_format = ".py"
    
    datasets = [file for file in os.listdir(dataset_folder) if file.endswith(dataset_format)]
    
    return datasets


def fetch_estimators(estimator_folder):
    estimator_format = ".py"
    
    estimators = [file for file in os.listdir(estimator_folder) if file.endswith(estimator_format)]
    
    return estimators


def fetch_scorers(scorer_folder):
    scorer_folder = "scorers"
    scorer_format = ".py"
    
    scorers = [file for file in os.listdir(scorer_folder) if file.endswith(scorer_format)]
    
    return scorers


if __name__ == "__main__":
    args = parser.parse_args()
    launch_experiments(args)
