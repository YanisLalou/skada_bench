# This script is used to launch the experiment.
import argparse

from sklearn.model_selection import cross_validate
from skada.model_selection import LeaveOneDomainOut
from skada import make_da_pipeline

from utils import seed_everything, find_class, fetch_all_classes, load_experiment_config, save_results
from base_classes import BaseClasses


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datasets", help = "Directory containing the datasets", type = str, default = "datasets")
parser.add_argument("-e", "--estimators", help = "Directory containing the estimators", type = str, default = "estimators")
parser.add_argument("-s", "--scorers", help = "Directory containing the scorers", type = str, default = "scorers")
parser.add_argument("-r", "--rerun", help = "Rerun all the experiments", type = bool, default = False)
parser.add_argument("-c", "--config", help = "YAML file containing the experiment configuration", type = str, default = "experiments.yaml")


def launch_experiments(args):
    # Fetch the datasets
    dataset_classes = fetch_all_classes(args.datasets, BaseClasses.dataset.value)

    # Fetch the estimators
    estimator_classes = fetch_all_classes(args.estimators, BaseClasses.estimator.value)

    # Fetch the scoring functions
    scorer_classes = fetch_all_classes(args.scorers, BaseClasses.scorer.value)

    # Load the experiment configurations from the YAML file
    experiment_configs = load_experiment_config(args.config)


    for experiment in experiment_configs:
        dataset_class = find_class(dataset_classes, experiment['dataset']['name'])
        scorer_class = find_class(scorer_classes, experiment['scorer']['name'])

        
        estimator_classes_to_run = []
        # Iterate over multiple estimators in the experiment
        for estimator_path in experiment['estimators']:
            estimator_class = find_class(estimator_classes, estimator_path['name'])
            estimator_classes_to_run.append(estimator_class)

        seed_everything()

        dataset_params = experiment['dataset'].get('params')
        scorer_params = experiment['scorer'].get('params')
        estimator_params_list = [estimator.get('params') for estimator in experiment['estimators']]
    
        # Load the dataset
        if dataset_params is None:
            dataset_params = {}
        dataset = dataset_class(**dataset_params)

        # Load the estimators
        estimator_to_run = []
        for estimator_class, estimator_param in zip(estimator_classes_to_run, estimator_params_list):
            if estimator_param is None:
                estimator_param = {}
            estimator = estimator_class(**estimator_param)
            estimator_to_run.append(estimator)

        # Create the pipeline
        pipe = make_da_pipeline(
            *estimator_to_run
        )

        # Load the scorer
        if scorer_params is None:
            scorer_params = {}
        scorer = scorer_class(**scorer_params)

        # Launch the experiment with all combinations of parameters
        
        print("Launching experiment for dataset {}, estimators {} and scorer {}".format(
            dataset, estimator_to_run, scorer, 
        ))

        # Run the experiment
        X, y, sample_domain = dataset.pack_lodo()

        cv = LeaveOneDomainOut(max_n_splits=len(dataset.domain_names_), test_size=0.3, random_state=0)
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            params={'sample_domain': sample_domain},
            scoring=scorer,
        )

        save_results(dataset, pipe, scorer, dataset_params, estimator_params_list, scorer_params, scores)


if __name__ == "__main__":
    args = parser.parse_args()

    seed_everything()

    launch_experiments(args)
