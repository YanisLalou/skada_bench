# This script is used to launch the experiment.

import os
import argparse
import pickle

import yaml

import importlib
import inspect

from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from skada.model_selection import LeaveOneDomainOut
from skada.datasets import DomainAwareDataset
from skada.metrics import _BaseDomainAwareScorer
from skada import make_da_pipeline

parser = argparse.ArgumentParser()


parser.add_argument("-d", "--datasets", help = "Directory containing the datasets", type = str, default = "datasets")
parser.add_argument("-e", "--estimators", help = "Directory containing the estimators", type = str, default = "estimators")
parser.add_argument("-s", "--scorers", help = "Directory containing the scorers", type = str, default = "scorers")
parser.add_argument("-r", "--rerun", help = "Rerun all the experiments", type = bool, default = False)
parser.add_argument("-c", "--config", help = "YAML file containing the experiment configuration", type = str, default = "experiments.yaml")


def launch_experiments(args):
    # Fetch the datasets
    dataset_classes = fetch_datasets(args.datasets)

    # Fetch the estimators
    estimator_classes = fetch_estimators(args.estimators)

    # Fetch the scoring functions
    scorer_classes = fetch_scorers(args.scorers)

    # Load the experiment configurations from the YAML file
    experiment_configs = load_experiment_config(args.config)

    for config in experiment_configs:
        dataset_class = find_class(dataset_classes, config['dataset'])
        scorer_class = find_class(scorer_classes, config['scorer'])
        
        estimator_classes_to_run = []
        # Iterate over multiple estimators in the experiment
        for estimator_path in config['estimators']:
            estimator_class = find_class(estimator_classes, estimator_path)
            estimator_classes_to_run.append(estimator_class)

        # Launch the experiment    
        print("Launching experiment for dataset {}, estimators {} and scorer {}".format(
            dataset_class, estimator_classes_to_run, scorer_class
        ))
        
        # Load the dataset
        dataset = dataset_class()

        # Load the estimators
        estimator = estimator_class()
        estimator_to_run = [estimator_class() for estimator_class in estimator_classes_to_run]

        # Create the pipeline
        pipe = make_da_pipeline(
            *estimator_to_run
        )

        # Load the scorer
        scorer = scorer_class()

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
    
    datasets_path = [file for file in os.listdir(dataset_folder) if file.endswith(dataset_format)]
    
    datasets = []
    for dataset_path in datasets_path:
        # Remove the '.py' extension to get the module name
        dataset_name = dataset_path[:-3]

        # Import the module dynamically
        module = importlib.import_module(f'{dataset_folder}.{dataset_name}')

        # Iterate through the items in the module
        for item_name, item in inspect.getmembers(module):
            # Check if it's a class and not an internal Python class
            if inspect.isclass(item) and issubclass(item, DomainAwareDataset) and item.__module__ == module.__name__:
                datasets.append(item)
                break  # We only expect one dataset per file

    return datasets


def fetch_estimators(estimator_folder):
    estimator_format = ".py"
    
    estimators_path = [file for file in os.listdir(estimator_folder) if file.endswith(estimator_format)]
    
    estimators = []
    for estimator_path in estimators_path:
        # Remove the '.py' extension to get the module name
        estimator_name = estimator_path[:-3]

        # Import the module dynamically
        module = importlib.import_module(f'{estimator_folder}.{estimator_name}')

        # Iterate through the items in the module
        for item_name, item in inspect.getmembers(module):
            # Check if it's a class and not an internal Python class
            if inspect.isclass(item) and issubclass(item, BaseEstimator) and item.__module__ == module.__name__:
                estimators.append(item)
                break  # We only expect one estimator per file
    
    return estimators


def fetch_scorers(scorer_folder):
    scorer_format = ".py"
    
    scorers_paths = [file for file in os.listdir(scorer_folder) if file.endswith(scorer_format)]
    
    scorers = []
    for scorers_path in scorers_paths:
        # Remove the '.py' extension to get the module name
        scorers_name = scorers_path[:-3]

        # Import the module dynamically
        module = importlib.import_module(f'{scorer_folder}.{scorers_name}')

        # Iterate through the items in the module
        for item_name, item in inspect.getmembers(module):
            # Check if it's a class and not an internal Python class
            if inspect.isclass(item) and issubclass(item, _BaseDomainAwareScorer) and item.__module__ == module.__name__:
                scorers.append(item)
                break  # We only expect one scorer per file
    
    return scorers


# Function to find a class by its path
def find_class(class_list, class_path):
    for class_item in class_list:
        if class_item.__module__ + '.' + class_item.__name__ == class_path:
            return class_item

    # If not found locally, try to import the class dynamically
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module_path_parts = module_path.split('.')
        module = importlib.import_module(module_path_parts[0])
        for part in module_path_parts[1:]:
            module = getattr(module, part)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        raise ValueError(f"Class not found for path: {class_path}")

# Add this function to load the configuration from the YAML file
def load_experiment_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['experiments']


if __name__ == "__main__":
    args = parser.parse_args()
    launch_experiments(args)
