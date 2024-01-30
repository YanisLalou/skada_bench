# This script is used to launch the experiment.

import os
import argparse
import pickle
import itertools

import yaml

import importlib
import inspect

from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from skada.model_selection import LeaveOneDomainOut
from skada.datasets import DomainAwareDataset
from skada.metrics import _BaseDomainAwareScorer
from skada import make_da_pipeline

from utils import seed_everything

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

        pipe_name = '|'.join(list(pipe.named_steps.keys()))
        # Save the results
        results = {
            'dataset': {
                'name': str(dataset),
                'params': dataset_params,
            },
            'estimator': {
                'name': pipe_name,
                'params': estimator_params_list,
            },
            'scorer': {
                'name': str(scorer),
                'params': scorer_params,
            },
            'scores': scores,
        }

        results_folder = os.path.join('results', str(dataset), pipe_name)
        print('Saving results to: ')
        print(results_folder)
        print("\n\n\n")
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


# # Function to generate all combinations of parameters
# def generate_experiment_combinations(experiment_config):
#     dataset_params = experiment_config['dataset'].get('params')
#     scorer_params = experiment_config['scorer'].get('params')
#     estimator_params_list = [estimator.get('params') for estimator in experiment_config['estimators']]

#     # Remove None values from the parameters
#     estimator_params_list = [x for x in estimator_params_list if x is not None]
    
#     if dataset_params is None:
#         dataset_params = {}

#     if scorer_params is None:
#         scorer_params = {}

    
#     # Extraire les valeurs de dataset_params
#     dataset_values = list(itertools.product(*dataset_params.values()))

#     # Extraire les valeurs de estimator_params_list
#     estimator_values = list(itertools.product(*[params.values() for params in estimator_params_list]))

#     # Extraire les valeurs de scorer_params
#     scorer_values = list(itertools.product(*scorer_params.values()))

#     param_combinations = []

#     # Combinaison des dictionnaires
#     for dataset_value in dataset_values:
#         for estimator_value in estimator_values:
#             for scorer_value in scorer_values:
#                 param_combinations.append({
#                     'dataset': dict(zip(dataset_params.keys(), dataset_value)),
#                     'estimator': dict(zip(estimator_params_list[0].keys(), estimator_value)),
#                     'scorer': dict(zip(scorer_params.keys(), scorer_value)),
#                 })
    
#     # Create a list of dictionaries representing all combinations
#     import pdb; pdb.set_trace()

#     param_combinations = []
#     for combo in itertools.product(dataset_params.items(), scorer_params.items(), *estimator_params_list):
#         params_dict = dict(combo)
#         param_combinations.append(params_dict)

#     return param_combinations


# Add this function to load the configuration from the YAML file
def load_experiment_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['experiments']


if __name__ == "__main__":
    args = parser.parse_args()

    seed_everything()

    launch_experiments(args)
