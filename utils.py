import os
import numpy as np
import random
#import torch

import importlib
import inspect
import yaml
import pickle

def seed_everything(seed: int=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # Torch stuff
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = True
    #     torch.use_deterministic_algorithms(True)
    
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


def find_class(class_list, class_path):
    """
    class_list: List of classes to search in
    class_path: Path of the class to search for
    
    Returns the class if found, otherwise raises a ValueError
    """
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


def fetch_all_classes(module_path, class_type):
    """
    module_path: Path of the module to search in
    class_type: Type of class to search for
    
    Returns a list of classes of the specified type found in the module
    """
    class_format = ".py"

    classes_path = [file for file in os.listdir(module_path) if file.endswith(class_format)]
    
    classes = []
    for class_path in classes_path:
        # Remove the '.py' extension to get the module name
        class_name = class_path[:-3]

        # Import the module dynamically
        module = importlib.import_module(f'{module_path}.{class_name}')

        # Iterate through the items in the module
        for _, item in inspect.getmembers(module):
            # Check if it's a class and not an internal Python class
            if inspect.isclass(item) and issubclass(item, class_type) and item.__module__ == module.__name__:
                classes.append(item)
                break  # We only expect one dataset per file

    return classes


# Add this function to load the configuration from the YAML file
def load_experiment_config(config_file):
    """
    config_file: Path of the experiments.yaml file

    Returns a dictionnary of the experiments
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['experiments']


def save_results(dataset,
                pipe,
                scorer,
                dataset_params,
                estimator_params_list,
                scorer_params,
                scores):
    pipe_name = '-'.join(list(pipe.named_steps.keys()))
    
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
    print(f'Saving results to: {results_folder} \n\n')

    os.makedirs(results_folder, exist_ok=True)

    results_filename = os.path.join(results_folder, str(scorer) + '.pkl')
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)


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
