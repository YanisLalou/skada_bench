# This script is used to generate the table of results of our experiment.
import os
import pickle
from tabulate import tabulate
import argparse

parser = argparse.ArgumentParser(description='Generate LaTeX table from pickle results.')
parser.add_argument('--display', action='store_true', help='Display the LaTeX table in the console.')
parser.add_argument('--save', metavar='output_file', help='Save the LaTeX table to a file.', default = 'result_table.tex')
parser.add_argument('--results', help="Directory containing the datasets", type = str, default = "results")

# Function to load results from pickle file
def load_results(results_filename):
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)
    return results

# Function to create a LaTeX table from the loaded results
def create_latex_table(args, results_list):
    headers = ['Dataset', 'Estimator', 'Scorer', 'Mean Score', 'Std Dev Score']

    data = []
    for results in results_list:
        # Extract relevant information from results
        dataset_name = results['dataset']
        estimator_name = results['estimator']
        scorer_name = results['scorer']
        scores = results['scores']['test_score']

        # Calculate mean and standard deviation of scores
        mean_score = round(scores.mean(), 3)
        std_dev_score = round(scores.std(), 3)

        data.append([dataset_name, estimator_name, scorer_name, mean_score, std_dev_score])

    # Generate LaTeX table
    if args.display:
        # Display LaTeX table in the console
        latex_table = tabulate(data, headers=headers, tablefmt='fancy_grid')
        print(latex_table)

    if args.save:
        # Save LaTeX table to a file
        latex_table = tabulate(data, headers=headers, tablefmt='latex_raw')
        with open(args.save, 'w') as f:
            f.write(latex_table)
    
        print(f'LaTeX table saved to {args.save}')


def get_results(args):
    # Root folder containing the dataset folders
    results_root_folder = args.results

    # List to store results for creating LaTeX table
    results_list = []

    # Loop through dataset folders
    for dataset_folder in os.listdir(results_root_folder):
        dataset_path = os.path.join(results_root_folder, dataset_folder)
        if os.path.isdir(dataset_path):
            # Loop through estimator folders
            for estimator_folder in os.listdir(dataset_path):
                estimator_path = os.path.join(dataset_path, estimator_folder)
                if os.path.isdir(estimator_path):
                    for file in os.listdir(estimator_path):
                        if file.endswith('.pkl'):
                            results_filename = os.path.join(estimator_path, file)
                            results = load_results(results_filename)
                            results_list.append(results)

    return results_list

if __name__ == "__main__":
    # Create LaTeX table
    args = parser.parse_args()

    results_list = get_results(args)
    create_latex_table(args, results_list)