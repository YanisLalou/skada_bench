# This script is used to plot the results of our experiment.
import os
import pickle
import matplotlib.pyplot as plt

# Function to load results from pickle file
def load_results(results_filename):
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)
    return results

# Function to create scatter plots from the loaded results
def create_scatter_plots(results_list):
    plt.figure(figsize=(12, 8))

    for results in results_list:
        # Extract relevant information from results
        dataset_name = results['dataset']
        estimator_name = results['estimator']
        scorer_name = results['scorer']
        scores = results['scores']

        # Scatter plot
        plt.scatter(range(len(scores['test_score'])), scores['test_score'], label=f'{dataset_name} - {estimator_name} - {scorer_name}')

    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Scatter Plots for Test Scores')
    plt.legend(fontsize='small')  # Adjust font size here
    plt.legend()
    plt.show()

# Root folder containing the dataset folders
results_root_folder = 'results'

# List to store results for creating scatter plots
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

# Create scatter plots
create_scatter_plots(results_list)
