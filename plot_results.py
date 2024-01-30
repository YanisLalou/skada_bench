# This script is used to plot the results of our experiment.
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Function to load results from pickle file
def load_results(results_filename):
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)
    return results


# Function to create scatter plots from the loaded results
def create_scatter_plots(results_list):
    plt.figure(figsize=(12, 8))

    experiment_names = []  # Store experiment names for x-axis

    for i, results in enumerate(results_list):
        # Extract relevant information from results
        dataset_name = results['dataset']['name']
        estimator_name = results['estimator']['name']
        scorer_name = results['scorer']['name']
        experiment_name = f'{dataset_name} - {estimator_name} - {scorer_name}'

        scores = results['scores']['test_score']

        # Calculate mean and std of test scores
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Plot mean score
        plt.scatter(i, mean_score, marker='o')

        # Plot error bar for standard deviation
        plt.errorbar(i, mean_score, yerr=std_score, capsize=5)

        experiment_names.append(experiment_name)

    plt.xlabel('Experiment')
    plt.ylabel('Score')
    plt.title('Mean and Standard Deviation of Test Scores')
    plt.xticks(range(len(experiment_names)), experiment_names, rotation=45, ha='right')
    plt.legend(fontsize='small')  # Adjust font size here
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
